"""
LIO (Learning to Incentivize Others) for SocialJax environments.

Implements the bilevel optimization from:
  "Learning to Incentivize Other Learning Agents" (Yang et al., NeurIPS 2020)

Supports all SocialJax environments (cleanup, coop_mining, coin_game, etc.)
via a single training script.  Environment-specific settings come from
the YAML config selected at launch:

  python lio_cnn.py --config-name lio_cnn_cleanup
  python lio_cnn.py --config-name lio_cnn_coop_mining
  python lio_cnn.py --config-name lio_cnn_coins

Each agent has:
  - Actor: policy network  (pi_j)
  - Critic: value network   (V_j)
  - Incentive: eta network   (eta_j) that outputs rewards for other agents

Training loop per update:
  1. Collect trajectory tau   with current actor params theta
  2. Inner update: theta' = theta - lr * grad(PG loss including incentive rewards)
  3. Collect trajectory tau'  with theta'
  4. Meta-gradient: update eta by differentiating through the inner gradient step
  5. Set theta = theta'
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import NamedTuple, Any
from flax.training.train_state import TrainState
import distrax
import socialjax
from socialjax.wrappers.baselines import LogWrapper
from socialjax.train_logging import (
    log_metrics_wandb_tensorboard,
    maybe_create_tensorboard_writer,
)
import hydra
from omegaconf import OmegaConf
import wandb
import pickle
import os
from pathlib import Path
from PIL import Image


# ======================= Networks =======================

class CNN(nn.Module):
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        act = nn.relu if self.activation == "relu" else nn.tanh
        x = nn.Conv(32, (5, 5), kernel_init=orthogonal(np.sqrt(2)),
                     bias_init=constant(0.0))(x)
        x = act(x)
        x = nn.Conv(32, (3, 3), kernel_init=orthogonal(np.sqrt(2)),
                     bias_init=constant(0.0))(x)
        x = act(x)
        x = nn.Conv(32, (3, 3), kernel_init=orthogonal(np.sqrt(2)),
                     bias_init=constant(0.0))(x)
        x = act(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)),
                      bias_init=constant(0.0))(x)
        x = act(x)
        return x


class Actor(nn.Module):
    action_dim: int
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        act = nn.relu if self.activation == "relu" else nn.tanh
        x = CNN(self.activation)(x)
        x = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)),
                      bias_init=constant(0.0))(x)
        x = act(x)
        x = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01),
                      bias_init=constant(0.0))(x)
        return distrax.Categorical(logits=x)


class Critic(nn.Module):
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        act = nn.relu if self.activation == "relu" else nn.tanh
        x = CNN(self.activation)(x)
        x = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)),
                      bias_init=constant(0.0))(x)
        x = act(x)
        x = nn.Dense(1, kernel_init=orthogonal(1.0),
                      bias_init=constant(0.0))(x)
        return jnp.squeeze(x, axis=-1)


class IncentiveNetwork(nn.Module):
    """Outputs per-agent incentive reward in [0, 1] via sigmoid."""
    num_agents: int
    activation: str = "relu"

    @nn.compact
    def __call__(self, obs, action_others_1hot):
        act = nn.relu if self.activation == "relu" else nn.tanh
        obs_embed = CNN(self.activation)(obs)
        x = jnp.concatenate([obs_embed, action_others_1hot], axis=-1)
        x = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)),
                      bias_init=constant(0.0))(x)
        x = act(x)
        x = nn.Dense(self.num_agents, kernel_init=orthogonal(0.01),
                      bias_init=constant(0.0))(x)
        return nn.sigmoid(x)


# ======================= Transition =======================

class Transition(NamedTuple):
    obs: jnp.ndarray           # (n_agents, n_envs, H, W, C)
    action: jnp.ndarray        # (n_agents, n_envs)
    reward: jnp.ndarray        # (n_agents, n_envs)  env reward
    done: jnp.ndarray          # (n_agents, n_envs)
    value: jnp.ndarray         # (n_agents, n_envs)
    log_prob: jnp.ndarray      # (n_agents, n_envs)
    r_from_others: jnp.ndarray # (n_agents, n_envs)  incentive received
    info: Any


# ======================= Helpers =======================

def get_action_others_1hot(actions_all, agent_id, action_dim, num_agents):
    """One-hot of other agents' actions.

    Args:
        actions_all: (batch, num_agents) int
    Returns: (batch, (num_agents-1)*action_dim)
    """
    mask = jnp.arange(num_agents) != agent_id
    others = actions_all[:, mask]
    return jax.nn.one_hot(others, action_dim).reshape(actions_all.shape[0], -1)


def compute_gae(rewards, values, dones, last_val, gamma, gae_lambda):
    """GAE advantage estimation.

    Args:
        rewards, values, dones: (T, B)
        last_val: (B,)
    Returns: advantages (T, B), targets (T, B)
    """
    def _step(carry, t):
        gae, nxt = carry
        r, v, d = t
        delta = r + gamma * nxt * (1 - d) - v
        gae = delta + gamma * gae_lambda * (1 - d) * gae
        return (gae, v), gae

    _, advantages = jax.lax.scan(
        _step,
        (jnp.zeros_like(last_val), last_val),
        (rewards, values, dones),
        reverse=True, unroll=16,
    )
    return advantages, advantages + values


def compute_returns(rewards, dones, gamma):
    """Simple discounted returns (for meta-loss weighting)."""
    def _step(carry, t):
        r, d = t
        ret = r + gamma * carry * (1 - d)
        return ret, ret

    _, returns = jax.lax.scan(
        _step, jnp.zeros(rewards.shape[1:]),
        (rewards, dones), reverse=True,
    )
    return returns


def _env_short_name(env_name: str) -> str:
    """Derive a short human-readable tag from ENV_NAME."""
    return {
        "clean_up": "cleanup",
        "coin_game": "coins",
        "harvest_common_open": "harvest",
        "territory_open": "territory",
    }.get(env_name, env_name)


# Per-environment metric key that should be scaled by num_inner_steps.
# If the env isn't listed here we simply skip the scaling.
_ENV_SCALE_METRIC = {
    "clean_up": "clean_action_info",
    "coop_mining": "mining_gold",
    "coin_game": "eat_own_coins",
    "gift": "give_actions",
    "mushrooms": "eat_blue_mushrooms",
}


# ======================= Training =======================

def make_train(config):
    env = socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    num_agents = env.num_agents
    action_dim = env.action_space().n
    obs_shape = env.observation_space()[0].shape

    config["NUM_UPDATES"] = int(
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"] // 2
    )

    env = LogWrapper(env, replace_info=False)
    tb_writer = maybe_create_tensorboard_writer(config)

    actor_net = Actor(action_dim, activation=config["ACTIVATION"])
    critic_net = Critic(activation=config["ACTIVATION"])
    incentive_net = IncentiveNetwork(num_agents, activation=config["ACTIVATION"])

    def train(rng):
        # ---------- init params ----------
        rng, *init_rngs = jax.random.split(rng, 1 + 3 * num_agents)
        dummy_obs = jnp.zeros((1, *obs_shape))
        dummy_act_others = jnp.zeros((1, (num_agents - 1) * action_dim))

        actor_params = [actor_net.init(init_rngs[3 * i], dummy_obs)
                        for i in range(num_agents)]
        critic_params = [critic_net.init(init_rngs[3 * i + 1], dummy_obs)
                         for i in range(num_agents)]
        incentive_params = [incentive_net.init(init_rngs[3 * i + 2],
                            dummy_obs, dummy_act_others)
                            for i in range(num_agents)]

        critic_tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(config["LR_CRITIC"], eps=1e-5),
        )
        incentive_tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(config["LR_INCENTIVE"], eps=1e-5),
        )
        critic_states = [TrainState.create(
            apply_fn=critic_net.apply, params=critic_params[i], tx=critic_tx)
            for i in range(num_agents)]
        incentive_states = [TrainState.create(
            apply_fn=incentive_net.apply, params=incentive_params[i],
            tx=incentive_tx) for i in range(num_agents)]

        # ---------- init env ----------
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset)(reset_rng)

        # ============================================================
        #  Rollout helper (captures actor/incentive/critic params via
        #  arguments so we can call it with different param sets)
        # ============================================================
        def run_rollout(a_params, i_params, c_params, env_st, last_obs, rng):
            """Collect NUM_STEPS transitions."""

            def _env_step(carry, _unused):
                env_st, last_obs, rng = carry
                rng, *agent_rngs, env_rng = jax.random.split(
                    rng, num_agents + 2)

                obs_all = jnp.transpose(last_obs, (1, 0, 2, 3, 4))

                actions, log_probs, values = [], [], []
                for i in range(num_agents):
                    pi_i = actor_net.apply(a_params[i], obs_all[i])
                    a_i = pi_i.sample(seed=agent_rngs[i])
                    actions.append(a_i)
                    log_probs.append(pi_i.log_prob(a_i))
                    values.append(critic_net.apply(c_params[i], obs_all[i]))

                actions_stack = jnp.stack(actions)
                actions_t = actions_stack.T

                r_from_others = jnp.zeros((num_agents, config["NUM_ENVS"]))
                for k in range(num_agents):
                    a_others_k = get_action_others_1hot(
                        actions_t, k, action_dim, num_agents)
                    eta_k = incentive_net.apply(
                        i_params[k], obs_all[k], a_others_k)
                    eta_k = eta_k * config["R_MULTIPLIER"]
                    mask_k = 1.0 - jax.nn.one_hot(k, num_agents)
                    eta_k = eta_k * mask_k[None, :]
                    r_from_others = r_from_others + eta_k.T

                env_act = [actions[i] for i in range(num_agents)]
                rng_step = jax.random.split(env_rng, config["NUM_ENVS"])
                new_obs, env_st, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, env_st, env_act)

                transition = Transition(
                    obs=obs_all,
                    action=actions_stack,
                    reward=jnp.stack(
                        [reward[:, i] for i in range(num_agents)]),
                    done=jnp.stack(
                        [done[str(a)] for a in env.agents]),
                    value=jnp.stack(values),
                    log_prob=jnp.stack(log_probs),
                    r_from_others=r_from_others,
                    info=info,
                )
                return (env_st, new_obs, rng), transition

            (env_st, last_obs, rng), traj = jax.lax.scan(
                _env_step, (env_st, last_obs, rng), None, config["NUM_STEPS"])
            return traj, env_st, last_obs, rng

        # ============================================================
        #  Main update step
        # ============================================================
        def _update_step(runner_state, _unused):
            (a_params, c_states, i_states,
             env_st, last_obs, update_step, rng) = runner_state

            c_params = [s.params for s in c_states]
            i_params = [s.params for s in i_states]

            # ====== Phase 1: collect trajectory with current policy ======
            traj1, env_st, last_obs, rng = run_rollout(
                a_params, i_params, c_params, env_st, last_obs, rng)

            # ====== Phase 2: inner policy + critic update ======
            a_prime = []
            for j in range(num_agents):
                obs_j = traj1.obs[:, j]
                act_j = traj1.action[:, j]
                rew_j = traj1.reward[:, j]
                done_j = traj1.done[:, j]
                val_j = traj1.value[:, j]
                r_inc_j = traj1.r_from_others[:, j]

                total_rew_j = rew_j + r_inc_j

                last_obs_j = last_obs[:, j]
                last_val_j = critic_net.apply(c_states[j].params, last_obs_j)

                adv_j, targets_j = compute_gae(
                    total_rew_j, val_j, done_j, last_val_j,
                    config["GAMMA"], config["GAE_LAMBDA"])

                obs_flat = obs_j.reshape(-1, *obs_shape)
                tgt_flat = targets_j.reshape(-1)

                def _critic_loss(cp, o, t):
                    return 0.5 * jnp.mean(jnp.square(
                        critic_net.apply(cp, o) - t))

                c_grad = jax.grad(_critic_loss)(
                    c_states[j].params, obs_flat, tgt_flat)
                c_states[j] = c_states[j].apply_gradients(grads=c_grad)

                adv_flat = adv_j.reshape(-1)
                act_flat = act_j.reshape(-1)

                def _actor_loss(ap, o, a, adv):
                    pi = actor_net.apply(ap, o)
                    lp = pi.log_prob(a)
                    ent = pi.entropy()
                    return -(jnp.mean(lp * jax.lax.stop_gradient(adv))
                             + config["ENT_COEF"] * jnp.mean(ent))

                a_grad = jax.grad(_actor_loss)(
                    a_params[j], obs_flat, act_flat, adv_flat)
                a_prime_j = jax.tree.map(
                    lambda p, g: p - config["LR_ACTOR"] * g,
                    a_params[j], a_grad)
                a_prime.append(a_prime_j)

            # ====== Phase 3: collect trajectory with prime policy ======
            c_params_new = [s.params for s in c_states]
            traj2, env_st, last_obs, rng = run_rollout(
                a_prime, i_params, c_params_new, env_st, last_obs, rng)

            # ====== Phase 4: meta-gradient for incentive networks ======
            actions_t1_flat = jnp.transpose(
                traj1.action, (0, 2, 1)).reshape(-1, num_agents)

            for i in range(num_agents):

                def _meta_loss(eta_i):
                    """Loss whose gradient w.r.t. eta_i is the LIO
                    meta-gradient for agent i.

                    Gradient chain:
                      eta_i  -->  incentive_reward
                             -->  total_reward_j
                             -->  returns_j  (via compute_returns, differentiable)
                             -->  adv_j
                             -->  inner_grad  (jax.grad w.r.t. actor_params;
                                              adv_j acts as coefficient,
                                              NOT stop-gradiented here so
                                              d(inner_grad)/d(eta_i) != 0)
                             -->  theta_hat_j = theta_j - lr * inner_grad
                             -->  pi_hat_j(theta_hat_j)
                             -->  outer_loss

                    jax.grad(_meta_loss) then computes d(outer_loss)/d(eta_i)
                    which involves the second-order Hessian-vector product
                    through the inner jax.grad call.  JAX handles this
                    natively via its higher-order AD.
                    """

                    loss = 0.0

                    ret_i = compute_returns(
                        traj2.reward[:, i], traj2.done[:, i],
                        config["GAMMA"])
                    ret_i_flat = jax.lax.stop_gradient(ret_i.reshape(-1))

                    for j in range(num_agents):
                        if j == i:
                            continue

                        inc_j_flat = jnp.zeros(
                            config["NUM_STEPS"] * config["NUM_ENVS"])

                        for k in range(num_agents):
                            if k == j:
                                continue
                            obs_k_flat = traj1.obs[:, k].reshape(
                                -1, *obs_shape)
                            ao_k = get_action_others_1hot(
                                actions_t1_flat, k, action_dim, num_agents)
                            eta_pk = eta_i if k == i else i_params[k]
                            eo = incentive_net.apply(
                                eta_pk, obs_k_flat, ao_k)
                            mask_k = 1.0 - jax.nn.one_hot(k, num_agents)
                            eo = eo * mask_k[None, :]
                            inc_j_flat = inc_j_flat + (
                                eo[:, j] * config["R_MULTIPLIER"])

                        inc_j = inc_j_flat.reshape(
                            config["NUM_STEPS"], config["NUM_ENVS"])
                        total_rj = traj1.reward[:, j] + inc_j
                        returns_j = compute_returns(
                            total_rj, traj1.done[:, j], config["GAMMA"])
                        returns_j_flat = returns_j.reshape(-1)

                        val_j_sg = jax.lax.stop_gradient(
                            traj1.value[:, j].reshape(-1))
                        adv_j = returns_j_flat - val_j_sg

                        obs_j_flat = traj1.obs[:, j].reshape(
                            -1, *obs_shape)
                        act_j_flat = traj1.action[:, j].reshape(-1)

                        def _inner(ap, o, a, adv):
                            pi = actor_net.apply(ap, o)
                            lp = pi.log_prob(a)
                            ent = pi.entropy()
                            return -(jnp.mean(lp * adv)
                                     + config["ENT_COEF"]
                                     * jnp.mean(ent))

                        ig = jax.grad(_inner)(
                            a_params[j], obs_j_flat, act_j_flat, adv_j)
                        theta_hat_j = jax.tree.map(
                            lambda p, g: p - config["LR_ACTOR"] * g,
                            a_params[j], ig)

                        obs_jn = traj2.obs[:, j].reshape(-1, *obs_shape)
                        act_jn = traj2.action[:, j].reshape(-1)
                        pi_hat = actor_net.apply(theta_hat_j, obs_jn)
                        lp_hat = pi_hat.log_prob(act_jn)

                        loss = loss + (-jnp.mean(lp_hat * ret_i_flat))

                    obs_i_flat = traj1.obs[:, i].reshape(-1, *obs_shape)
                    ao_i = get_action_others_1hot(
                        actions_t1_flat, i, action_dim, num_agents)
                    eo_i = incentive_net.apply(eta_i, obs_i_flat, ao_i)
                    self_mask = 1.0 - jax.nn.one_hot(i, num_agents)
                    if config["REG"] == "l1":
                        reg = jnp.mean(jnp.abs(eo_i * self_mask[None, :]))
                    else:
                        reg = jnp.mean(jnp.square(
                            eo_i * self_mask[None, :]))
                    loss = loss + config["REG_COEFF"] * reg
                    return loss

                mg = jax.grad(_meta_loss)(i_states[i].params)
                i_states[i] = i_states[i].apply_gradients(grads=mg)

            # ====== Phase 5: copy prime -> main ======
            a_params = a_prime

            # ====== logging ======
            update_step = update_step + 1
            metric = jax.tree.map(lambda x: x.mean(), traj1.info)
            metric["update_step"] = update_step
            metric["env_step"] = (
                update_step * config["NUM_STEPS"]
                * config["NUM_ENVS"] * 2)

            scale_key = _ENV_SCALE_METRIC.get(config["ENV_NAME"])
            if scale_key is not None and scale_key in metric:
                metric[scale_key] = (
                    metric[scale_key]
                    * config["ENV_KWARGS"]["num_inner_steps"])

            avg_inc = 0.0
            for j in range(num_agents):
                avg_inc = avg_inc + traj1.r_from_others[:, j].mean()
            metric["avg_incentive_received"] = avg_inc / num_agents
            metric["avg_env_reward"] = traj1.reward.mean()

            def _log_metrics(m):
                log_metrics_wandb_tensorboard(m, tb_writer)

            jax.debug.callback(_log_metrics, metric)

            runner_state = (a_params, c_states, i_states,
                            env_st, last_obs, update_step, rng)
            return runner_state, metric

        # ============================================================
        #  Kick off training
        # ============================================================
        rng, _rng = jax.random.split(rng)
        runner_state = (actor_params, critic_states, incentive_states,
                        env_state, obsv, 0, _rng)
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"])
        return {"runner_state": runner_state, "metrics": metrics}

    return train


# ======================= Entry points =======================

def save_params(params, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(jax.tree.map(lambda x: np.array(x), params), f)


def load_params(path):
    with open(path, "rb") as f:
        return jax.tree.map(lambda x: jnp.array(x), pickle.load(f))


def evaluate(actor_params_list, env, config):
    rng = jax.random.PRNGKey(0)
    rng, _rng = jax.random.split(rng)
    obs, state = env.reset(_rng)

    actor_net = Actor(env.action_space().n, activation=config["ACTIVATION"])
    pics = [env.render(state)]

    short = _env_short_name(config["ENV_NAME"])
    root_dir = f"evaluation/lio_{short}"
    Path(root_dir).mkdir(parents=True, exist_ok=True)

    for t in range(config["GIF_NUM_FRAMES"]):
        obs_batch = jnp.stack([obs[a] for a in env.agents])
        env_act = {}
        for i in range(env.num_agents):
            o = jnp.expand_dims(obs_batch[i], 0)
            pi = actor_net.apply(actor_params_list[i], o)
            rng, _rng = jax.random.split(rng)
            env_act[env.agents[i]] = pi.sample(seed=_rng)

        rng, _rng = jax.random.split(rng)
        obs, state, reward, done, info = env.step(
            _rng, state, [v.item() for v in env_act.values()])
        pics.append(env.render(state))

    imgs = [Image.fromarray(np.array(p)) for p in pics]
    gif_path = f"{root_dir}/{env.num_agents}-agents_seed-{config['SEED']}.gif"
    imgs[0].save(gif_path, format="GIF", save_all=True,
                 append_images=imgs[1:], duration=200, loop=0)
    wandb.log({"Episode GIF": wandb.Video(
        gif_path, caption="LIO Evaluation", format="gif")})


def single_run(config):
    config = OmegaConf.to_container(config)
    short = _env_short_name(config["ENV_NAME"])

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["LIO", "CNN", short],
        config=config,
        mode=config["WANDB_MODE"],
        name=f"lio_cnn_{short}",
    )

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    train_jit = jax.jit(make_train(config))
    out = jax.vmap(train_jit)(rngs)

    print("** Saving Results **")
    fname = f'{config["ENV_NAME"]}_lio_seed{config["SEED"]}'
    actor_params = jax.tree.map(lambda x: x[0], out["runner_state"][0])
    for i in range(config["ENV_KWARGS"]["num_agents"]):
        save_params(actor_params[i],
                     f"./checkpoints/lio/{fname}_actor_{i}.pkl")

    loaded = [load_params(f"./checkpoints/lio/{fname}_actor_{i}.pkl")
              for i in range(config["ENV_KWARGS"]["num_agents"])]
    evaluate(loaded,
             socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"]),
             config)


def tune(default_config):
    import copy as _copy
    default_config = OmegaConf.to_container(default_config)
    short = _env_short_name(default_config["ENV_NAME"])

    sweep_config = {
        "name": f"lio_{short}",
        "method": "grid",
        "metric": {"name": "avg_env_reward", "goal": "maximize"},
        "parameters": {
            "SEED": {"values": [42, 52, 62]},
        },
    }

    def _wrapped():
        wandb.init(project=default_config["PROJECT"])
        cfg = _copy.deepcopy(default_config)
        for k, v in dict(wandb.config).items():
            if "." in k:
                p, c = k.split(".", 1)
                cfg[p][c] = v
            else:
                cfg[k] = v
        wandb.run.name = f"sweep_lio_{cfg['ENV_NAME']}_seed{cfg['SEED']}"
        rng = jax.random.PRNGKey(cfg["SEED"])
        rngs = jax.random.split(rng, cfg["NUM_SEEDS"])
        jax.block_until_ready(jax.jit(jax.vmap(make_train(cfg)))(rngs))

    wandb.login()
    sid = wandb.sweep(sweep_config, entity=default_config["ENTITY"],
                      project=default_config["PROJECT"])
    wandb.agent(sid, _wrapped, count=1000)


@hydra.main(version_base=None, config_path="config",
            config_name="lio_cnn_cleanup")
def main(config):
    if config.get("TUNE", False):
        tune(config)
    else:
        single_run(config)


if __name__ == "__main__":
    main()
