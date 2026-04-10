"""
LOLA (Learning with Opponent-Learning Awareness) for SocialJax environments.

Implements the second-order policy gradient correction from:
  "Learning with Opponent-Learning Awareness" (Foerster et al., AAMAS 2018)

Supports all SocialJax environments via a single training script:

  python lola_cnn.py --config-name lola_cnn_cleanup
  python lola_cnn.py --config-name lola_cnn_coins

Core idea:
  Standard (Naive Learner):  θᵢ ← θᵢ + α·∇_{θᵢ} Vᵢ

  LOLA adds a second-order correction that accounts for opponents' learning:

    θᵢ ← θᵢ + α·[ ∇_{θᵢ} Vᵢ  +  η·∇_{θᵢ}(sg(∇_{θⱼ}Vᵢ)ᵀ · ∇_{θⱼ}Vⱼ) ]
                   ──────────     ────────────────────────────────────────
                   standard PG              LOLA correction

  The correction answers: "how does my θᵢ affect opponent j's gradient,
  and through j's update, how does that change MY value Vᵢ?"

  ∇_{θⱼ}Vⱼ must be differentiable w.r.t. θᵢ, so we use a "cross" value
  estimator built from products of both agents' cumulative log-probs.
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


# ======================= Transition =======================

class Transition(NamedTuple):
    obs: jnp.ndarray      # (n_agents, n_envs, H, W, C)
    action: jnp.ndarray   # (n_agents, n_envs)
    reward: jnp.ndarray   # (n_agents, n_envs)
    done: jnp.ndarray     # (n_agents, n_envs)
    value: jnp.ndarray    # (n_agents, n_envs)
    log_prob: jnp.ndarray # (n_agents, n_envs)
    info: Any


# ======================= Helpers =======================

def compute_gae(rewards, values, dones, last_val, gamma, gae_lambda):
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


def _tree_dot(tree1, tree2):
    """Dot product of two pytrees with identical structure."""
    leaves1 = jax.tree_util.tree_leaves(tree1)
    leaves2 = jax.tree_util.tree_leaves(tree2)
    return sum(jnp.sum(l1 * l2) for l1, l2 in zip(leaves1, leaves2))


def _env_short_name(env_name: str) -> str:
    return {
        "clean_up": "cleanup",
        "coin_game": "coins",
        "harvest_common_open": "harvest",
        "territory_open": "territory",
    }.get(env_name, env_name)


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
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )

    env = LogWrapper(env, replace_info=False)

    actor_net = Actor(action_dim, activation=config["ACTIVATION"])
    critic_net = Critic(activation=config["ACTIVATION"])

    gamma_disc = config["GAMMA"] ** jnp.arange(config["NUM_STEPS"])

    def train(rng):
        # ---------- init params ----------
        rng, *init_rngs = jax.random.split(rng, 1 + 2 * num_agents)
        dummy_obs = jnp.zeros((1, *obs_shape))

        actor_params = [actor_net.init(init_rngs[2 * i], dummy_obs)
                        for i in range(num_agents)]
        critic_params = [critic_net.init(init_rngs[2 * i + 1], dummy_obs)
                         for i in range(num_agents)]

        critic_tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(config["LR_CRITIC"], eps=1e-5),
        )
        critic_states = [TrainState.create(
            apply_fn=critic_net.apply, params=critic_params[i], tx=critic_tx)
            for i in range(num_agents)]

        # ---------- init env ----------
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset)(reset_rng)

        # ============================================================
        #  Rollout helper
        # ============================================================
        def run_rollout(a_params, c_params, env_st, last_obs, rng):
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
                    info=info,
                )
                return (env_st, new_obs, rng), transition

            (env_st, last_obs, rng), traj = jax.lax.scan(
                _env_step, (env_st, last_obs, rng), None,
                config["NUM_STEPS"])
            return traj, env_st, last_obs, rng

        # ============================================================
        #  Main update step
        # ============================================================
        def _update_step(runner_state, _unused):
            (a_params, c_states,
             env_st, last_obs, update_step, rng) = runner_state

            c_params = [s.params for s in c_states]

            # ====== Phase 1: collect trajectory ======
            traj, env_st, last_obs, rng = run_rollout(
                a_params, c_params, env_st, last_obs, rng)

            # ====== Phase 2: advantages, critic update, sample rewards ======
            advs_list = []
            sample_rews = []

            for j in range(num_agents):
                obs_j = traj.obs[:, j]
                rew_j = traj.reward[:, j]
                done_j = traj.done[:, j]
                val_j = traj.value[:, j]

                last_obs_j = last_obs[:, j]
                last_val_j = critic_net.apply(
                    c_states[j].params, last_obs_j)

                adv_j, targets_j = compute_gae(
                    rew_j, val_j, done_j, last_val_j,
                    config["GAMMA"], config["GAE_LAMBDA"])
                advs_list.append(adv_j)

                obs_flat = obs_j.reshape(-1, *obs_shape)
                tgt_flat = targets_j.reshape(-1)

                def _critic_loss(cp, o, t):
                    return 0.5 * jnp.mean(jnp.square(
                        critic_net.apply(cp, o) - t))

                c_grad = jax.grad(_critic_loss)(
                    c_states[j].params, obs_flat, tgt_flat)
                c_states[j] = c_states[j].apply_gradients(grads=c_grad)

                # Centered + discounted per-step reward for cross-value
                centered = rew_j - rew_j.mean()
                sample_rews.append(
                    centered * gamma_disc[:, None])  # (T, E)

            # ====== Phase 3: LOLA actor update ======
            #
            # Compute ALL deltas first using the ORIGINAL a_params,
            # then apply updates simultaneously (matches original LOLA
            # where both agents' deltas are computed before any update).

            a_params_new = []

            for i in range(num_agents):
                obs_i_flat = traj.obs[:, i].reshape(-1, *obs_shape)
                acts_i_flat = traj.action[:, i].reshape(-1)
                advs_i_flat = advs_list[i].reshape(-1)

                # --- Standard PG gradient ---
                def _pg_loss(ap, o, a, adv):
                    pi = actor_net.apply(ap, o)
                    lp = pi.log_prob(a)
                    ent = pi.entropy()
                    return -(jnp.mean(
                        lp * jax.lax.stop_gradient(adv))
                        + config["ENT_COEF"] * jnp.mean(ent))

                pg_grad = jax.grad(_pg_loss)(
                    a_params[i], obs_i_flat, acts_i_flat, advs_i_flat)

                if config["CORRECTIONS"]:
                    # Accumulate corrections from all opponents
                    total_corr = jax.tree.map(jnp.zeros_like, pg_grad)

                    for j in range(num_agents):
                        if j == i:
                            continue

                        obs_j_flat = traj.obs[:, j].reshape(
                            -1, *obs_shape)
                        acts_j_flat = traj.action[:, j].reshape(-1)
                        srew_j = sample_rews[j]  # (T, E)
                        T = config["NUM_STEPS"]
                        E = config["NUM_ENVS"]

                        # ∇_{θⱼ} Vᵢ  (cross-gradient: how j's
                        # policy affects i's value, via REINFORCE
                        # through π_j with agent i's advantages)
                        def _vi_via_j(tj):
                            lp_j = actor_net.apply(
                                tj, obs_j_flat).log_prob(acts_j_flat)
                            return jnp.mean(
                                lp_j * jax.lax.stop_gradient(
                                    advs_i_flat))

                        cross_grad_sg = jax.lax.stop_gradient(
                            jax.grad(_vi_via_j)(a_params[j]))

                        # LOLA correction:
                        # ∇_{θᵢ}[ sg(∇_{θⱼ}Vᵢ)ᵀ · ∇_{θⱼ}Vⱼ^{cross} ]
                        #
                        # _multiply(θᵢ) returns a scalar whose
                        # gradient w.r.t. θᵢ is the correction.
                        # Inside, we compute ∇_{θⱼ} of the
                        # cross-value Vⱼ (which uses products of
                        # both agents' cumulative log-probs, making
                        # it differentiable w.r.t. θᵢ through
                        # log πᵢ).  This requires second-order AD
                        # (nested jax.grad), which JAX handles
                        # natively.
                        def _multiply(ti):
                            def _vj_cross(ti_, tj_):
                                lp_i = actor_net.apply(
                                    ti_, obs_i_flat
                                ).log_prob(acts_i_flat).reshape(T, E)
                                lp_j = actor_net.apply(
                                    tj_, obs_j_flat
                                ).log_prob(acts_j_flat).reshape(T, E)
                                cum_i = jnp.cumsum(lp_i, axis=0)
                                cum_j = jnp.cumsum(lp_j, axis=0)
                                return 2.0 * jnp.sum(
                                    cum_i * cum_j * srew_j) / E

                            grad_j_vj = jax.grad(
                                _vj_cross, argnums=1)(
                                    ti, a_params[j])
                            return _tree_dot(
                                cross_grad_sg, grad_j_vj)

                        corr_ij = jax.grad(_multiply)(a_params[i])
                        total_corr = jax.tree.map(
                            jnp.add, total_corr, corr_ij)

                    # θ ← θ - lr*(pg_grad - lr_corr*correction)
                    #   = θ + lr*PG + lr*lr_corr*correction
                    a_params_new.append(jax.tree.map(
                        lambda p, g, c: p - config["LR_ACTOR"] * (
                            g - config["LR_CORRECTION"] * c),
                        a_params[i], pg_grad, total_corr))
                else:
                    # Naive learner: standard PG only
                    a_params_new.append(jax.tree.map(
                        lambda p, g: p - config["LR_ACTOR"] * g,
                        a_params[i], pg_grad))

            a_params = a_params_new

            # ====== Logging ======
            update_step = update_step + 1
            metric = jax.tree.map(lambda x: x.mean(), traj.info)
            metric["update_step"] = update_step
            metric["env_step"] = (
                update_step * config["NUM_STEPS"] * config["NUM_ENVS"])

            scale_key = _ENV_SCALE_METRIC.get(config["ENV_NAME"])
            if scale_key is not None and scale_key in metric:
                metric[scale_key] = (
                    metric[scale_key]
                    * config["ENV_KWARGS"]["num_inner_steps"])

            metric["avg_env_reward"] = traj.reward.mean()

            jax.debug.callback(lambda m: wandb.log(m), metric)

            runner_state = (a_params, c_states,
                            env_st, last_obs, update_step, rng)
            return runner_state, metric

        # ============================================================
        #  Kick off training
        # ============================================================
        rng, _rng = jax.random.split(rng)
        runner_state = (actor_params, critic_states,
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

    actor_net_eval = Actor(
        env.action_space().n, activation=config["ACTIVATION"])
    pics = [env.render(state)]

    short = _env_short_name(config["ENV_NAME"])
    root_dir = f"evaluation/lola_{short}"
    Path(root_dir).mkdir(parents=True, exist_ok=True)

    for t in range(config["GIF_NUM_FRAMES"]):
        obs_batch = jnp.stack([obs[a] for a in env.agents])
        env_act = {}
        for i in range(env.num_agents):
            o = jnp.expand_dims(obs_batch[i], 0)
            pi = actor_net_eval.apply(actor_params_list[i], o)
            rng, _rng = jax.random.split(rng)
            env_act[env.agents[i]] = pi.sample(seed=_rng)

        rng, _rng = jax.random.split(rng)
        obs, state, reward, done, info = env.step(
            _rng, state, [v.item() for v in env_act.values()])
        pics.append(env.render(state))

    imgs = [Image.fromarray(np.array(p)) for p in pics]
    gif_path = (f"{root_dir}/{env.num_agents}-agents"
                f"_seed-{config['SEED']}.gif")
    imgs[0].save(gif_path, format="GIF", save_all=True,
                 append_images=imgs[1:], duration=200, loop=0)
    wandb.log({"Episode GIF": wandb.Video(
        gif_path, caption="LOLA Evaluation", format="gif")})


def single_run(config):
    config = OmegaConf.to_container(config)
    short = _env_short_name(config["ENV_NAME"])

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["LOLA", "CNN", short],
        config=config,
        mode=config["WANDB_MODE"],
        name=f"lola_cnn_{short}",
    )

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    train_jit = jax.jit(make_train(config))
    out = jax.vmap(train_jit)(rngs)

    print("** Saving Results **")
    fname = f'{config["ENV_NAME"]}_lola_seed{config["SEED"]}'
    actor_params = jax.tree.map(
        lambda x: x[0], out["runner_state"][0])
    for i in range(config["ENV_KWARGS"]["num_agents"]):
        save_params(actor_params[i],
                     f"./checkpoints/lola/{fname}_actor_{i}.pkl")

    loaded = [
        load_params(f"./checkpoints/lola/{fname}_actor_{i}.pkl")
        for i in range(config["ENV_KWARGS"]["num_agents"])]
    evaluate(loaded,
             socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"]),
             config)


def tune(default_config):
    import copy as _copy
    default_config = OmegaConf.to_container(default_config)
    short = _env_short_name(default_config["ENV_NAME"])

    sweep_config = {
        "name": f"lola_{short}",
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
        wandb.run.name = (
            f"sweep_lola_{cfg['ENV_NAME']}_seed{cfg['SEED']}")
        rng = jax.random.PRNGKey(cfg["SEED"])
        rngs = jax.random.split(rng, cfg["NUM_SEEDS"])
        jax.block_until_ready(
            jax.jit(jax.vmap(make_train(cfg)))(rngs))

    wandb.login()
    sid = wandb.sweep(sweep_config, entity=default_config["ENTITY"],
                      project=default_config["PROJECT"])
    wandb.agent(sid, _wrapped, count=1000)


@hydra.main(version_base=None, config_path="config",
            config_name="lola_cnn_cleanup")
def main(config):
    if config.get("TUNE", False):
        tune(config)
    else:
        single_run(config)


if __name__ == "__main__":
    main()
