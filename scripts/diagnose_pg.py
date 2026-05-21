import sys, time, functools, yaml
sys.path.insert(0, "/home/ayman/projects/orcaqd")
import jax, jax.numpy as jnp
import src.jax_env
from src.envs.dex_env import DexHandEnv, EnvConfig
from src.envs.hand_config import ORCAHAND_RIGHT
from src.qd_engine.train import TrainConfig, PolicyNetwork, make_scoring_fn
from qdax.core.containers.mapelites_repertoire import compute_euclidean_centroids
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.core.emitters.mutation_operators import isoline_variation
from qdax.core.emitters.multi_emitter import MultiEmitter
from qdax.core.map_elites import MAPElites
from qdax.utils.metrics import default_qd_metrics
from src.qd_engine.pg_emitter import PGEmitter, PGEmitterConfig

with open("configs/paper1_smoke.yaml") as f:
    yc = yaml.safe_load(f)

cfg = TrainConfig(
    grid_shape=tuple(yc["grid_shape"]), min_bd=tuple(yc["min_bd"]), max_bd=tuple(yc["max_bd"]),
    ga_batch_size=yc["ga_batch_size"], pg_batch_size=yc["pg_batch_size"],
    num_iterations=10, episode_length=yc["episode_length"], env_batch_size=yc["env_batch_size"],
    policy_hidden=yc["policy_hidden"], iso_sigma=yc["iso_sigma"], line_sigma=yc["line_sigma"],
    critic_hidden=yc["critic_hidden"], critic_lr=yc["critic_lr"], actor_lr=yc["actor_lr"],
    pg_steps=yc["pg_steps"], critic_steps=yc.get("critic_steps", 300),
    pg_buffer_size=yc["pg_buffer_size"], pg_warmup=yc.get("pg_warmup", 100),
)

print("=== PGA-MAP-Elites Diagnostic ===")
print("GA=%d, PG=%d, env_batch=%d, ep_len=%d" % (cfg.ga_batch_size, cfg.pg_batch_size, cfg.env_batch_size, cfg.episode_length))

env = DexHandEnv(EnvConfig(hand=ORCAHAND_RIGHT, episode_length=cfg.episode_length))
print("Env: obs=%d, act=%d" % (env.obs_dim, env.action_dim))

policy_net = PolicyNetwork(hidden_size=cfg.policy_hidden, action_dim=env.action_dim)
dummy_obs = jnp.zeros(env.obs_dim)
key = jax.random.PRNGKey(42)
key, init_key = jax.random.split(key)
params = policy_net.init(init_key, dummy_obs)
n_params = sum(p.size for p in jax.tree.leaves(params))
print("Policy: %d params" % n_params)

scoring_fn = make_scoring_fn(env, policy_net, cfg)
centroids = compute_euclidean_centroids(grid_shape=cfg.grid_shape, minval=jnp.array(cfg.min_bd), maxval=jnp.array(cfg.max_bd))

vf = functools.partial(isoline_variation, iso_sigma=cfg.iso_sigma, line_sigma=cfg.line_sigma)
ga_e = MixingEmitter(mutation_fn=lambda x, k: (x, k), variation_fn=vf, variation_percentage=1.0, batch_size=cfg.ga_batch_size)
pg_e = PGEmitter(
    config=PGEmitterConfig(env_batch_size=cfg.pg_batch_size, num_critic_training_steps=cfg.critic_steps,
        num_pg_training_steps=cfg.pg_steps, replay_buffer_size=cfg.pg_buffer_size,
        critic_hidden_layer_size=(cfg.critic_hidden, cfg.critic_hidden),
        critic_learning_rate=cfg.critic_lr, actor_learning_rate=cfg.actor_lr),
    policy_network=policy_net, obs_dim=env.obs_dim, action_dim=env.action_dim,
)
emitter = MultiEmitter((ga_e, pg_e))
me = MAPElites(scoring_function=scoring_fn, emitter=emitter, metrics_function=functools.partial(default_qd_metrics, qd_offset=0.0))

key, ik = jax.random.split(key)
init_keys = jax.random.split(ik, cfg.total_batch_size)
init_g = jax.vmap(lambda k: policy_net.init(k, dummy_obs))(init_keys)

t0 = time.time()
rep, es, met = me.init(init_g, centroids, key)
print("Init: %.1fs, buf=%d" % (time.time()-t0, int(es.emitter_states[1].replay_buffer.current_size)))

update_fn = jax.jit(me.update)
for i in range(cfg.num_iterations):
    key, uk = jax.random.split(key)
    t1 = time.time()
    rep, es, met = update_fn(rep, es, uk)
    dt = time.time() - t1
    qs = float(met["qd_score"])
    cov = float(met["coverage"])
    bs = int(es.emitter_states[1].replay_buffer.current_size)
    print("  [%d/%d] QD=%.3f Cov=%.4f dt=%.2fs buf=%d" % (i+1, cfg.num_iterations, qs, cov, dt, bs))

expected = cfg.total_batch_size * cfg.env_batch_size * cfg.episode_length * cfg.num_iterations
print("Final QD=%.3f, buf=%d, expected_total=%d" % (qs, bs, expected))
