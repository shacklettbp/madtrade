import jax
from jax import lax, random, numpy as jnp
from jax.experimental import checkify
import flax
from flax import linen as nn

import argparse
from functools import partial
import numpy as np

import mad_trade
from mad_trade import Task, SimFlags

import madrona_learn
from madrona_learn import (
    ActorCritic,
    ActionsConfig,
    EvalConfig,
    eval_load_ckpt,
    eval_policies,
)

from common import print_elos
from jax_policy import actions_config, make_policy

madrona_learn.cfg_jax_mem(0.6)

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--num-worlds', type=int, required=True)
arg_parser.add_argument('--num-agents', type=int, required=True)
arg_parser.add_argument('--num-steps', type=int, default=200)
arg_parser.add_argument('--num-policies', type=int, default=1)

arg_parser.add_argument('--ckpt-path', type=str, required=True)

arg_parser.add_argument('--gpu-sim', action='store_true')
arg_parser.add_argument('--gpu-id', type=int, default=0)
arg_parser.add_argument('--fp16', action='store_true')
arg_parser.add_argument('--bf16', action='store_true')

args = arg_parser.parse_args()

team_size = 6

dev = jax.devices()[0]

if args.fp16:
    dtype = jnp.float16
elif args.bf16:
    dtype = jnp.bfloat16
else:
    dtype= jnp.float32

policy = make_policy(dtype, actions_cfg)

if args.single_policy != None:
    assert not args.crossplay
    policy_states, num_policies = eval_load_ckpt(
        policy, args.ckpt_path, single_policy = args.single_policy)
elif args.crossplay:
    policy_states, num_policies = eval_load_ckpt(
        policy, args.ckpt_path,
        train_only=False if args.crossplay_include_past else True)

sim = mad_trade.SimManager(
    exec_mode = mad_trade.madrona.ExecMode.CUDA if args.gpu_sim else mad_trade.madrona.ExecMode.CPU,
    gpu_id = args.gpu_id,
    num_worlds = args.num_worlds,
    num_agents_per_world = args.num_agents,
    sim_flags = SimFlags.AutoReset,
    num_pbt_policies = num_policies,
)

jax_gpu = jax.devices()[0].platform == 'gpu'

sim_fns = sim.jax(jax_gpu)

step_idx = 0

def print_step_cb(actions):
    global step_idx
    print("Step:", step_idx)
    print(actions)
    step_idx += 1

def iter_cb(step_data):
    cb = partial(jax.experimental.io_callback, print_step_cb, ())
    cb(step_data['actions'])

eval_cfg = EvalConfig(
    num_worlds = args.num_worlds,
    num_teams = args.num_agents,
    team_size = 1,
    actions = actions_config,
    num_eval_steps = args.num_steps,
    policy_dtype = dtype,
    eval_competitive = True,
    use_deterministic_policy = False,
    reward_gamma = 0.998,
    #custom_policy_ids = [ -1 ],
)

print_elos(policy_states.mmr.elo)

mmrs = eval_policies(dev, eval_cfg, sim_fns,
    policy, jnp.array([1], jnp.int32), policy_states, iter_cb)

print_elos(mmrs.elo)

del sim
