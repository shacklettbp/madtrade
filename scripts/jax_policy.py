import jax
import numpy as np
from jax import lax, random, numpy as jnp
from jax.experimental import checkify
import flax
from flax import linen as nn
from flax.core import frozen_dict, FrozenDict
from typing import Callable

import argparse
from functools import partial

import madrona_learn
from madrona_learn import (
    DiscreteActionsConfig, DiscreteActionDistributions,
    ActorCritic, TrainConfig, PPOConfig,
    BackboneShared, BackboneSeparate,
    BackboneEncoder, RecurrentBackboneEncoder,
    ObservationsEMANormalizer,
    Policy,
)

from madrona_learn.models import (
    MLP,
    EntitySelfAttentionNet,
    DenseLayerDiscreteActor,
    DenseLayerCritic,
    DreamerV3Critic,
    LayerNorm,
)
from madrona_learn.rnn import LSTM

actions_config = {
    'buy_sell': DiscreteActionsConfig(
        actions_num_buckets = [ 2, 2 ],
    ),
}

def process_obs_for_mlp(obs, train=False):
    return jnp.concatenate([
        obs['orders'].reshape(*obs['orders'].shape[:-2], -1),
        obs['position'],
    ], axis=-1)


class ActorDistributions(flax.struct.PyTreeNode):
    buy_sell: DiscreteActionDistributions

    def sample(self, prng_key):
        buy_sell_actions, buy_sell_log_probs = self.buy_sell.sample(prng_key)

        return frozen_dict.freeze({
            'buy_sell': buy_sell_actions,
        }), frozen_dict.freeze({
            'buy_sell': buy_sell_log_probs,
        })

    def best(self):
        return frozen_dict.freeze({
            'buy_sell': self.buy_sell.best(),
        })

    def action_stats(self, actions):
        buy_sell_log_probs, buy_sell_entropies = self.buy_sell.action_stats(
            actions['buy_sell'])

        return frozen_dict.freeze({
            'buy_sell': buy_sell_log_probs,
        }), frozen_dict.freeze({
            'buy_sell': buy_sell_entropies,
        })


class ActorHead(nn.Module):
    dtype: jnp.dtype

    @nn.compact
    def __call__(
        self,
        features,
        train=False,
    ):
        buy_sell_dist = DenseLayerDiscreteActor(
            cfg = actions_config['buy_sell'],
            dtype = self.dtype,
        )(features)

        return ActorDistributions(
            buy_sell = buy_sell_dist,
        )


def make_policy(dtype, actions_cfg):
    encoder = BackboneEncoder(
        net = MLP(
            num_channels = 128,
            num_layers = 2,
            dtype=dtype,
        )
    )

    backbone = BackboneShared(
        prefix = process_obs_for_mlp,
        encoder = encoder,
    )

    actor_critic = ActorCritic(
        backbone = backbone,
        actor = ActorHead(dtype=dtype),
        critic = DenseLayerCritic(dtype=dtype),
    )

    obs_preprocess = ObservationsEMANormalizer.create(
        decay = 0.99999,
        dtype = dtype,
        prep_fns = {
        },
        skip_normalization = {
        },
    )

    def get_episode_scores(match_result):
        return False # FIXME, match making code doesn't really support more than 2 teams

    return Policy(
        actor_critic = actor_critic,
        obs_preprocess = obs_preprocess,
        get_episode_scores = get_episode_scores, 
    )
