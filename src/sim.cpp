#include <madrona/mw_gpu_entry.hpp>

#include <madrona/registry.hpp>

#include <cstdint>
#include <algorithm>

#include "sim.hpp"

#ifdef MADRONA_GPU_MODE
#include <madrona/state.hpp>
#endif

using namespace madrona;
using namespace madrona::math;

namespace madtrade {

// Register all the ECS components and archetypes that will be
// use in the simulation
void Sim::registerTypes(ECSRegistry &registry,
                        const TaskConfig &cfg)
{
  (void)cfg;
  base::registerTypes(registry);

  registry.registerSingleton<WorldReset>();

  registry.registerComponent<Reward>();
  registry.registerComponent<Done>();
  registry.registerComponent<AgentPolicy>();

  registry.registerComponent<Action>();
  registry.registerComponent<PlayerState>();
  registry.registerComponent<PlayerOrder>();

  registry.registerComponent<PriceKey>();
  registry.registerComponent<OrderInfo>();

  registry.registerComponent<OrderObservation>();

  registry.registerArchetype<Agent>();
  registry.registerArchetype<Bid>();
  registry.registerArchetype<Ask>();

  registry.registerSingleton<MatchResult>();
  registry.registerSingleton<Market>();

  registry.exportSingleton<WorldReset>(
      (uint32_t)ExportID::Reset);

  registry.exportSingleton<MatchResult>(
      (uint32_t)ExportID::MatchResult);

  registry.exportColumn<Agent, Action>(
      (uint32_t)ExportID::Action);
  registry.exportColumn<Agent, Reward>(
      (uint32_t)ExportID::Reward);
  registry.exportColumn<Agent, Done>(
      (uint32_t)ExportID::Done);
  registry.exportColumn<Agent, AgentPolicy>(
      (uint32_t)ExportID::AgentPolicy);

  registry.exportColumn<Agent, OrderObservation>(
      (uint32_t)ExportID::OrdersObservation);
  registry.exportColumn<Agent, PlayerState>(
      (uint32_t)ExportID::AgentStateObservation);
}

static void cleanupWorld(Engine &ctx)
{
  (void)ctx;
}

static void initWorld(Engine &ctx)
{
  (void)ctx;
  // Do some initialization
}

inline void resetSystem(Engine &ctx, WorldReset &reset)
{
  int32_t force_reset = reset.reset;

  if (force_reset != 0) {
    reset.reset = 0;

    cleanupWorld(ctx);
    initWorld(ctx);
  } 
}

static RewardHyperParams getRewardHyperParamsForPolicy(
  Engine &ctx,
  AgentPolicy agent_policy)
{
  int32_t idx = agent_policy.idx;

  if (idx < 0) {
    idx = 0;
  }

  return ctx.data().rewardHyperParams[idx];
}

inline void rewardSystem(Engine &ctx,
                         AgentPolicy agent_policy,
                         PlayerState &player_state,
                         Reward &out_reward)
{
  (void)ctx;
  (void)player_state;

  RewardHyperParams reward_hyper_params =
      getRewardHyperParamsForPolicy(ctx, agent_policy);

  (void)reward_hyper_params.somePerPolicyRewardScaling;

  out_reward.v = 0;
}

inline void doneSystem(Engine &ctx,
                       Done &done)
{
  (void)ctx;

  done.done = 0;
}

inline void actionsSystem(Engine &,
                          Entity e,
                          Action action,
                          PlayerOrder &order)
{
  // Set the current player order information
  order.type = action.type;

  order.info = OrderInfo {
    .price = action.dollars,
    .size = action.size,
    .issuer = e,
  };
}

struct WorldState {
  uint32_t numPlayerOrders;
  PlayerOrder *playerOrders;

  uint32_t numPlayers;
  PlayerState *playerStates;

  struct {
    uint32_t numAsks;
    OrderInfo *asks;
    Entity *askHandles;

    uint32_t numBids;
    OrderInfo *bids;
    Entity *bidHandles;

    uint32_t curAskOffset;
    uint32_t curBidOffset;

    // These are orders which could never be traded against.
    // We keep these in case there are no trades in the global
    // book.
    OrderInfo dummyAsk;
    OrderInfo dummyBid;
  } globalBook;

  OrderInfo &getLowestAsk()
  {
    if (globalBook.numAsks == 0) {
      return globalBook.dummyAsk;
    } else {
      return globalBook.asks[globalBook.curAskOffset];
    }
  }

  Entity getLowestAskHandle()
  {
    if (globalBook.numAsks == 0) {
      return Entity::none();
    } else {
      return globalBook.askHandles[globalBook.curAskOffset];
    }
  }

  OrderInfo &getHighestBid()
  {
    if (globalBook.numBids == 0) {
      return globalBook.dummyBid;
    } else {
      return globalBook.bids[globalBook.curBidOffset];
    }
  }

  Entity getHighestBidHandle()
  {
    if (globalBook.numBids == 0) {
      return Entity::none();
    } else {
      return globalBook.bidHandles[globalBook.curBidOffset];
    }
  }

  void updateWorldState(Engine &ctx)
  {
    OrderInfo &lowest_ask = getLowestAsk();
    OrderInfo &highest_bid = getHighestBid();

    if (lowest_ask.size == 0) {
      ctx.destroyEntity(lowest_ask.issuer);
      globalBook.curAskOffset++;
    }

    if (highest_bid.size == 0) {
      ctx.destroyEntity(highest_bid.issuer);
      globalBook.curBidOffset++;
    }
  }

  void addOrder(Engine &ctx,
                PlayerOrder order)
  {
    switch (order.type) {
    case OrderType::Ask: {
      Entity ask = ctx.makeEntity<Ask>();

      ctx.get<PriceKey>(ask).v = order.info.price;
      ctx.get<OrderInfo>(ask) = order.info;
    } break;

    case OrderType::Bid: {
      Entity bid = ctx.makeEntity<Bid>();

      // We want the highest bid to appear first
      ctx.get<PriceKey>(bid).v = kMaxPrice - order.info.price;
      ctx.get<OrderInfo>(bid) = order.info;
    } break;
    }
  }
};

template <typename ArchetypeT, typename ComponentT>
std::pair<ComponentT *, uint32_t> getWorldComponentsAndCount(
    StateManager * state_mgr,
    uint32_t world_id)
{
#ifdef MADRONA_GPU_MODE
  ComponentT *glob_comps = state_mgr->getArchetypeComponent<
    ArchetypeT, ComponentT>();
  int32_t *world_offsets = state_mgr->getArchetypeWorldOffsets<
    ArchetypeT>();
  int32_t *world_counts = state_mgr->getArchetypeWorldCounts<
    ArchetypeT>();

  return std::make_pair(glob_comps + world_offsets[world_id],
                        world_counts[world_id]);
#else
  ComponentT *comps = state_mgr->getWorldComponents<
    ArchetypeT, ComponentT>(world_id);
  uint32_t num_comps = (uint32_t)state_mgr->numRows<ArchetypeT>(world_id);

  return std::make_pair(comps, num_comps);
#endif
}

template <typename ArchetypeT, typename ComponentT>
ComponentT * getWorldComponents(
    StateManager * state_mgr,
    uint32_t world_id)
{
#ifdef MADRONA_GPU_MODE
  ComponentT *glob_comps = state_mgr->getArchetypeComponent<
    ArchetypeT, ComponentT>();
  int32_t *world_offsets = state_mgr->getArchetypeWorldOffsets<
    ArchetypeT>();

  return glob_comps + world_offsets[world_id];
#else
  ComponentT *comps = state_mgr->getWorldComponents<
    ArchetypeT, ComponentT>(world_id);

  return comps;
#endif
}

static WorldState getWorldState(Engine &ctx)
{
  (void)ctx;

  // TODO: Provide a nicer unified API in the state manager to
  // get a pointer to the components of an archetype in a single world.
  //
  // On the GPU backend, that means internally fetching the world offset
  // instead of manually adding it yourself.
#ifdef MADRONA_GPU_MODE
  StateManager *state_mgr = mwGPU::getStateManager();
#else
  StateManager *state_mgr = ctx.getStateManager();
#endif
  
  auto [player_orders, num_player_orders] = getWorldComponentsAndCount<
    Agent, PlayerOrder>(state_mgr, ctx.worldID().idx);

  auto [player_states, num_player_states] = getWorldComponentsAndCount<
    Agent, PlayerState>(state_mgr, ctx.worldID().idx);

  auto [asks, num_asks] = getWorldComponentsAndCount<
    Ask, OrderInfo>(state_mgr, ctx.worldID().idx);
  Entity *ask_handles = getWorldComponents<
    Ask, Entity>(state_mgr, ctx.worldID().idx);

  auto [bids, num_bids] = getWorldComponentsAndCount<
    Bid, OrderInfo>(state_mgr, ctx.worldID().idx);
  Entity *bid_handles = getWorldComponents<
    Bid, Entity>(state_mgr, ctx.worldID().idx);

  return WorldState {
    // Current orders
    num_player_orders,
    player_orders,

    num_player_states,
    player_states,

    { // Global book info
      num_asks,
      asks,
      ask_handles,

      num_bids,
      bids,
      bid_handles,

      0, 0,

      OrderInfo {  0xFFFF'FFFF, 0, Entity::none() },
      OrderInfo { 0, 0, Entity::none() },
    },
  };
}

static void genRandomPerm(Engine &ctx,
                          uint32_t *rand_perm,
                          uint32_t num_elems)
{
  for (uint32_t i = 0; i <= num_elems - 2; ++i) {
    uint32_t j = i + rand::sampleI32(
        ctx.data().initRandKey,
        0, num_elems - i);
    std::swap(rand_perm[i], rand_perm[j]);
  }
}

static bool executeTrade(OrderInfo &ask,
                         OrderInfo &bid,
                         PlayerState &asker_state,
                         PlayerState &bidder_state)
{
  // Global has a higher bid; do trade with this
  uint32_t traded_quantity = std::min(
      ask.size, bid.size);

  // Make sure that the bidder has enough dollars to pay
  traded_quantity = std::min(traded_quantity,
                             bidder_state.dollars / ask.price);

  ask.size -= traded_quantity;
  bid.size -= traded_quantity;

  asker_state.dollars += traded_quantity * bid.price;
  bidder_state.dollars -= traded_quantity * bid.price;

  return (traded_quantity > 0);
}

inline void matchSystem(Engine &ctx,
                        Market &)
{
  WorldState world_state = getWorldState(ctx);

  // Get a random permutation of current orders
  uint32_t *rand_perm = (uint32_t *)ctx.tmpAlloc(
      sizeof(uint32_t) * world_state.numPlayerOrders);
  genRandomPerm(ctx, rand_perm, world_state.numPlayerOrders);

  for (int32_t i = 0; i < (int32_t)world_state.numPlayerOrders; ++i) {
    uint32_t i_agent_idx = rand_perm[i];
    PlayerOrder &i_order = world_state.playerOrders[i_agent_idx];
    PlayerState &i_state = world_state.playerStates[i_agent_idx];

    for (int32_t j = (int32_t)i - 1; j >= 0; --j) {
      uint32_t j_agent_idx = rand_perm[j];
      PlayerOrder &j_order = world_state.playerOrders[j_agent_idx];
      PlayerState &j_state = world_state.playerStates[j_agent_idx];

      if (i_order.type == j_order.type || j_order.info.size == 0) {
        continue;
      }

      switch (i_order.type) {
      case OrderType::Ask: {
        // I is ask; J is bid
        if (j_order.info.price > i_order.info.price) {
          // Check if the global book has a better trade
          OrderInfo &glob_bid = world_state.getHighestBid();

          if (glob_bid.price > j_order.info.price) {
            PlayerState &issuer_state = ctx.get<PlayerState>(glob_bid.issuer);

            executeTrade(i_order.info, glob_bid,
                i_state, issuer_state);

            // Makes sure to clean up the global 
            world_state.updateWorldState(ctx);
          } else {
            executeTrade(i_order.info, j_order.info,
                i_state, j_state);
          }
        }
      } break;

      case OrderType::Bid: {
        if (j_order.info.price < i_order.info.price) {
          // Check if global book has a better trade
          OrderInfo &glob_ask = world_state.getLowestAsk();

          if (glob_ask.price < j_order.info.price) {
            PlayerState &issuer_state = ctx.get<PlayerState>(glob_ask.issuer);

            executeTrade(glob_ask, i_order.info,
                issuer_state, i_state);

            world_state.updateWorldState(ctx);
          } else {
            executeTrade(j_order.info, i_order.info,
                j_state, i_state);
          }
        }
      } break;
      }
    }
  }

  // Maybe it's still possible to trade against stuff in the global book
  // if going through the current orders didn't exhaust everything
  for (uint32_t i = 0; i < world_state.numPlayerOrders; ++i) {
    uint32_t i_agent_idx = rand_perm[i];
    PlayerOrder &i_order = world_state.playerOrders[i_agent_idx];
    PlayerState &i_state = world_state.playerStates[i_agent_idx];

    // Keep taking from the global book until order has been exhausted
    bool traded = false;
    do {
      switch (i_order.type) {
      case OrderType::Ask: {
        OrderInfo &glob_bid = world_state.getHighestBid();

        if (glob_bid.price > i_order.info.price) {
          PlayerState &issuer_state = ctx.get<PlayerState>(glob_bid.issuer);

          traded = executeTrade(i_order.info, glob_bid,
                       i_state, issuer_state);

          world_state.updateWorldState(ctx);
        }
      } break;

      case OrderType::Bid: {
        OrderInfo &glob_ask = world_state.getLowestAsk();

        if (glob_ask.price < i_order.info.price) {
          PlayerState &issuer_state = ctx.get<PlayerState>(glob_ask.issuer);

          traded = executeTrade(glob_ask, i_order.info,
                       issuer_state, i_state);

          world_state.updateWorldState(ctx);
        }
      } break;
      }
    } while (traded);

    // If the order still has stuff in it, push it to the end of the global book
    if (i_order.info.size > 0) {
      world_state.addOrder(
          ctx, 
          i_order);
    }
  }
}

inline void fillOrderObservationsSystem(Engine &ctx,
                                        const PlayerState &player_state,
                                        OrderObservation &obs)
{
  (void)ctx;
  (void)player_state;
  (void)obs;
}

static TaskGraphNodeID resetAndObsTasks(TaskGraphBuilder &builder, 
                                        const TaskConfig &cfg,
                                        Span<const TaskGraphNodeID> deps)
{
  (void)cfg;

  // Conditionally reset the world if the episode is over
  auto reset_sys = builder.addToGraph<ParallelForNode<Engine,
    resetSystem,
      WorldReset
    >>(deps);

  auto obs_sys = builder.addToGraph<ParallelForNode<Engine,
    fillOrderObservationsSystem,
      PlayerState,
      OrderObservation
    >>({reset_sys});

  return obs_sys;
}

static void setupInitTasks(TaskGraphBuilder &builder, const TaskConfig &cfg)
{
  resetAndObsTasks(builder, cfg, {});
}

static void setupStepTasks(TaskGraphBuilder &builder, const TaskConfig &cfg)
{
  auto node = builder.addToGraph<ParallelForNode<Engine,
    actionsSystem,
      Entity,
      Action,
      PlayerOrder
    >>({});

  node = builder.addToGraph<SortArchetypeNode<
    Ask, PriceKey>>({node});
  node = builder.addToGraph<CompactArchetypeNode<Ask>>({node});

  node = builder.addToGraph<SortArchetypeNode<
    Bid, PriceKey>>({node});
  node = builder.addToGraph<CompactArchetypeNode<Bid>>({node});

  node = builder.addToGraph<CompactArchetypeNode<Agent>>({node});

  node = builder.addToGraph<ParallelForNode<Engine,
    matchSystem,
      Market
    >>({node});

  node = builder.addToGraph<ParallelForNode<Engine,
    rewardSystem,
      AgentPolicy,
      PlayerState,
      Reward
    >>({node});

  // Set done values if match is finished
  node = builder.addToGraph<ParallelForNode<Engine,
    doneSystem,
      Done
    >>({node});

  resetAndObsTasks(builder, cfg, { node });
}

void Sim::setupTasks(TaskGraphManager &taskgraph_mgr, const TaskConfig &cfg)
{
  setupInitTasks(taskgraph_mgr.init(TaskGraphID::Init), cfg);
  setupStepTasks(taskgraph_mgr.init(TaskGraphID::Step), cfg);
}

Sim::Sim(Engine &ctx,
         const TaskConfig &cfg,
         const WorldInit &)
    : WorldBase(ctx)
{
  rewardHyperParams = cfg.rewardHyperParamsBuffer;
  initRandKey = cfg.initRandKey;
  rng = RNG(rand::split_i(initRandKey,
        0, (uint32_t)ctx.worldID().idx));

  ctx.singleton<Market>() = {};
  initWorld(ctx);
}


MADRONA_BUILD_MWGPU_ENTRY(Engine, Sim, TaskConfig, Sim::WorldInit);

}
