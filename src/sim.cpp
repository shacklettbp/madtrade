#include <madrona/mw_gpu_entry.hpp>

#include <madrona/registry.hpp>

#include <cstdint>
#include <algorithm>

#include "sim.hpp"

#ifdef MADRONA_GPU_MODE
#include <madrona/state.hpp>
#endif

//#define ENABLE_LOGGING

#ifdef ENABLE_LOGGING
#define LOG(...) printf(__VA_ARGS__)
#else
#define LOG(...)
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
  registry.registerComponent<NPCState>();
  registry.registerComponent<NPCOrder>();

  registry.registerComponent<PriceKey>();
  registry.registerComponent<OrderInfo>();

  registry.registerComponent<FullObservation>();

  registry.registerArchetype<Agent>();
  registry.registerArchetype<Bid>();
  registry.registerArchetype<Ask>();
  registry.registerArchetype<NPC>();

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

  registry.exportColumn<Agent, FullObservation>(
      (uint32_t)ExportID::Observation);
}

static void cleanupWorld(Engine &ctx)
{
#ifdef MADRONA_GPU_MODE
  StateManager *state_mgr = mwGPU::getStateManager();
#else
  StateManager *state_mgr = ctx.getStateManager();
#endif

  { // Destroy all agents
    Entity *agents = state_mgr->getWorldEntities<Agent>(
        ctx.worldID().idx);
    for (uint32_t i = 0; i < ctx.data().numAgents; ++i) {
      ctx.destroyEntity(agents[i]);
    }
  }

  { // Destroy NPCs
    Entity *npcs = state_mgr->getWorldEntities<NPC>(
        ctx.worldID().idx);
    for (uint32_t i = 0; i < ctx.data().numNPCs; ++i) {
      ctx.destroyEntity(npcs[i]);
    }
  }

  { // Destroy asks
    auto [asks, num_asks] = state_mgr->getWorldComponentsAndCount<
      Ask, OrderInfo>(ctx.worldID().idx);
    Entity *ask_handles = state_mgr->getWorldEntities<
      Ask>(ctx.worldID().idx);
    (void)asks;

    for (uint32_t i = 0; i < num_asks; ++i) {
      ctx.destroyEntity(ask_handles[i]);
    }
  }

  { // Destroy bids
    auto [bids, num_bids] = state_mgr->getWorldComponentsAndCount<
      Bid, OrderInfo>(ctx.worldID().idx);
    Entity *bid_handles = state_mgr->getWorldEntities<
      Bid>(ctx.worldID().idx);
    (void)bids;

    for (uint32_t i = 0; i < num_bids; ++i) {
      ctx.destroyEntity(bid_handles[i]);
    }
  }
}

static void initWorld(Engine &ctx)
{
  uint32_t num_agents = ctx.data().numAgents;
  uint32_t num_npcs = ctx.data().numNPCs;

  // Initialize agents
  for (uint32_t i = 0; i < num_agents; ++i) {
    Entity a = ctx.makeEntity<Agent>();

    ctx.get<PlayerState>(a) = {
      .position = 10,
      .dollars = 1000,
      .positionIfAsksFilled = 10,
      .dollarsIfBidsFilled = 1000,
      .prevAsk = Entity::none(),
      .prevBid = Entity::none(),
    };

    ctx.get<PlayerOrder>(a) = {
      .type = OrderType::None,
      .info = {}
    };

    ctx.get<Reward>(a) = {
      .v = 0.f
    };

    ctx.get<Done>(a) = {
      .done = 0
    };

    FullObservation &all_obs = ctx.get<FullObservation>(a);
    memset(&all_obs, 0, sizeof(FullObservation));
  }

  // Initialize NPCs
  for (uint32_t i = 0; i < num_npcs; ++i) {
    Entity n = ctx.makeEntity<NPC>();

    // Generate random secret number between 0 and D-1
    int32_t secret_number = ctx.data().rng.sampleI32(0, ctx.data().D - 1);

    ctx.get<NPCState>(n) = {
      .position = 10,
      .dollars = 1000,
      .secretNumber = secret_number,
      .positionIfAsksFilled = 10,
      .dollarsIfBidsFilled = 1000,
      .prevAsk = Entity::none(),
      .prevBid = Entity::none(),
    };

    ctx.get<NPCOrder>(n) = {
      .type = OrderType::None,
      .info = {}
    };

    ctx.get<Reward>(n) = {
      .v = 0.f
    };

    ctx.get<Done>(n) = {
      .done = 0
    };
  }

  ctx.singleton<WorldReset>().reset = 0;
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

  // Current amount of dollars plus position times settlement price
  out_reward.v = player_state.dollars + player_state.position * ctx.data().settlementPrice;
}

inline void doneSystem(Engine &ctx,
                       Done &done)
{
  (void)ctx;

  done.done = 0;
}

inline void actionsSystem(Engine &ctx,
                          Entity e,
                          Action &action,
                          PlayerOrder &order)
{
  // Set the current player order information
  order.type = action.type;

  order.info = OrderInfo {
    .price = action.dollars,
    .size = action.size,
    .issuer = e,
  };

  LOG("Agent %d (entity %d) placed order (%s); price = %d; size = %d\n",
      ctx.loc(e).row,
      e.id,
      action.type == OrderType::Ask ? "ask" : (action.type == OrderType::Bid ? "bid" : "hold"),
      order.info.price,
      order.info.size);
}

struct StepStats {
  int32_t volume = 0;
  int32_t dirVolume = 0;
  int32_t totalDollarsTraded = 0;

  Order executedTrades[NUM_TRACKED_EXECUTED_ORDERS] = {};
  int numExecutedTrades = 0;
};

struct WorldState {
  uint32_t numPlayerOrders;
  PlayerOrder *playerOrders;
  uint32_t numNPCOrders;
  NPCOrder *npcOrders;

  uint32_t numPlayers;
  PlayerState *playerStates;
  uint32_t numNPCs;
  NPCState *npcStates;

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

  StepStats stepStats;

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
    if (globalBook.curAskOffset >= globalBook.numAsks) {
      return Entity::none();
    } else {
      return globalBook.askHandles[globalBook.curAskOffset];
    }
  }

  OrderInfo &getHighestBid()
  {
    if (globalBook.curBidOffset >= globalBook.numBids) {
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

  // Clean up orders with zero size from the global book
  void cleanupZeroSizeOrders(Engine &ctx)
  {
    LOG("Global book has %d asks (offset = %d) and %d bids (offset = %d)\n", globalBook.numAsks, globalBook.curAskOffset, globalBook.numBids, globalBook.curBidOffset);

    // Clean up zero-size asks
    while (globalBook.curAskOffset < globalBook.numAsks) {
      OrderInfo &ask = getLowestAsk();
      Entity ask_handle = getLowestAskHandle();
      
      // Break if we hit a valid ask or the dummy ask
      if (ask.price == kMaxPrice || ask_handle == Entity::none()) {
        break;
      }

      // If the ask has zero size, destroy it and move to next
      if (ask.size == 0) {
        ctx.destroyEntity(ask_handle);
        LOG("Destroyed zero-size ask from global book issued by agent %d\n", ctx.loc(ask.issuer).row);
        globalBook.curAskOffset++;
      } else {
        break;
      }
    }

    // Clean up zero-size bids
    while (globalBook.curBidOffset < globalBook.numBids) {
      OrderInfo &bid = getHighestBid();
      Entity bid_handle = getHighestBidHandle();
      
      // Break if we hit a valid bid or the dummy bid
      if (bid.price == 0 || bid_handle == Entity::none()) {
        break;
      }

      // If the bid has zero size, destroy it and move to next
      if (bid.size == 0) {
        ctx.destroyEntity(bid_handle);
        LOG("Destroyed zero-size bid from global book issued by agent %d\n", ctx.loc(bid.issuer).row);
        globalBook.curBidOffset++;
      } else {
        break;
      }
    }
  }


  Entity addOrder(Engine &ctx,
                PlayerOrder order)
  {
    switch (order.type) {
    case OrderType::Ask: {
      Entity ask = ctx.makeEntity<Ask>();

      ctx.get<PriceKey>(ask).v = order.info.price;
      ctx.get<OrderInfo>(ask) = order.info;

      return ask;
    } break;

    case OrderType::Bid: {
      Entity bid = ctx.makeEntity<Bid>();

      // We want the highest bid to appear first
      ctx.get<PriceKey>(bid).v = kMaxPrice - order.info.price;
      ctx.get<OrderInfo>(bid) = order.info;

      return bid;
    } break;
    case OrderType::Hold: {
      return Entity::none();
    } break;
    default: {
      MADRONA_UNREACHABLE();
    } break;
    }
  }
};

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
  
  auto [player_orders, num_player_orders] = state_mgr->getWorldComponentsAndCount<
    Agent, PlayerOrder>(ctx.worldID().idx);

  auto [player_states, num_player_states] = state_mgr->getWorldComponentsAndCount<
    Agent, PlayerState>(ctx.worldID().idx);

  auto [npc_orders, num_npc_orders] = state_mgr->getWorldComponentsAndCount<
    NPC, NPCOrder>(ctx.worldID().idx);

  auto [npc_states, num_npcs] = state_mgr->getWorldComponentsAndCount<
    NPC, NPCState>(ctx.worldID().idx);

  auto [asks, num_asks] = state_mgr->getWorldComponentsAndCount<
    Ask, OrderInfo>(ctx.worldID().idx);
  Entity *ask_handles = state_mgr->getWorldEntities<
    Ask>(ctx.worldID().idx);

  auto [bids, num_bids] = state_mgr->getWorldComponentsAndCount<
    Bid, OrderInfo>(ctx.worldID().idx);
  Entity *bid_handles = state_mgr->getWorldEntities<
    Bid>(ctx.worldID().idx);

  return WorldState {
    // Current orders
    num_player_orders,
    player_orders,
    num_npc_orders,
    npc_orders,

    num_player_states,
    player_states,
    num_npcs,
    npc_states,

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
    StepStats {},
  };
}

static void genRandomPerm(Engine &ctx,
                          uint32_t *rand_perm,
                          uint32_t num_elems)
{
  for (uint32_t i = 0; i < num_elems; ++i) {
    rand_perm[i] = i;
  }

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
                         PlayerState &bidder_state,
                         StepStats &stats,
                         bool ask_is_resting)
{
  // Global has a higher bid; do trade with this
  uint32_t traded_quantity = std::min(
      ask.size, bid.size);

  uint32_t price;
  if (ask_is_resting) {
    price = ask.price;
  } else {
    price = bid.price;
  }

  // Make sure that the bidder has enough dollars to pay
  traded_quantity = std::min(traded_quantity,
                             bidder_state.dollars / price);

  ask.size -= traded_quantity;
  bid.size -= traded_quantity;

  // Update actual balances
  asker_state.dollars += traded_quantity * price;
  bidder_state.dollars -= traded_quantity *price;

  // Update actual positions
  asker_state.position -= traded_quantity;  // Seller loses position
  bidder_state.position += traded_quantity; // Buyer gains position

  // Don't update positionIfAsksFilled or dollarsIfBidsFilled here
  // because these values already account for these orders being filled
  
  if (ask_is_resting) {
    // The bidder overbid but is paying less because the ask order's price is used
    uint32_t delta = bid.price - ask.price;
    bidder_state.dollarsIfBidsFilled += delta * traded_quantity;
  }

  stats.volume += traded_quantity;

  if (ask_is_resting) {
    stats.dirVolume += traded_quantity;
  } else {
    stats.dirVolume -= traded_quantity;
  }

  stats.totalDollarsTraded += price * traded_quantity;

  int executed_trade_idx = stats.numExecutedTrades++;
  executed_trade_idx %= NUM_TRACKED_EXECUTED_ORDERS;

  stats.executedTrades[executed_trade_idx] = {
    .size = (int32_t)traded_quantity,
    .price = price,
  };

  return (traded_quantity > 0);
}

inline void npcSystem(Engine &ctx,
                     Entity e,
                     NPCState &npc_state)
{
  // Generate random price between 0 and N * (D-1)
  uint32_t max_price = ctx.data().numAgents * (ctx.data().D - 1);
  uint32_t quote_price = ctx.data().rng.sampleI32(0, max_price);

  // Store the quote price in the NPC's state
  npc_state.currentQuotePrice = quote_price;

  LOG("[npcSystem] NPC %d (entity %d) quotes price: %d\n", 
         ctx.loc(e).row, e.id, quote_price);

  // Create two orders in the global book - one ask and one bid at the quote price
  // with effectively infinite size
  WorldState world_state = getWorldState(ctx);

  // Add an ask at the quote price (players can buy at this price)
  PlayerOrder ask_order = {
    .type = OrderType::Ask,
    .info = {
      .price = quote_price,
      .size = 1000000,  // Large number to represent "infinite" liquidity
      .issuer = e,
    }
  };
  Entity ask = world_state.addOrder(ctx, ask_order);
  npc_state.prevAsk = ask;

  // Add a bid at the quote price (players can sell at this price)
  PlayerOrder bid_order = {
    .type = OrderType::Bid,
    .info = {
      .price = quote_price,
      .size = 1000000,  // Large number to represent "infinite" liquidity
      .issuer = e,
    }
  };
  Entity bid = world_state.addOrder(ctx, bid_order);
  npc_state.prevBid = bid;
}

inline void matchSystem(Engine &ctx,
                        Market &market)
{
  (void)market;
  WorldState world_state = getWorldState(ctx);

  {
    world_state.stepStats = StepStats {};
  }

  // Get a random permutation of current orders
  uint32_t *rand_perm = (uint32_t *)ctx.tmpAlloc(
      sizeof(uint32_t) * world_state.numPlayerOrders);
  genRandomPerm(ctx, rand_perm, world_state.numPlayerOrders);

  for (int32_t i = 0; i < (int32_t)world_state.numPlayerOrders; ++i) {
    uint32_t i_agent_idx = rand_perm[i];
    PlayerOrder &i_order = world_state.playerOrders[i_agent_idx];
    PlayerState &i_state = world_state.playerStates[i_agent_idx];

    LOG("\n[i loop] Agent (%d) has order (%s; price=%d; size=%d)\n", 
        i_agent_idx, 
        i_order.type == OrderType::Ask ? "ask" : (i_order.type == OrderType::Bid ? "bid" : "hold"), 
        i_order.info.price, 
        i_order.info.size
    );

    if (ctx.data().simFlags == SimFlags::InterpretAddAsReplace) {
      // Handle replacement mode - cancel any existing orders first
      if (i_order.type == OrderType::Ask) {
        if (i_state.prevAsk != Entity::none()) {
            LOG("[AddAsReplace] Cancelling agent %d's previous ask\n", i_agent_idx);
          // Get the previous ask order info
          OrderInfo &prev_order_info = ctx.get<OrderInfo>(i_state.prevAsk);
          
          // Add back the position that would have been sold
          i_state.positionIfAsksFilled += prev_order_info.size;
          
          // Delete the old ask
          prev_order_info.size = 0;
          i_state.prevAsk = Entity::none();
        }
      } else if (i_order.type == OrderType::Bid) {
        if (i_state.prevBid != Entity::none()) {
            LOG("[AddAsReplace] Cancelling agent %d's previous bid\n", i_agent_idx);
          // Get the previous bid order info
          OrderInfo &prev_order_info = ctx.get<OrderInfo>(i_state.prevBid);
          
          // Add back the dollars that would have been spent
          i_state.dollarsIfBidsFilled += prev_order_info.size * prev_order_info.price;
          prev_order_info.size = 0;
          i_state.prevBid = Entity::none();
        }
      }
    }

    if (i_order.type == OrderType::Hold) {
      i_order.info.size = 0;
      i_order.info.price = 0;
    }

    // Validate if order can be executed
    if (i_order.type == OrderType::Bid) {
      if (i_order.info.size * i_order.info.price > (int64_t)i_state.dollarsIfBidsFilled) {
        LOG("[Invalid] Agent %d's bid is too large: %d * %d > %d\n", i_agent_idx, i_order.info.size, i_order.info.price, i_state.dollarsIfBidsFilled);
        i_order.info.size = 0;
        continue;
      }
      // Update the state if the order is valid
      i_state.dollarsIfBidsFilled -= i_order.info.size * i_order.info.price;
    } else if (i_order.type == OrderType::Ask) {
      if (i_order.info.size > i_state.positionIfAsksFilled) {
        LOG("[Invalid] Agent %d's ask is too large: %d > %d\n", i_agent_idx, i_order.info.size, i_state.positionIfAsksFilled);
        i_order.info.size = 0;
        continue;
      }
      // Update the state if the order is valid
      i_state.positionIfAsksFilled -= i_order.info.size;
    }

    bool didExecute = false;
    while (i_order.info.size) {
      // First, find the best order of new orders not in the book yet from prior players in the ordering
      int32_t best_trade_idx = -1;
      uint32_t best_price = (i_order.type == OrderType::Ask) ?
        world_state.getHighestBid().price : world_state.getLowestAsk().price;

      for (int32_t j = 0; j < i; ++j) {
        uint32_t j_agent_idx = rand_perm[j];
        PlayerOrder &j_order = world_state.playerOrders[j_agent_idx];

        if (i_order.type == j_order.type) {
          LOG("[Skip] Agents %d and %d have orders of same type (%s), skipping\n",
                 i_agent_idx, j_agent_idx,
                 j_order.type == OrderType::Ask ? "ask" : "bid");
          continue;
        }

        if (j_order.info.size == 0) {
            LOG("[Skip] Agent %d has order of size 0, skipping\n", j_agent_idx);
            continue;
        }

        if (j_order.type == OrderType::Ask) {
          if (j_order.info.price < best_price) {
            // LOG("[best_trade_idx] Agent %d's ask is better than current best price: %d < %d\n", j_agent_idx, j_order.info.price, best_price);
            best_trade_idx = j_agent_idx;
            best_price = j_order.info.price;
          }
        } else if (j_order.type == OrderType::Bid) {
          if (j_order.info.price > best_price) {
            // LOG("[best_trade_idx] Agent %d's bid is better than current best price: %d > %d\n", j_agent_idx, j_order.info.price, best_price);
            best_trade_idx = j_agent_idx;
            best_price = j_order.info.price;
          }
        } else {
          assert(false);
        }
      }

    //   LOG("[best_trade_idx] best_trade_idx = %d (best_price = %d)\n", best_trade_idx, best_price);

      if (i_order.type == OrderType::Ask) {
        if (best_price < i_order.info.price) {
          break;
        }

        if (best_trade_idx == -1) {
          OrderInfo &glob_bid = world_state.getHighestBid();

          if (glob_bid.issuer == Entity::none()) {
            LOG("[Match Global] Agent %d's bid is not in the book\n", i_agent_idx);
            break;
          }

          LOG("[DEBUG] Issuer of glob_bid is %d, none entity is %d\n", glob_bid.issuer.id, Entity::none().id);

          PlayerState &issuer_state = ctx.get<PlayerState>(glob_bid.issuer);

          LOG("[Match Global] Agent %d bidding with agent %d's ask of (price = %d; size = %d) from global book\n",
                 ctx.loc(glob_bid.issuer).row, i_agent_idx, i_order.info.price, i_order.info.size);

          executeTrade(i_order.info, glob_bid,
                       i_state, issuer_state, world_state.stepStats, false);

          // Makes sure to clean up the global 
          world_state.cleanupZeroSizeOrders(ctx);
          didExecute = true;
        } else {
          PlayerOrder &other_order = world_state.playerOrders[best_trade_idx];
          PlayerState &other_state = world_state.playerStates[best_trade_idx];

          LOG("[Match] Agent %d asking with agent %d's bid of (price = %d; size = %d)\n",
                 i_agent_idx, best_trade_idx, other_order.info.price, other_order.info.size);

          executeTrade(i_order.info, other_order.info,
                       i_state, other_state, world_state.stepStats, false);
          didExecute = true;
        }
      } else if (i_order.type == OrderType::Bid) {
        if (best_price > i_order.info.price) {
          break;
        }

        if (best_trade_idx == -1) {
          OrderInfo &glob_ask = world_state.getLowestAsk();

          if (glob_ask.issuer == Entity::none()) {
            LOG("[Match Global] Agent %d's ask is not in the book\n", i_agent_idx);
            break;
          }

          PlayerState &issuer_state = ctx.get<PlayerState>(glob_ask.issuer);

          LOG("[Match Global] Agent %d bidding with agent %d's ask of (price = %d; size = %d) from global book\n",
                 i_agent_idx, ctx.loc(glob_ask.issuer).row, glob_ask.price, glob_ask.size);

          executeTrade(glob_ask, i_order.info,
                       issuer_state, i_state, world_state.stepStats, true);

          // Makes sure to clean up the global 
          world_state.cleanupZeroSizeOrders(ctx);
          didExecute = true;
        } else {
          PlayerOrder &other_order = world_state.playerOrders[best_trade_idx];
          PlayerState &other_state = world_state.playerStates[best_trade_idx];

          LOG("[Match] Agent %d bidding with agent %d's ask of (price = %d; size = %d)\n",
                 i_agent_idx, best_trade_idx, other_order.info.price, other_order.info.size);

          executeTrade(other_order.info, i_order.info,
                       other_state, i_state, world_state.stepStats, true);
          didExecute = true;
        }
      } else {
        assert(false);
      }
    }

    if (!didExecute) {
      LOG("[No match] Agent %d's order was not executed\n", i_agent_idx);
    }
  }

  LOG("\n[matchSystem] Done with i loop\n\n");

  // After going through all orders, add any remaining orders to the global book
  for (uint32_t i = 0; i < world_state.numPlayerOrders; ++i) {
    uint32_t i_agent_idx = rand_perm[i];
    PlayerOrder &i_order = world_state.playerOrders[i_agent_idx];
    PlayerState &i_state = world_state.playerStates[i_agent_idx];

    // If the order still has stuff in it, push it to the end of the global book
    if (i_order.info.size > 0) {
      LOG("[Remaining] Adding agent %d's order to the global book (type=%s; price=%d; size=%d)\n", 
          i_agent_idx,
          i_order.type == OrderType::Ask ? "ask" : "bid",
          i_order.info.price,
          i_order.info.size);
      Entity new_order = world_state.addOrder(ctx, i_order);

      // Store reference to the order
      if (i_order.type == OrderType::Ask) {
        if (ctx.data().simFlags == SimFlags::InterpretAddAsReplace) {
          assert(i_state.prevAsk == Entity::none());
        }
        i_state.prevAsk = new_order;
      } else if (i_order.type == OrderType::Bid) {
        if (ctx.data().simFlags == SimFlags::InterpretAddAsReplace) {
          assert(i_state.prevBid == Entity::none());
        }
        i_state.prevBid = new_order;
      }
    }
  }
}

inline void fillOrderObservationsSystem(Engine &ctx,
                                        const PlayerState &player_state,
                                        FullObservation &all_obs)
{
  (void)ctx;
  (void)player_state;

  // Every player will fill in top K orders from the book
  WorldState world_state = getWorldState(ctx);

  for (int32_t i = OBSERVATION_HISTORY_LEN - 2; i >= 0; i--) {
    all_obs.obs[i + 1] = all_obs.obs[i];
  }

  TimeStepObservation &cur_obs = all_obs.obs[0];
  StepStats &stats = world_state.stepStats;

  cur_obs.me.position = player_state.position;
  cur_obs.me.dollars = player_state.dollars;
  cur_obs.me.positionIfAsksFilled = player_state.positionIfAsksFilled;
  cur_obs.me.dollarsIfBidsFilled = player_state.dollarsIfBidsFilled;
  
  if (stats.volume == 0) {
    cur_obs.avgPrice = -1;
  } else {
    cur_obs.avgPrice = stats.totalDollarsTraded / stats.volume;
  }
  cur_obs.dirVolume = stats.dirVolume;
  cur_obs.volume = stats.volume;

  for (int i = 0; i < NUM_TRACKED_EXECUTED_ORDERS; i++) {
    cur_obs.executedTrades[i] = stats.executedTrades[i];
  }

  // Fill ask observations
  uint32_t to_cpy = std::min((uint32_t)K, world_state.globalBook.numAsks);
  for (uint32_t i = 0; i < to_cpy; ++i) {
    cur_obs.bookAsks[i] = Order {
      world_state.globalBook.asks[i].size,
      world_state.globalBook.asks[i].price,
    };
  }
  for (uint32_t i = to_cpy; i < K; ++i) {
    cur_obs.bookAsks[i] = Order { 0, 0 };
  }

  // Fill bid observations
  to_cpy = std::min((uint32_t)K, world_state.globalBook.numBids);
  for (uint32_t i = 0; i < to_cpy; ++i) {
    cur_obs.bookBids[i] = Order {
      world_state.globalBook.bids[i].size,
      world_state.globalBook.bids[i].price,
    };
  }

  for (uint32_t i = to_cpy; i < K; ++i) {
    cur_obs.bookBids[i] = Order { 0, 0 };
  }

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
      FullObservation
    >>({reset_sys});

  return obs_sys;
}

static void setupInitTasks(TaskGraphBuilder &builder, const TaskConfig &cfg)
{
  auto node = builder.addToGraph<SortArchetypeNode<
    Ask, PriceKey>>({});
  node = builder.addToGraph<CompactArchetypeNode<Ask>>({node});

  node = builder.addToGraph<SortArchetypeNode<
    Bid, PriceKey>>({node});
  node = builder.addToGraph<CompactArchetypeNode<Bid>>({node});

  node = builder.addToGraph<CompactArchetypeNode<Agent>>({node});

  resetAndObsTasks(builder, cfg, {node});
}

static void setupStepTasks(TaskGraphBuilder &builder, const TaskConfig &cfg)
{
  auto node = builder.addToGraph<ParallelForNode<Engine,
    actionsSystem,
      Entity,
      Action,
      PlayerOrder
    >>({});

  // Add NPC system before match system
  node = builder.addToGraph<ParallelForNode<Engine,
    npcSystem,
      Entity,
      NPCState
    >>({node});

  node = builder.addToGraph<ParallelForNode<Engine,
    matchSystem,
      Market
    >>({node});

  node = builder.addToGraph<CompactArchetypeNode<Ask>>({node});
  node = builder.addToGraph<CompactArchetypeNode<Bid>>({node});

  node = builder.addToGraph<SortArchetypeNode<
    Ask, PriceKey>>({node});
  node = builder.addToGraph<SortArchetypeNode<
    Bid, PriceKey>>({node});

  node = builder.addToGraph<CompactArchetypeNode<Ask>>({node});
  node = builder.addToGraph<CompactArchetypeNode<Bid>>({node});

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
    : WorldBase(ctx),
      numAgents(cfg.numAgents),
      simFlags(cfg.simFlags),
      settlementPrice(cfg.settlementPrice),
      numNPCs(cfg.numNPCs),
      D(cfg.D)
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
