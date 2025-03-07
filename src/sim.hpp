#pragma once

#include <madrona/rand.hpp>
#include <madrona/taskgraph.hpp>
#include <madrona/taskgraph_builder.hpp>
#include <madrona/custom_context.hpp>
#include <madrona/components.hpp>

namespace madtrade {

inline constexpr int32_t K = 10;
inline constexpr uint32_t kMaxPrice = 100;

class Engine;

// This enum is used by the Sim and Manager classes to track the export slots
// for each component exported to the training code.
enum class ExportID : uint32_t {
    Reset,
    // TODO: This isn't filled
    MatchResult,
    Action,
    Reward,
    Done,
    AgentPolicy,
    AskOrdersObservation,
    BidOrdersObservation,
    AgentStateObservation,

    NumExports,
};

enum class TaskGraphID : uint32_t {
  Init,
  Step,
  NumGraphs,
};

enum class OrderType : uint32_t {
  Ask,
  Bid,
  None
};

struct SimControl {
  // Useful for changing sim behavior for train vs eval
  int32_t someParamForChangingSimBehavior = 0;
};

enum class SimFlags : uint32_t {
  InterpretAddAsReplace,
};

struct WorldReset {
  int32_t reset;
};

struct RewardHyperParams {
  float somePerPolicyRewardScaling;
};

struct Reward {
  float v;
};

struct Done {
  int32_t done;
};

struct MatchResult {
  int32_t matchWinner;
};

struct AgentPolicy {
  int32_t idx;
};

struct Action {
  OrderType type;
  uint32_t dollars;
  int32_t size;
};

struct OrderInfo {
  uint32_t price;
  int32_t size;
  madrona::Entity issuer;
};

struct PlayerOrder {
  OrderType type;
  OrderInfo info;
};

// Used for sorting. This is either the price or the maxprice-price
struct PriceKey {
  uint32_t v;
};

struct Bid : madrona::Archetype<
  PriceKey,
  OrderInfo
> {};

struct Ask : madrona::Archetype<
  PriceKey,
  OrderInfo
> {};

struct PlayerState {
  int32_t position;
  int32_t dollars;

  // How many units would be left if all outstanding asks were filled
  int32_t positionIfAsksFilled;
    // How many dollars would be left if all outstanding bids were filled
  int32_t dollarsIfBidsFilled;

  Entity prevAsk;
  Entity prevBid;
};

struct Order {
  int32_t size;
  uint32_t price;
};

struct AskOrderObservation {
  Order orders[K];
};

struct BidOrderObservation {
  Order orders[K];
};

struct Agent : madrona::Archetype<
  PlayerState,
  PlayerOrder,
  AskOrderObservation,
  BidOrderObservation,
  Action,
  Reward,
  Done,
  AgentPolicy
> {};

struct Market {

};

struct TaskConfig {
  RewardHyperParams *rewardHyperParamsBuffer;
  madrona::RandKey initRandKey;
  uint32_t numAgents;
};

// The Sim class encapsulates the per-world state of the simulation.
// Sim is always available by calling ctx.data() given a reference
// to the Engine / Context object that is passed to each ECS system.
//
// Per-World state that is frequently accessed but only used by a few
// ECS systems should be put in a singleton component rather than
// in this class in order to ensure efficient access patterns.
struct Sim : public madrona::WorldBase {
  struct WorldInit {};

  // Sim::registerTypes is called during initialization
  // to register all components & archetypes with the ECS.
  static void registerTypes(madrona::ECSRegistry &registry,
                            const TaskConfig &cfg);

  // Sim::setupTasks is called during initialization to build
  // the system task graph that will be invoked by the 
  // Manager class (src/mgr.hpp) for each step.
  static void setupTasks(madrona::TaskGraphManager &taskgraph_mgr,
                         const TaskConfig &cfg);

  // The constructor is called for each world during initialization.
  // TaskConfig is global across all worlds, while WorldInit (src/init.hpp)
  // can contain per-world initialization data, created in (src/mgr.cpp)
  Sim(Engine &ctx,
      const TaskConfig &cfg,
      const WorldInit &init);

  RewardHyperParams *rewardHyperParams;
  madrona::RandKey initRandKey;
  madrona::RNG rng;

  uint32_t numAgents;

  SimFlags flags = SimFlags::InterpretAddAsReplace;
};

class Engine : public ::madrona::CustomContext<Engine, Sim> {
public:
    using CustomContext::CustomContext;
};

}

#include "sim.inl"
