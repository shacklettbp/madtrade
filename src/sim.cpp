#include <madrona/mw_gpu_entry.hpp>

#include <madrona/registry.hpp>

#include <cstdint>

#include "sim.hpp"

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

  registry.registerComponent<OrderObservation>();

  registry.registerArchetype<Agent>();

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
  Market &market = ctx.singleton<Market>();

  for (int32_t i = 0; i < K; i++) {
    market.orders[i] = {
      .type = OrderType::Buy,
      .size = 0,
      .price = 0,
    };
  }
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

inline void actionsSystem(Engine &ctx,
                          Action action)
{
  Market &market = ctx.singleton<Market>();

  // Take action, create order in Market singleton
  
  action.bid;

  i32 order_idx = market.currentOrder.fetch_add_relaxed(1);
}

inline void fillOrderObservationsSystem(Engine &ctx,
                                        const PlayerState &player_state,
                                        OrderObservation &obs)
{
  (void)ctx;
  (void)player_state;
  (void)obs;


  Market &market = ctx.singleton<Market>();

  for (int i = 0; i < K; i++) {
    obs.orders[i] = market.orders[i];
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
      OrderObservation
    >>({reset_sys});

  return obs_sys;
}

inline void matchOrdersSystem(Engine &ctx, Market &market)
{
  (void)market;

  // Iterate over the market and match orders??

  for (int i = 0; i < K; i++) {
    
  }
}

static void setupInitTasks(TaskGraphBuilder &builder, const TaskConfig &cfg)
{
  resetAndObsTasks(builder, cfg, {});
}

static void setupStepTasks(TaskGraphBuilder &builder, const TaskConfig &cfg)
{
  auto action_sys = builder.addToGraph<ParallelForNode<Engine,
    actionsSystem,
      Action
    >>({});

  auto match_orders = builder.addToGraph<ParallelForNode<Engine,
    matchOrdersSystem,
      Market
    >>({action_sys});

  auto reward_sys = builder.addToGraph<ParallelForNode<Engine,
    rewardSystem,
      AgentPolicy,
      PlayerState,
      Reward
    >>({match_orders});

  // Set done values if match is finished
  auto done_sys = builder.addToGraph<ParallelForNode<Engine,
    doneSystem,
      Done
    >>({reward_sys});

  resetAndObsTasks(builder, cfg, { done_sys });
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

  ctx.singleton<Market>() = {};
  initWorld(ctx);
}


MADRONA_BUILD_MWGPU_ENTRY(Engine, Sim, TaskConfig, Sim::WorldInit);

}
