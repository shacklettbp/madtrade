#include "mgr.hpp"
#include "sim.hpp"

#include <madrona/utils.hpp>
#include <madrona/tracing.hpp>
#include <madrona/mw_cpu.hpp>

#include <cassert>

#ifdef MADRONA_LINUX
#include <unistd.h>
#endif

#ifdef MADRONA_CUDA_SUPPORT
#include <madrona/mw_gpu.hpp>
#include <madrona/cuda_utils.hpp>
#endif

using namespace madrona;
using namespace madrona::math;
using namespace madrona::py;

namespace madtrade {

struct Manager::Impl {
  Config cfg;
  uint32_t numAgentsPerWorld;
  SimControl *simCtrl;
  RewardHyperParams *rewardHyperParams;
  TrainInterface trainInterface;

  inline Impl(const Manager::Config &mgr_cfg,
              SimControl *sim_ctrl,
              RewardHyperParams *reward_hyper_params)
    : cfg(mgr_cfg),
      numAgentsPerWorld(cfg.numAgentsPerWorld),
      simCtrl(sim_ctrl),
      rewardHyperParams(reward_hyper_params),
      trainInterface()
  {}

  inline virtual ~Impl() {}

  virtual void run(TaskGraphID taskgraph_id = TaskGraphID::Step) = 0;

#ifdef MADRONA_CUDA_SUPPORT
  virtual void gpuStreamInit(cudaStream_t strm, void **buffers) = 0;
  virtual void gpuStreamStep(cudaStream_t strm, void **buffers) = 0;
#endif

  virtual Tensor exportTensor(ExportID slot,
                              TensorElementType type,
                              madrona::Span<const int64_t> dimensions) const = 0;

  virtual Tensor rewardHyperParamsTensor() const = 0;
  virtual Tensor simControlTensor() const = 0;

  static inline Impl * init(const Config &cfg);

  virtual Action * getActions() = 0;
};

struct Manager::CPUImpl final : Manager::Impl {
  using TaskGraphT =
      TaskGraphExecutor<Engine, Sim, TaskConfig, Sim::WorldInit>;

  TaskGraphT cpuExec;
  Action *actions;

  inline CPUImpl(const Manager::Config &mgr_cfg,
                 SimControl *sim_ctrl,
                 RewardHyperParams *reward_hyper_params,
                 TaskGraphT &&cpu_exec,
                 Action *actions)
    : Impl(mgr_cfg, sim_ctrl, reward_hyper_params),
      cpuExec(std::move(cpu_exec)),
      actions(actions)
  {}

  inline virtual ~CPUImpl() final
  {
    free(rewardHyperParams);
  }

  inline virtual void run(TaskGraphID graph_id)
  {
    cpuExec.runTaskGraph(graph_id);
  }

#ifdef MADRONA_CUDA_SUPPORT
  virtual void gpuStreamInit(cudaStream_t, void **)
  {
    assert(false);
  }

  virtual void gpuStreamStep(cudaStream_t, void **)
  {
    assert(false);
  }
#endif

  virtual Tensor rewardHyperParamsTensor() const final
  {
    return Tensor(rewardHyperParams, TensorElementType::Float32,
                  {
                    cfg.numPBTPolicies,
                    sizeof(RewardHyperParams) / sizeof(float),
                  }, Optional<int>::none());
  }

  virtual Tensor simControlTensor() const final
  {
    return Tensor(simCtrl, TensorElementType::Int32,
                  {
                    sizeof(SimControl) / sizeof(int32_t),
                  }, Optional<int>::none());
  }

  virtual inline Tensor exportTensor(ExportID slot,
                                     TensorElementType type,
                                     madrona::Span<const int64_t> dims) const final
  {
    void *dev_ptr = cpuExec.getExported((uint32_t)slot);
    return Tensor(dev_ptr, type, dims, Optional<int>::none());
  }

  virtual Action * getActions() final
  {
    return actions;
  }
};

#if defined(MADRONA_CUDA_SUPPORT) && defined(ENABLE_MWGPU)
struct Manager::CUDAImpl final : Manager::Impl {
  MWCudaExecutor gpuExec;
  MWCudaLaunchGraph stepGraph;

  inline CUDAImpl(const Manager::Config &mgr_cfg,
                  SimControl *sim_ctrl,
                  RewardHyperParams *reward_hyper_params,
                  MWCudaExecutor &&gpu_exec)
    : Impl(mgr_cfg, sim_ctrl, reward_hyper_params),
      gpuExec(std::move(gpu_exec)),
      stepGraph(gpuExec.buildLaunchGraph(TaskGraphID::Step))
  {
  }

  inline virtual ~CUDAImpl() final
  {
    REQ_CUDA(cudaFree(rewardHyperParams));
  }

  inline virtual void run(TaskGraphID graph_id)
  {
    assert(graph_id == TaskGraphID::Step);
    gpuExec.run(stepGraph);
  }

  virtual void gpuStreamInit(cudaStream_t strm, void **buffers)
  {
    auto init_graph = gpuExec.buildLaunchGraph(TaskGraphID::Init);
    gpuExec.runAsync(init_graph, strm);

    trainInterface.cudaCopyObservations(strm, buffers);
  }

  virtual void gpuStreamStep(cudaStream_t strm, void **buffers)
  {
    buffers = trainInterface.cudaCopyStepInputs(strm, buffers);

    gpuExec.runAsync(stepGraph, strm);

    trainInterface.cudaCopyStepOutputs(strm, buffers);
  }

  virtual Tensor rewardHyperParamsTensor() const final
  {
    return Tensor(rewardHyperParams, TensorElementType::Float32,
                  {
                    cfg.numPBTPolicies,
                    sizeof(RewardHyperParams) / sizeof(float),
                  }, cfg.gpuID);
  }

  virtual Tensor simControlTensor() const final
  {
    return Tensor(simCtrl, TensorElementType::Int32,
                  {
                    sizeof(SimControl) / sizeof(int32_t),
                  }, cfg.gpuID);
  }

  virtual inline Tensor exportTensor(ExportID slot,
                                     TensorElementType type,
                                     madrona::Span<const int64_t> dims) const final
  {
    void *dev_ptr = gpuExec.getExported((uint32_t)slot);
    return Tensor(dev_ptr, type, dims, cfg.gpuID);
  }

  virtual Action * getActions() final
  {
    // TODO: Implement
    assert(false);
    return nullptr;
  }
};
#endif

Manager::Impl * Manager::Impl::init(
    const Manager::Config &mgr_cfg)
{
#if defined(MADRONA_CUDA_SUPPORT) && defined(ENABLE_MWGPU)
  CUcontext cu_ctx = MWCudaExecutor::initCUDA(mgr_cfg.gpuID);
#endif

  SimControl *sim_ctrl;
  {
    sim_ctrl = (SimControl *)malloc(
        sizeof(SimControl));

    new (sim_ctrl) SimControl {
    };
  }

  RewardHyperParams *reward_hyper_params;

  switch (mgr_cfg.execMode) {
    case ExecMode::CUDA: {
#if defined(MADRONA_CUDA_SUPPORT) && defined(ENABLE_MWGPU)
      if (mgr_cfg.numPBTPolicies > 0) {
        reward_hyper_params = (RewardHyperParams *)cu::allocGPU(
            sizeof(RewardHyperParams) * mgr_cfg.numPBTPolicies);
      } else {
        reward_hyper_params = (RewardHyperParams *)cu::allocGPU(
            sizeof(RewardHyperParams));

        RewardHyperParams default_reward_hyper_params {};

        REQ_CUDA(cudaMemcpy(reward_hyper_params,
                            &default_reward_hyper_params,
                            sizeof(RewardHyperParams),
                            cudaMemcpyHostToDevice));
      }

      SimControl *gpu_sim_ctrl = (SimControl *)cu::allocGPU(
          sizeof(SimControl));
      cudaMemcpy(gpu_sim_ctrl, sim_ctrl,
                 sizeof(SimControl), cudaMemcpyHostToDevice);

      free(sim_ctrl);
      sim_ctrl = gpu_sim_ctrl;
#else
      FATAL("No CUDA");
#endif
    } break;
    case ExecMode::CPU: {
      if (mgr_cfg.numPBTPolicies > 0) {
        reward_hyper_params = (RewardHyperParams *)malloc(
            sizeof(RewardHyperParams) * mgr_cfg.numPBTPolicies);
      } else {
        reward_hyper_params = (RewardHyperParams *)malloc(
            sizeof(RewardHyperParams));

        *(reward_hyper_params) = RewardHyperParams {};
      }
    } break;
    default: {
        assert(false);
        MADRONA_UNREACHABLE();
    } break;
  }

  TaskConfig task_cfg {
    .rewardHyperParamsBuffer = reward_hyper_params,
    // Temporarily just setting the seed to 0
    .initRandKey = rand::initKey(0),
    .numAgents = mgr_cfg.numAgentsPerWorld,
    .numNPCs = mgr_cfg.numNPCsPerWorld,
    .D = mgr_cfg.D,
    .simFlags = mgr_cfg.simFlags,
    .settlementPrice = mgr_cfg.settlementPrice
  };

  switch (mgr_cfg.execMode) {
    case ExecMode::CUDA: {
#if defined(MADRONA_CUDA_SUPPORT) && defined(ENABLE_MWGPU)
      HeapArray<Sim::WorldInit> world_inits(mgr_cfg.numWorlds);

#ifdef MADRONA_LINUX
      {
        char *env = getenv("MAD_TRADE_DEBUG_WAIT");

        if (env && env[0] == '1') {
            volatile int done = 0;
            while (!done) { sleep(1); }
        }
      }
#endif

      MWCudaExecutor gpu_exec({
          .worldInitPtr = world_inits.data(),
          .numWorldInitBytes = sizeof(Sim::WorldInit),
          .userConfigPtr = (void *)&task_cfg,
          .numUserConfigBytes = sizeof(TaskConfig),
          .numWorldDataBytes = sizeof(Sim),
          .worldDataAlignment = alignof(Sim),
          .numWorlds = mgr_cfg.numWorlds,
          .numTaskGraphs = (uint32_t)TaskGraphID::NumGraphs,
          .numExportedBuffers = (uint32_t)ExportID::NumExports, 
      }, {
          { GPU_HIDESEEK_SRC_LIST },
          { GPU_HIDESEEK_COMPILE_FLAGS },
          CompileConfig::OptMode::LTO,
      }, cu_ctx);

      return new CUDAImpl {
          mgr_cfg,
          sim_ctrl,
          reward_hyper_params,
          std::move(gpu_exec),
      };
#else
      FATAL("Madrona was not compiled with CUDA support");
#endif
    } break;
    case ExecMode::CPU: {
      HeapArray<Sim::WorldInit> world_inits(mgr_cfg.numWorlds);

      CPUImpl::TaskGraphT cpu_exec {
          ThreadPoolExecutor::Config {
              .numWorlds = mgr_cfg.numWorlds,
              .numExportedBuffers = (uint32_t)ExportID::NumExports,
          },
          task_cfg,
          world_inits.data(),
          (CountT)TaskGraphID::NumGraphs,
      };

      Action *actions = (Action *)cpu_exec.getExported(
          (uint32_t)ExportID::Action);

      auto cpu_impl = new CPUImpl {
          mgr_cfg,
          sim_ctrl,
          reward_hyper_params,
          std::move(cpu_exec),
          actions
      };

      return cpu_impl;
    } break;
    default: MADRONA_UNREACHABLE();
  }
}

Manager::Manager(const Config &cfg)
    : impl_(Impl::init(cfg))
{
  impl_->trainInterface = trainInterface();
}

Manager::~Manager() {}

void Manager::init()
{
  impl_->run(TaskGraphID::Init);
}

void Manager::step()
{
  impl_->run(TaskGraphID::Step);
}

#ifdef MADRONA_CUDA_SUPPORT
void Manager::gpuStreamInit(cudaStream_t strm, void **buffers)
{
  impl_->gpuStreamInit(strm, buffers);
}

void Manager::gpuStreamStep(cudaStream_t strm, void **buffers)
{
  impl_->gpuStreamStep(strm, buffers);
}
#endif

Tensor Manager::resetTensor() const
{
  return impl_->exportTensor(ExportID::Reset,
                             TensorElementType::Int32,
                             {
                               impl_->cfg.numWorlds,
                               sizeof(WorldReset) / sizeof(int32_t),
                             });
}

Tensor Manager::simControlTensor() const
{
  return impl_->simControlTensor();
}

Tensor Manager::buySellActionTensor() const
{
  return impl_->exportTensor(ExportID::Action,
                             TensorElementType::Int32,
                             {
                               impl_->cfg.numWorlds * impl_->numAgentsPerWorld,
                               sizeof(Action) / sizeof(uint32_t),
                             });
}

Tensor Manager::rewardTensor() const
{
  return impl_->exportTensor(ExportID::Reward, TensorElementType::Float32,
                             {
                               impl_->cfg.numWorlds * impl_->numAgentsPerWorld,
                               1,
                             });
}

Tensor Manager::doneTensor() const
{
  return impl_->exportTensor(ExportID::Done, TensorElementType::Int32,
                             {
                               impl_->cfg.numWorlds * impl_->numAgentsPerWorld,
                               1,
                             });
}

Tensor Manager::policyAssignmentTensor() const
{
  return impl_->exportTensor(ExportID::AgentPolicy,
                             TensorElementType::Int32,
                             {
                               impl_->cfg.numWorlds * impl_->numAgentsPerWorld,
                               1,
                             });
}

Tensor Manager::observationTensor() const
{
  return impl_->exportTensor(ExportID::Observation,
                             TensorElementType::Int32,
                             {
                               impl_->cfg.numWorlds * impl_->numAgentsPerWorld,
                               OBSERVATION_HISTORY_LEN,
                               sizeof(TimeStepObservation) / sizeof(int32_t),
                             });
}

Tensor Manager::rewardHyperParamsTensor() const
{
  return impl_->rewardHyperParamsTensor();
}

Tensor Manager::matchResultTensor() const
{
  return impl_->exportTensor(ExportID::MatchResult,
                             TensorElementType::Int32,
                             {
                               impl_->cfg.numWorlds,
                               sizeof(MatchResult) / sizeof(int32_t),
                             });
}

TrainInterface Manager::trainInterface() const
{
  auto pbt_inputs = std::to_array<NamedTensor>({
    { "policy_assignments", policyAssignmentTensor() },
  });

  return TrainInterface {
    {
      .actions = {
        { "buy_sell", buySellActionTensor() },
      },
      .resets = resetTensor(),
      .simCtrl = simControlTensor(),
      .pbt = impl_->cfg.numPBTPolicies > 0 ?
        pbt_inputs : Span<const NamedTensor>(nullptr, 0),
    },
    {
      .observations = {
        { "obs", observationTensor() },
      },
      .rewards = rewardTensor(),
      .dones = doneTensor(),
      .pbt = {
        { "episode_results", matchResultTensor() },
      },
    },
  };
}

ExecMode Manager::execMode() const
{
  return impl_->cfg.execMode;
}

void Manager::setAction(
    uint32_t world_id,
    uint32_t agent_id,
    Action action)
{
  if (impl_->cfg.execMode == ExecMode::CPU) {
    Action *actions = impl_->getActions();

    actions[world_id * impl_->numAgentsPerWorld + agent_id] = action;
  } else {
    assert(false);
  }
}

}
