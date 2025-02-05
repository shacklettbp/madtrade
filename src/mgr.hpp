#pragma once

#include <memory>

#include <madrona/py/utils.hpp>
#include <madrona/exec_mode.hpp>

#include "sim_flags.hpp"
#include "sim.hpp"

namespace madtrade {

// The Manager class encapsulates the linkage between the outside training
// code and the internal simulation state (src/sim.hpp / src/sim.cpp)
//
// Manager is responsible for initializing the simulator, loading physics
// and rendering assets off disk, and mapping ECS components to tensors
// for learning
class Manager {
public:
  struct Config {
    madrona::ExecMode execMode; // CPU or CUDA
    int gpuID; // Which GPU for CUDA backend?
    uint32_t numWorlds; // Simulation batch size
    uint32_t numAgentsPerWorld;
    SimFlags simFlags;
    uint32_t numPBTPolicies;
  };

  Manager(const Config &cfg);
  ~Manager();

  void init();
  void step();

  inline void cpuJAXInit(void **, void **) {};
  inline void cpuJAXStep(void **, void **) {};

#ifdef MADRONA_CUDA_SUPPORT
  void gpuStreamInit(cudaStream_t strm, void **buffers);
  void gpuStreamStep(cudaStream_t strm, void **buffers);
#endif

  // These functions export Tensor objects that link the ECS
  // simulation state to the python bindings / PyTorch tensors (src/bindings.cpp)
  madrona::py::Tensor resetTensor() const;
  madrona::py::Tensor simControlTensor() const;
  madrona::py::Tensor buySellActionTensor() const;
  madrona::py::Tensor rewardTensor() const;
  madrona::py::Tensor doneTensor() const;
  madrona::py::Tensor policyAssignmentTensor() const;
  madrona::py::Tensor rewardHyperParamsTensor() const;
  madrona::py::Tensor matchResultTensor() const;

  madrona::py::Tensor ordersObservationTensor() const;
  madrona::py::Tensor agentStateObservationTensor() const;

  madrona::py::TrainInterface trainInterface() const;

  madrona::ExecMode execMode() const;

private:
  struct Impl;
  struct CPUImpl;
  struct CUDAImpl;

  std::unique_ptr<Impl> impl_;
};

}
