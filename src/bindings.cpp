#include "mgr.hpp"

#include <madrona/py/bindings.hpp>

#include <filesystem>

namespace nb = nanobind;

namespace madtrade {

NB_MODULE(mad_trade, m) {
    madrona::py::setupMadronaSubmodule(m);

    nb::enum_<SimFlags>(m, "SimFlags", nb::is_arithmetic())
      .value("Default", SimFlags::Default)
      .value("AutoReset", SimFlags::AutoReset)
      .value("InterpretAddAsReplace", SimFlags::InterpretAddAsReplace)
    ;

    nb::class_<Manager> (m, "SimManager")
        .def("__init__", [](Manager *self,
                            madrona::py::PyExecMode exec_mode,
                            int64_t gpu_id,
                            int64_t num_worlds,
                            int64_t num_agents_per_world,
                            int64_t num_npcs_per_world,
                            int64_t D,
                            uint32_t settlement_price,
                            uint32_t sim_flags,
                            uint32_t num_pbt_policies) {
            new (self) Manager(Manager::Config {
                .execMode = exec_mode,
                .gpuID = (int)gpu_id,
                .numWorlds = (uint32_t)num_worlds,
                .numAgentsPerWorld = (uint32_t)num_agents_per_world,
                .numNPCsPerWorld = (uint32_t)num_npcs_per_world,
                .D = (uint32_t)D,
                .settlementPrice = settlement_price,
                .simFlags = SimFlags(sim_flags),
                .numPBTPolicies = num_pbt_policies,
            });
        }, nb::arg("exec_mode"),
           nb::arg("gpu_id"),
           nb::arg("num_worlds"),
           nb::arg("num_agents_per_world"),
           nb::arg("num_npcs_per_world"),
           nb::arg("D"),
           nb::arg("settlement_price"),
           nb::arg("sim_flags"),
           nb::arg("num_pbt_policies"))
        .def("step", &Manager::step)
        .def("reset_tensor", &Manager::resetTensor)
        .def("sim_ctrl_tensor", &Manager::simControlTensor)
        .def("reward_tensor", &Manager::rewardTensor)
        .def("done_tensor", &Manager::doneTensor)
        .def("policy_assignment_tensor", &Manager::policyAssignmentTensor)
        .def("obs_tensor", &Manager::observationTensor)
        .def("jax", madrona::py::JAXInterface::buildEntry<
                &Manager::trainInterface,
                &Manager::cpuJAXInit,
                &Manager::cpuJAXStep
#ifdef MADRONA_CUDA_SUPPORT
                ,
                &Manager::gpuStreamInit,
                &Manager::gpuStreamStep
#endif
             >())
    ;
}

}
