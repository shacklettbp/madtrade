set(SIMULATOR_SRCS
    sim.hpp sim.inl sim.cpp
    sim_flags.hpp sim_flags.inl
)

add_library(mad_trade_cpu_impl STATIC
    ${SIMULATOR_SRCS}
)

target_link_libraries(mad_trade_cpu_impl
    PUBLIC
        madrona_mw_core
    PRIVATE
        madrona_common
)

target_compile_definitions(mad_trade_cpu_impl PRIVATE
    -DDATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../data/"
)

add_library(mad_trade_mgr STATIC
    mgr.hpp mgr.cpp
)

target_link_libraries(mad_trade_mgr 
    PUBLIC
        madrona_python_utils
    PRIVATE
        mad_trade_cpu_impl
        madrona_mw_cpu
        madrona_common
)

if (TARGET madrona_cuda)
    target_link_libraries(mad_trade_mgr
        PRIVATE madrona_cuda
    )
endif ()

if (TARGET madrona_mw_gpu)
    madrona_build_compile_defns(
        OUT_TARGET
            mad_trade_gpu_srcs
        SOURCES_DEFN
            GPU_HIDESEEK_SRC_LIST
        FLAGS_DEFN
            GPU_HIDESEEK_COMPILE_FLAGS 
        SRCS
            ${SIMULATOR_SRCS}
    )

    target_link_libraries(mad_trade_mgr PRIVATE
        mad_trade_gpu_srcs
        madrona_mw_gpu
    )

    target_compile_definitions(mad_trade_mgr PRIVATE
        -DENABLE_MWGPU=1
    )
endif ()

target_compile_definitions(mad_trade_mgr PRIVATE
    -DDATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../data/"
)

madrona_python_module(mad_trade
    bindings.cpp
)

target_link_libraries(mad_trade PRIVATE
    mad_trade_mgr
)
