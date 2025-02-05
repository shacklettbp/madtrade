#pragma once

#include <cstdint>

namespace madtrade {

enum class SimFlags : uint32_t {
  Default                = 0,
  AutoReset              = 1 << 1, // Immediately generate new world on episode end
};

inline SimFlags & operator|=(SimFlags &a, SimFlags b);
inline SimFlags operator|(SimFlags a, SimFlags b);
inline SimFlags & operator&=(SimFlags &a, SimFlags b);
inline SimFlags operator&(SimFlags a, SimFlags b);

}

#include "sim_flags.inl"
