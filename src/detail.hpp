#pragma once
/// @file detail.hpp
/// @brief Internal helpers: modular arithmetic and parameter validation (not part of the public API).

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "chirp/stft.hpp"

namespace chirp::detail {

/// @brief Mathematical modulo: result in [0, n) even for negative a.
inline long mod(long a, long n) {
    long r = a % n;
    return r < 0 ? r + n : r;
}

/// @brief Throw std::invalid_argument unless the window has odd, non-zero length.
inline void check_window(WindowView g) {
    if (g.size == 0 || g.size % 2 == 0)
        throw std::invalid_argument("window length must be odd (2M+1)");
}

/// @brief Validate the STFT parameters shared by every routine.
/// @throws std::invalid_argument if the window length is even, R < 1, p is outside [0, R)
///         or N < 2M+1 (the inverse DFT would alias)
inline void check_params(WindowView g, long N, long R, long p) {
    check_window(g);
    if (R < 1) throw std::invalid_argument("hop R must be >= 1");
    if (p < 0 || p >= R) throw std::invalid_argument("shift p must satisfy 0 <= p < R");
    if (N < static_cast<long>(g.size))
        throw std::invalid_argument("N must be >= 2M+1 (got N=" + std::to_string(N) +
                                    ", 2M+1=" + std::to_string(g.size) + ")");
}

}  // namespace chirp::detail
