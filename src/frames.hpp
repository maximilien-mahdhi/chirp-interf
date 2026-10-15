#pragma once
/// @file frames.hpp
/// @brief Per-frame analysis / synthesis kernels shared by the STFT and the recovery code.

#include <algorithm>
#include <vector>

#include "chirp/stft.hpp"
#include "fft.hpp"

namespace chirp::detail {

/// Per-thread buffers: one frame of N coefficients and the FFT scratch space.
struct Workspace {
    std::vector<cplx> buf;      ///< frame being processed (N values)
    std::vector<cplx> scratch;  ///< Bluestein scratch (empty for powers of two)
    explicit Workspace(const Fft &fft) : buf(fft.size()), scratch(fft.scratch_size()) {}
};

/// @brief ws.buf <- V_g f(t, k/N) for k = 0..N-1.
///
/// Windows the segment around t, places offset n in bin n mod N (no aliasing
/// because N >= 2M+1, so the 2M+1 offsets fall in distinct bins), then applies
/// one forward FFT. Samples outside [0, L) count as zero.
inline void analyse(SignalView f, WindowView g, long t, const Fft &fft, Workspace &ws) {
    const long N = static_cast<long>(fft.size());
    const long M = g.half();
    const long L = static_cast<long>(f.size);
    std::fill(ws.buf.begin(), ws.buf.end(), cplx(0.0));
    for (long n = -M; n <= M; ++n) {
        const long idx = t + n;
        if (idx >= 0 && idx < L) ws.buf[n >= 0 ? n : n + N] = f.data[idx] * g.data[n + M];
    }
    fft.forward(ws.buf.data(), ws.scratch.data());
}

/// @brief Hard-threshold ws.buf, invert it and write the 2M+1 useful samples.
///
/// Writes u[d + M] = (1/N) sum_k V_k exp(2 i pi k d / N) for |d| <= M, where V
/// is ws.buf after zeroing every coefficient of modulus below `threshold`.
/// The comparison is done on squared moduli to avoid a square root per bin.
inline void synthesise(long M, double threshold, const Fft &fft, Workspace &ws, cplx *u) {
    const long N = static_cast<long>(fft.size());
    if (threshold > 0.0) {
        const double t2 = threshold * threshold;
        for (auto &v : ws.buf)
            if (std::norm(v) < t2) v = 0.0;
    }
    fft.inverse(ws.buf.data(), ws.scratch.data());
    const double inv = 1.0 / double(N);
    for (long d = -M; d <= M; ++d) u[d + M] = ws.buf[d >= 0 ? d : d + N] * inv;
}

}  // namespace chirp::detail
