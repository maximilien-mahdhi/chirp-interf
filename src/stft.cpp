/// @file stft.cpp
/// @brief Pointwise and frame-wise STFT (see stft.hpp).
///
/// Every coefficient is evaluated directly from the definition
///   V_g f(t, w) = sum_{n=-M}^{M} f[t+n] g[n] exp(-2 i pi w n),
/// at a cost of O(2M+1) per coefficient. Frames are independent, so they are
/// computed in parallel with OpenMP.

#include "chirp/stft.hpp"

#include <cmath>

#include "detail.hpp"

namespace chirp {

cplx stft_point(SignalView f, WindowView g, long t, double w) {
    detail::check_window(g);
    const long M = g.half();
    const long L = static_cast<long>(f.size);
    double re = 0.0, im = 0.0;   // separate accumulators so that the loop can be vectorised
#pragma omp simd reduction(+ : re, im)
    for (long n = -M; n <= M; ++n) {
        const long idx = t + n;
        if (idx < 0 || idx >= L) continue;  // signal edges: zero padding
        const cplx v = f.data[idx] * g.data[n + M] * std::polar(1.0, -2.0 * M_PI * w * double(n));
        re += v.real();
        im += v.imag();
    }
    return {re, im};
}

long n_frames_inside(long L, long R, long p) {
    if (L <= p) return 0;
    return (L - 1 - p) / R + 1;
}

std::vector<cplx> stft_frames(SignalView f, WindowView g, long N,
                              long R, long p, long m_min, long m_max) {
    detail::check_params(g, N, R, p);
    if (m_max < m_min) return {};
    const long n_frames = m_max - m_min + 1;
    std::vector<cplx> V(static_cast<std::size_t>(n_frames * N));

#pragma omp parallel for schedule(static)
    for (long i = 0; i < n_frames; ++i) {
        const long t = (m_min + i) * R + p;
        for (long k = 0; k < N; ++k) V[i * N + k] = stft_point(f, g, t, double(k) / double(N));
    }
    return V;
}

}  // namespace chirp
