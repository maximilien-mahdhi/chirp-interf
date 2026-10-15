/// @file stft.cpp
/// @brief Pointwise and frame-wise STFT (see stft.hpp).

#include "chirp/stft.hpp"

#include <algorithm>
#include <cmath>

#include "detail.hpp"
#include "frames.hpp"

namespace chirp {

// Direct O(2M+1) evaluation; used for arbitrary (off-grid) frequencies and in the tests.
cplx stft_point(SignalView f, WindowView g, long t, double w) {
    detail::check_window(g);
    const long M = g.half();
    const long L = static_cast<long>(f.size);
    double re = 0.0, im = 0.0;   // separate accumulators: cheaper than std::complex +=
    for (long n = -M; n <= M; ++n) {
        const long idx = t + n;
        if (idx < 0 || idx >= L) continue;  // zero padding
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
    const detail::Fft fft(static_cast<std::size_t>(N));   // plan shared (read-only) by all threads
    std::vector<cplx> V(static_cast<std::size_t>(n_frames * N));

    // One FFT per frame, O(n_frames * N log N); frames are independent.
#pragma omp parallel
    {
        detail::Workspace ws(fft);                          // per-thread buffers
#pragma omp for schedule(static)
        for (long i = 0; i < n_frames; ++i) {
            detail::analyse(f, g, (m_min + i) * R + p, fft, ws);
            std::copy(ws.buf.begin(), ws.buf.end(), V.begin() + i * N);
        }
    }
    return V;
}

}  // namespace chirp
