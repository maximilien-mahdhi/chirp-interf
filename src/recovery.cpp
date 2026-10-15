/// @file recovery.cpp
/// @brief Implementation of the recovery techniques RT1, RT2 and RT3 (see recovery.hpp).

#include "chirp/recovery.hpp"

#include <stdexcept>

#include "detail.hpp"
#include "frames.hpp"

namespace chirp {
namespace {

/// @brief Analyse, hard-threshold and invert every frame m_min..m_max.
///
/// The STFT itself is never stored: each frame goes through
/// window -> FFT -> threshold -> inverse FFT inside a single thread.
///
/// @return row-major n_frames x (2M+1) matrix U with U[i][d+M] = u_{m_min+i}[d]
std::vector<cplx> synthesise_frames(SignalView f, WindowView g, long N, long R, long p,
                                    long m_min, long m_max, double threshold) {
    const long M = g.half();
    const long W = 2 * M + 1;
    const long n_frames = m_max - m_min + 1;
    const detail::Fft fft(static_cast<std::size_t>(N));   // plan shared (read-only) by all threads
    std::vector<cplx> U(static_cast<std::size_t>(n_frames * W));

#pragma omp parallel
    {
        detail::Workspace ws(fft);                          // per-thread buffers
#pragma omp for schedule(static)
        for (long i = 0; i < n_frames; ++i) {
            detail::analyse(f, g, (m_min + i) * R + p, fft, ws);
            detail::synthesise(M, threshold, fft, ws, U.data() + i * W);
        }
    }
    return U;
}

/// Dual window used to combine the frames.
enum class Dual {
    LeastSquares,   ///< RT2: weights g, normalisation sum g^2
    OverlapAdd,     ///< RT3: weights 1, normalisation sum g
};

/// @brief Common implementation of RT2 and RT3.
///
/// Every frame whose support meets [0, L) is used, including frames centred
/// outside the signal, so samples near the edges are covered like interior ones.
std::vector<cplx> overlap_add(SignalView f, WindowView g, long N, long R, long p,
                              double threshold, Dual dual) {
    detail::check_params(g, N, R, p);
    const long M = g.half();
    const long W = 2 * M + 1;
    const long L = static_cast<long>(f.size);
    if (R > W) throw std::invalid_argument("RT2/RT3 need R <= 2M+1 so every sample is covered");

    // Frames m with |n - mR - p| <= M for some n in [0, L).
    const long m_min = ceil_div(-M - p, R);
    const long m_max = floor_div(L - 1 + M - p, R);
    const std::vector<cplx> U = synthesise_frames(f, g, N, R, p, m_min, m_max, threshold);

    std::vector<cplx> out(static_cast<std::size_t>(L));
#pragma omp parallel for schedule(static)
    for (long n = 0; n < L; ++n) {
        cplx num(0.0);
        double den = 0.0;
        // Frames covering sample n: n - M <= mR + p <= n + M.
        for (long m = ceil_div(n - M - p, R); m <= floor_div(n + M - p, R); ++m) {
            const long d = n - m * R - p;        // offset from the frame centre, |d| <= M
            const double gd = g.data[d + M];
            const cplx u = U[(m - m_min) * W + d + M];
            if (dual == Dual::LeastSquares) {
                num += gd * u;
                den += gd * gd;
            } else {
                num += u;
                den += gd;
            }
        }
        out[n] = den != 0.0 ? num / den : cplx(0.0);
    }
    return out;
}

}  // namespace

std::vector<cplx> rt1(SignalView f, WindowView g, long N, long R, long p, double threshold) {
    detail::check_params(g, N, R, p);
    const long M = g.half();
    const long L = static_cast<long>(f.size);
    // Sample n = mR + q is read at offset d = q - p of frame m: it must lie in the window...
    if (p > M || R - 1 - p > M)
        throw std::invalid_argument("RT1 needs |q - p| <= M for all 0 <= q < R");
    // ...and the window value we divide by must be non-zero.
    for (long q = 0; q < R; ++q)
        if (g.data[q - p + M] == 0.0)
            throw std::invalid_argument("RT1 divides by g[q-p], which is zero for some q");
    if (L == 0) return {};

    const long m_max = (L - 1) / R;   // last frame needed (its centre may lie past L-1)
    const std::vector<cplx> U = synthesise_frames(f, g, N, R, p, 0, m_max, threshold);
    const long W = 2 * M + 1;

    std::vector<cplx> out(static_cast<std::size_t>(L));
#pragma omp parallel for schedule(static)
    for (long n = 0; n < L; ++n) {
        const long m = n / R, d = n % R - p;
        out[n] = U[m * W + d + M] / g.data[d + M];
    }
    return out;
}

std::vector<cplx> rt2(SignalView f, WindowView g, long N, long R, long p, double threshold) {
    return overlap_add(f, g, N, R, p, threshold, Dual::LeastSquares);
}

std::vector<cplx> rt3(SignalView f, WindowView g, long N, long R, long p, double threshold) {
    return overlap_add(f, g, N, R, p, threshold, Dual::OverlapAdd);
}

}  // namespace chirp
