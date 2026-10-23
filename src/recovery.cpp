/// @file recovery.cpp
/// @brief Implementation of the recovery techniques RT1, RT2 and RT3 (see recovery.hpp).
///
/// The thresholded STFT is computed once on every frame that is needed, then
/// each sample n is rebuilt from the inverse DFT of the frames that cover it:
///
///   u_m(d) = (1/N) sum_k V(mR+p, k/N) exp(2 i pi k d / N),   d = n - mR - p.

#include "chirp/recovery.hpp"

#include <cmath>
#include <stdexcept>

#include "detail.hpp"

namespace chirp {

namespace {

/// @brief Thresholded STFT on the frames m_min..m_max.
/// @return row-major n_frames x N matrix; coefficients with |V| < threshold are set to zero
std::vector<cplx> thresholded_stft(SignalView f, WindowView g, long N, long R, long p,
                                   long m_min, long m_max, double threshold) {
    std::vector<cplx> V = stft_frames(f, g, N, R, p, m_min, m_max);
    if (threshold > 0.0)
        for (auto &v : V)
            if (std::abs(v) < threshold) v = 0.0;
    return V;
}

/// @brief Inverse DFT of one frame at offset d from its centre (frame = N coefficients).
cplx inverse_at(const cplx *frame, long N, long d) {
    double re = 0.0, im = 0.0;
    for (long k = 0; k < N; ++k) {
        const cplx v = frame[k] * std::polar(1.0, 2.0 * M_PI * double(k) * double(d) / double(N));
        re += v.real();
        im += v.imag();
    }
    return cplx(re, im) / double(N);
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
    const long L = static_cast<long>(f.size);
    if (R > 2 * M + 1) throw std::invalid_argument("RT2/RT3 need R <= 2M+1 so every sample is covered");

    // Frames m with |n - mR - p| <= M for some n in [0, L).
    const long m_min = ceil_div(-M - p, R);
    const long m_max = floor_div(L - 1 + M - p, R);
    const std::vector<cplx> V = thresholded_stft(f, g, N, R, p, m_min, m_max, threshold);

    std::vector<cplx> out(static_cast<std::size_t>(L));
#pragma omp parallel for schedule(static)
    for (long n = 0; n < L; ++n) {
        cplx num(0.0);
        double den = 0.0;
        // Frames covering sample n: n - M <= mR + p <= n + M.
        for (long m = ceil_div(n - M - p, R); m <= floor_div(n + M - p, R); ++m) {
            const long d = n - m * R - p;        // offset from the frame centre, |d| <= M
            const double gd = g.data[d + M];
            const cplx u = inverse_at(V.data() + (m - m_min) * N, N, d);
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
    const std::vector<cplx> V = thresholded_stft(f, g, N, R, p, 0, m_max, threshold);

    std::vector<cplx> out(static_cast<std::size_t>(L));
#pragma omp parallel for schedule(static)
    for (long n = 0; n < L; ++n) {
        const long m = n / R, d = n % R - p;     // n = mR + q, d = q - p
        out[n] = inverse_at(V.data() + m * N, N, d) / g.data[d + M];
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
