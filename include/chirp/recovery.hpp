#pragma once
/// @file recovery.hpp
/// @brief Signal recovery from a hard-thresholded, sub-sampled STFT.
///
/// Each technique observes V_g f(mR+p, k/N), sets to zero every coefficient with
/// |V| < threshold, inverts every frame,
///
///   u_m[d] = (1/N) sum_k V(mR+p, k/N) exp(2 i pi k d / N),   |d| <= M,
///
/// (equal to f[mR+p+d] g[d] when nothing is thresholded) and combines the frames:
///
///   RT1 : for n = mR + q,   f[n] = u_m[q-p] / g[q-p]
///         one frame per sample; cheapest, but sensitive to p once thresholded.
///
///   RT2 : f[n] = sum_m g[n-mR-p] u_m[n-mR-p] / sum_m g[n-mR-p]^2
///         least-squares (canonical) dual window.
///
///   RT3 : f[n] = sum_m u_m[n-mR-p] / sum_m g[n-mR-p]
///         plain overlap-add.
///
/// With threshold = 0 the three techniques are exact up to rounding.

#include <vector>

#include "chirp/stft.hpp"

namespace chirp {

/// @brief Recovery technique 1 (one frame per sample).
/// @param f signal of length L
/// @param g window of length 2M+1, with g[q-p] != 0 for all 0 <= q < R
/// @param N number of frequency bins (N >= 2M+1)
/// @param R hop, with max(p, R-1-p) <= M
/// @param p shift (0 <= p < R)
/// @param threshold coefficients with |V| < threshold are set to zero
/// @return recovered signal of length L
/// @throws std::invalid_argument if a constraint is violated
std::vector<cplx> rt1(SignalView f, WindowView g, long N, long R, long p, double threshold);

/// @brief Recovery technique 2 (least-squares dual window).
/// @param f signal of length L
/// @param g window of length 2M+1
/// @param N number of frequency bins (N >= 2M+1)
/// @param R hop, with R <= 2M+1 so that every sample is covered
/// @param p shift (0 <= p < R)
/// @param threshold coefficients with |V| < threshold are set to zero
/// @return recovered signal of length L
/// @throws std::invalid_argument if a constraint is violated
std::vector<cplx> rt2(SignalView f, WindowView g, long N, long R, long p, double threshold);

/// @brief Recovery technique 3 (plain overlap-add normalised by sum g).
/// @param f signal of length L
/// @param g window of length 2M+1 (non-negative)
/// @param N number of frequency bins (N >= 2M+1)
/// @param R hop, with R <= 2M+1
/// @param p shift (0 <= p < R)
/// @param threshold coefficients with |V| < threshold are set to zero
/// @return recovered signal of length L
/// @throws std::invalid_argument if a constraint is violated
std::vector<cplx> rt3(SignalView f, WindowView g, long N, long R, long p, double threshold);

}  // namespace chirp
