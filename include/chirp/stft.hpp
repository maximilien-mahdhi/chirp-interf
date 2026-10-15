#pragma once
/// @file stft.hpp
/// @brief Discrete short-time Fourier transform on a sub-sampled time lattice.
///
/// Conventions shared by the whole library:
///   - `f`      : complex signal of length L, zero outside [0, L);
///   - `g`      : real window of odd length 2M+1, centred (`g.data[n + M]` is the value at offset n);
///   - lattice  : frame centres t_m = m R + p, with hop R >= 1 and shift 0 <= p < R.
///
///   V_g f(t, w) = sum_{n=-M}^{M} f[t+n] g[n] exp(-2 i pi w n)
///
/// The frequency axis is discretised as w_k = k / N, k = 0..N-1. With N >= 2M+1
/// the inverse DFT over k recovers the windowed segment without aliasing.

#include <complex>
#include <cstddef>
#include <vector>

namespace chirp {

/// Complex scalar used throughout the library.
using cplx = std::complex<double>;

/// Read-only view of a complex signal (no ownership, no copy).
struct SignalView {
    const cplx *data;   ///< first sample
    std::size_t size;   ///< number of samples L
};

/// Read-only view of a real, centred window of odd length 2M+1.
struct WindowView {
    const double *data;  ///< g[-M], ..., g[M]
    std::size_t size;    ///< 2M+1 (must be odd)

    /// Half-length M.
    long half() const { return static_cast<long>((size - 1) / 2); }
};

/// @brief STFT at a single time-frequency point.
/// @param f signal
/// @param g window
/// @param t time index (any integer; samples outside [0, L) count as zero)
/// @param w normalised frequency, in cycles per sample
/// @return V_g f(t, w)
/// @throws std::invalid_argument if the window has even length
cplx stft_point(SignalView f, WindowView g, long t, double w);

/// @brief STFT on a range of lattice frames and on the frequency grid k / N.
/// @param f signal
/// @param g window of length 2M+1
/// @param N number of frequency bins (N >= 2M+1)
/// @param R hop (R >= 1)
/// @param p shift (0 <= p < R)
/// @param m_min first frame index (frame centre m_min R + p, may be outside the signal)
/// @param m_max last frame index (inclusive)
/// @return row-major (m_max - m_min + 1) x N matrix, V[i][k] = V_g f((m_min + i) R + p, k / N);
///         empty if m_max < m_min
/// @throws std::invalid_argument if the parameters are inconsistent
std::vector<cplx> stft_frames(SignalView f, WindowView g, long N,
                              long R, long p, long m_min, long m_max);

/// @brief Number of lattice points m R + p (m >= 0) inside [0, L).
/// @return floor((L - 1 - p) / R) + 1, or 0 if L <= p
long n_frames_inside(long L, long R, long p);

/// @brief Floor division, correct for negative numerators.
/// @param a numerator
/// @param b denominator, b > 0
inline long floor_div(long a, long b) { return a >= 0 ? a / b : -((-a + b - 1) / b); }

/// @brief Ceiling division, correct for negative numerators.
/// @param a numerator
/// @param b denominator, b > 0
inline long ceil_div(long a, long b) { return -floor_div(-a, b); }

}  // namespace chirp
