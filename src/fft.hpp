#pragma once
/// @file fft.hpp
/// @brief Minimal FFT for arbitrary sizes: iterative radix-2 for powers of two,
// Bluestein (chirp-z) on top of it otherwise. A plan is immutable and can be
// shared between threads; each thread brings its own scratch buffer.
//
// Complex products are written out by hand: std::complex operator* follows
// Annex G (NaN/inf recovery) and is several times slower without -ffast-math.

#include <cmath>
#include <cstddef>
#include <vector>

#include "chirp/stft.hpp"

namespace chirp::detail {

/// @brief Plain complex product (no Annex G NaN/inf handling).
inline cplx mul(cplx a, cplx b) {
    return {a.real() * b.real() - a.imag() * b.imag(), a.real() * b.imag() + a.imag() * b.real()};
}

/// @brief Iterative Cooley-Tukey FFT for power-of-two sizes.
///
/// The constructor precomputes the bit-reversal permutation and the n/2 twiddle factors.
class Radix2 {
public:
    /// @param n transform size, a power of two
    explicit Radix2(std::size_t n) : n_(n), rev_(n), tw_(n / 2) {
        std::size_t bits = 0;
        while ((std::size_t(1) << bits) < n) ++bits;
        for (std::size_t i = 0; i < n; ++i) {
            std::size_t r = 0;
            for (std::size_t b = 0; b < bits; ++b) r |= ((i >> b) & 1u) << (bits - 1 - b);
            rev_[i] = r;
        }
        for (std::size_t k = 0; k < n / 2; ++k) tw_[k] = std::polar(1.0, -2.0 * M_PI * double(k) / double(n));
    }

    /// @brief In-place forward DFT: X[k] = sum_j x[j] exp(-2 i pi j k / n).
    void forward(cplx *x) const {
        for (std::size_t i = 0; i < n_; ++i)
            if (i < rev_[i]) std::swap(x[i], x[rev_[i]]);
        for (std::size_t len = 2; len <= n_; len <<= 1) {
            const std::size_t half = len / 2, stride = n_ / len;
            for (std::size_t s = 0; s < n_; s += len)
                for (std::size_t j = 0; j < half; ++j) {
                    const cplx u = x[s + j];
                    const cplx v = mul(x[s + j + half], tw_[j * stride]);
                    x[s + j] = u + v;
                    x[s + j + half] = u - v;
                }
        }
    }

    /// Transform size.
    std::size_t size() const { return n_; }

private:
    std::size_t n_;
    std::vector<std::size_t> rev_;   // bit-reversed index of every position
    std::vector<cplx> tw_;
};

/// @brief FFT of arbitrary size n.
///
/// Powers of two go directly to Radix2. Other sizes use Bluestein's identity
/// jk = (j^2 + k^2 - (k-j)^2) / 2, which turns the DFT into a circular
/// convolution of length m >= 2n-1 (a power of two) computed with two radix-2 FFTs.
class Fft {
public:
    /// @param n transform size (any n >= 1)
    explicit Fft(std::size_t n) : n_(n), pow2_((n & (n - 1)) == 0), r2_(pow2_ ? n : next_pow2(2 * n - 1)) {
        if (pow2_) return;
        const std::size_t m = r2_.size();
        w_.resize(n);
        bhat_.assign(m, cplx(0.0));
        for (std::size_t k = 0; k < n; ++k) {
            // k^2 mod 2n keeps the argument small (exact in integers)
            const std::size_t k2 = (k * k) % (2 * n);
            w_[k] = std::polar(1.0, -M_PI * double(k2) / double(n));
        }
        bhat_[0] = std::conj(w_[0]);
        for (std::size_t k = 1; k < n; ++k) bhat_[k] = bhat_[m - k] = std::conj(w_[k]);
        r2_.forward(bhat_.data());
        const double scale = 1.0 / double(m);  // fold the inverse normalisation in
        for (auto &b : bhat_) b *= scale;
    }

    /// Transform size n.
    std::size_t size() const { return n_; }
    /// Number of elements the caller must provide as scratch (0 for powers of two).
    std::size_t scratch_size() const { return pow2_ ? 0 : r2_.size(); }

    /// @brief In-place forward DFT of length n.
    /// @param x data, n elements
    /// @param scratch at least scratch_size() elements (unused for powers of two)
    void forward(cplx *x, cplx *scratch) const {
        if (pow2_) {
            r2_.forward(x);
            return;
        }
        const std::size_t m = r2_.size();
        for (std::size_t k = 0; k < n_; ++k) scratch[k] = mul(x[k], w_[k]);
        for (std::size_t k = n_; k < m; ++k) scratch[k] = 0.0;
        r2_.forward(scratch);
        for (std::size_t k = 0; k < m; ++k) scratch[k] = std::conj(mul(scratch[k], bhat_[k]));
        r2_.forward(scratch);  // conj(FFT(conj(.))) = m * IFFT, scale already in bhat
        for (std::size_t k = 0; k < n_; ++k) x[k] = mul(std::conj(scratch[k]), w_[k]);
    }

    /// @brief In-place unnormalised inverse DFT: x[j] <- sum_k x[k] exp(+2 i pi j k / n).
    ///
    /// Uses IDFT(x) = conj(DFT(conj(x))), so no second plan is needed. The caller divides by n.
    void inverse(cplx *x, cplx *scratch) const {
        for (std::size_t k = 0; k < n_; ++k) x[k] = std::conj(x[k]);
        forward(x, scratch);
        for (std::size_t k = 0; k < n_; ++k) x[k] = std::conj(x[k]);
    }

private:
    /// Smallest power of two >= v.
    static std::size_t next_pow2(std::size_t v) {
        std::size_t p = 1;
        while (p < v) p <<= 1;
        return p;
    }

    std::size_t n_;
    bool pow2_;
    Radix2 r2_;
    std::vector<cplx> w_;      // Bluestein chirp w_k = exp(-i pi k^2 / n)
    std::vector<cplx> bhat_;   // FFT of the conjugate chirp, pre-divided by m
};

}  // namespace chirp::detail
