/// @file core.cpp
/// @brief nanobind module `chirp_interf._core`.
///
/// NumPy arrays go in and out without copies:
///   - inputs are accepted as read-only, C-contiguous float64 / complex128 views
///     (the Python wrappers call np.ascontiguousarray beforehand);
///   - outputs are std::vector buffers handed over to NumPy through a capsule
///     that frees them when the array is garbage-collected.
/// The GIL is released during every computation, so the routines can run in
/// parallel with other Python threads.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/complex.h>

#include <initializer_list>
#include <vector>

#include "chirp/recovery.hpp"
#include "chirp/stft.hpp"

namespace nb = nanobind;
using namespace nb::literals;
using chirp::cplx;

/// 1-D read-only complex128 array on the CPU.
using CIn = nb::ndarray<const cplx, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
/// 1-D read-only float64 array on the CPU.
using RIn = nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

namespace {

/// Wrap a NumPy signal in a non-owning core view.
chirp::SignalView view(const CIn &f) { return {f.data(), f.shape(0)}; }
/// Wrap a NumPy window in a non-owning core view.
chirp::WindowView view(const RIn &g) { return {g.data(), g.shape(0)}; }

/// @brief Move a std::vector to the heap and expose it as a NumPy array of the given shape.
/// The capsule owns the vector and deletes it when NumPy releases the array.
auto to_numpy(std::vector<cplx> &&v, std::initializer_list<std::size_t> shape) {
    auto *heap = new std::vector<cplx>(std::move(v));
    nb::capsule owner(heap, [](void *p) noexcept { delete static_cast<std::vector<cplx> *>(p); });
    return nb::ndarray<nb::numpy, cplx>(heap->data(), shape, owner);
}

/// @brief Generic binding for rt1 / rt2 / rt3 (same signature): run without the GIL, return a NumPy array.
template <auto Fn>
auto recovery(const CIn &f, const RIn &g, long N, long R, long p, double threshold) {
    std::vector<cplx> out;
    {
        nb::gil_scoped_release release;
        out = Fn(view(f), view(g), N, R, p, threshold);
    }
    const std::size_t L = out.size();
    return to_numpy(std::move(out), {L});
}

}  // namespace

NB_MODULE(_core, m) {
    m.doc() = "C++/OpenMP core of chirp_interf: STFT on a time lattice and recovery techniques RT1-RT3.";

    m.def("has_openmp", [] {
#ifdef _OPENMP
        return true;
#else
        return false;
#endif
    }, "Return True if the extension was compiled with OpenMP.");

    m.def("stft_point",
          [](const CIn &f, const RIn &g, long t, double w) { return chirp::stft_point(view(f), view(g), t, w); },
          "f"_a, "g"_a, "t"_a, "w"_a,
          "V_g f(t, w) at integer time t and normalised frequency w (cycles/sample); "
          "samples outside the signal count as zero.");

    m.def("stft_frames",
          [](const CIn &f, const RIn &g, long N, long R, long p, long m_min, long m_max) {
              std::vector<cplx> V;
              {
                  nb::gil_scoped_release release;
                  V = chirp::stft_frames(view(f), view(g), N, R, p, m_min, m_max);
              }
              const std::size_t rows = m_max >= m_min ? std::size_t(m_max - m_min + 1) : 0;
              return to_numpy(std::move(V), {rows, std::size_t(N)});
          },
          "f"_a, "g"_a, "N"_a, "R"_a, "p"_a, "m_min"_a, "m_max"_a,
          "STFT on frames t_m = m R + p (m_min <= m <= m_max) and frequencies k/N; "
          "returns an array of shape (frames, N).");

    m.def("rt1", &recovery<chirp::rt1>, "f"_a, "g"_a, "N"_a, "R"_a, "p"_a, "threshold"_a = 0.0,
          "Recovery technique 1 (one frame per sample) from the STFT thresholded at `threshold`.");
    m.def("rt2", &recovery<chirp::rt2>, "f"_a, "g"_a, "N"_a, "R"_a, "p"_a, "threshold"_a = 0.0,
          "Recovery technique 2 (least-squares dual window) from the thresholded STFT.");
    m.def("rt3", &recovery<chirp::rt3>, "f"_a, "g"_a, "N"_a, "R"_a, "p"_a, "threshold"_a = 0.0,
          "Recovery technique 3 (overlap-add normalised by sum g) from the thresholded STFT.");
}
