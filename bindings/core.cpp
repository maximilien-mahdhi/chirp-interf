/// @file core.cpp
/// @brief pybind11 module `chirp_interf._core`.
///
/// NumPy arrays are read in place (the Python wrappers pass C-contiguous
/// complex128 / float64 arrays) and results are returned as NumPy arrays.
/// The GIL is released during every computation.

#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <vector>

#include "chirp/recovery.hpp"
#include "chirp/stft.hpp"

namespace py = pybind11;
using namespace py::literals;
using chirp::cplx;

/// 1-D complex128 array (cast and made contiguous if needed).
using CIn = py::array_t<cplx, py::array::c_style | py::array::forcecast>;
/// 1-D float64 array (cast and made contiguous if needed).
using RIn = py::array_t<double, py::array::c_style | py::array::forcecast>;

namespace {

/// Non-owning view of a NumPy signal.
chirp::SignalView view(const CIn &f) { return {f.data(), static_cast<std::size_t>(f.size())}; }

/// Non-owning view of a NumPy window.
chirp::WindowView view(const RIn &g) { return {g.data(), static_cast<std::size_t>(g.size())}; }

/// Copy a std::vector into a new NumPy array of the given shape.
py::array_t<cplx> to_numpy(const std::vector<cplx> &v, std::vector<py::ssize_t> shape) {
    py::array_t<cplx> a(shape);
    std::copy(v.begin(), v.end(), a.mutable_data());
    return a;
}

/// Generic binding for rt1 / rt2 / rt3 (same signature).
template <std::vector<cplx> (*Fn)(chirp::SignalView, chirp::WindowView, long, long, long, double)>
py::array_t<cplx> recovery(const CIn &f, const RIn &g, long N, long R, long p, double threshold) {
    std::vector<cplx> out;
    {
        py::gil_scoped_release release;
        out = Fn(view(f), view(g), N, R, p, threshold);
    }
    return to_numpy(out, {static_cast<py::ssize_t>(out.size())});
}

}  // namespace

PYBIND11_MODULE(_core, m) {
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
                  py::gil_scoped_release release;
                  V = chirp::stft_frames(view(f), view(g), N, R, p, m_min, m_max);
              }
              const py::ssize_t rows = m_max >= m_min ? m_max - m_min + 1 : 0;
              return to_numpy(V, {rows, static_cast<py::ssize_t>(N)});
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
