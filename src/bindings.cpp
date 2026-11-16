#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "sobel.h"

namespace py = pybind11;

py::array_t<float> sobel_filter(py::array_t<float, py::array::c_style | py::array::forcecast> input) {
    py::buffer_info buf = input.request();
    if (buf.ndim != 2) {
        throw std::runtime_error("Input image must be a 2D float32 NumPy array.");
    }

    int height = static_cast<int>(buf.shape[0]);
    int width = static_cast<int>(buf.shape[1]);

    auto result = py::array_t<float>({height, width});
    py::buffer_info res_buf = result.request();

    compute_sobel(static_cast<const float*>(buf.ptr),
                  static_cast<float*>(res_buf.ptr),
                  width, height);

    return result;
}

PYBIND11_MODULE(sobel_ops, m) {
    m.doc() = "Shared-memory tiled CUDA Sobel filter with automatic CPU fallback";
    m.def("sobel_filter", &sobel_filter, "Compute Sobel edge magnitude on a 2D float32 array");
    m.def("is_cuda_enabled", &is_cuda_enabled, "Returns True if compiled with CUDA backend");
}
