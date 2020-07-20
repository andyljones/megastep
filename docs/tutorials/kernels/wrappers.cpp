#include <torch/extension.h>

using TT = at::Tensor;

TT addone(TT x) { return x + 1; }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("addone", &addone);
}