#include <torch/extension.h>

int addone(int x) { return x + 1; }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("addone", &addone);
}