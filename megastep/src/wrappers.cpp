#include <torch/extension.h>
#include <torch/csrc/autograd/variable.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "common.h"
namespace py = pybind11;
using namespace pybind11::literals;
using namespace std::string_literals;

TT variable(TT t) { return torch::autograd::make_variable(t); }

/// Proxy for converting the Tensor of `progress` into a TensorProxy
void _physics(const Scene& scene, Agents& agents, TT progress) {
    return physics(scene, agents, progress); 
}

template<typename T>
void ragged(py::module &m, std::string name) {
    py::class_<T>(m, name.c_str(), py::module_local())
        .def(py::init<TT, TT>(), 
            "vals"_a, "widths"_a)
        .def_readonly("vals", &T::vals)
        .def_readonly("widths", &T::widths)
        .def_readonly("starts", &T::starts)
        .def_property_readonly("inverse", [](T r) { return variable(r.inverse); });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = R"pbdoc(
        The C++ side of megastep,
        
        This module contains all the rendering and physics CUDA kernels, and they operate on the state tensors held
        by :class:`megastep.core.Core`. It's dynamically compiled upon import of :mod:`megastep.core`.
        
        For ease of use, all the kernels are rebound as attributes on :mod:`megastep.core`. The only reason the
        ``.cuda`` module is exposed is that I expect many users of this library to be curious as to how it's put
        together.

        **Internals**

        The best explanation of how the CUDA side of things is nailed onto the Python side of things is the `PyTorch
        C++ extension tutorial <https://pytorch.org/tutorials/advanced/cpp_extension.html>`_ .
        
        In short though, this is a `PyBind <https://pybind11.readthedocs.io/>`_ module. You can find the PyBind
        wrappers in ``wrappers.cpp``, and the actual code they call into in ``common.h`` and ``kernels.cu``.
        
        I have very limited experience with distributing binaries, so while I've _tried_ to reference the library
        paths in a platform-independent way, there is a good chance they'll turn out to be dependent after all.
        Sorry. Submit an issue and explain a better way to me!
        
        The libraries listed are - I believe - the minimal possible to allow megastep's compilation. The default
        library set for PyTorch extensions is much larger and slower to compile.
    )pbdoc";

    ragged<Textures>(m, "Textures");
    ragged<Lines>(m, "Lines");
    ragged<Baked>(m, "Baked");
    // ragged<Lights>(m, "Lights"); // Forbidden as replicates Textures

    //TODO: Swap out this Agents/Scene stuff for direct access to the arrays.
    // Will have to replicate the Ragged logic on the Python side, but it's worth it to 
    // avoid all this indirection.
    py::class_<Agents>(m, "Agents", py::module_local())
        .def(py::init<TT, TT, TT, TT>(),
            "angles"_a, "positions"_a, "angmomenta"_a, "momenta"_a)
        .def_property_readonly("angles", [](Agents a) { return a.angles.t; })
        .def_property_readonly("positions", [](Agents a) { return a.positions.t; })
        .def_property_readonly("angmomenta", [](Agents a) { return a.angmomenta.t; })
        .def_property_readonly("momenta", [](Agents a) { return a.momenta.t; });

    py::class_<Scene>(m, "Scene", py::module_local(), R"pbdoc(
        Datastructure describing the scenes.
    )pbdoc") 
        .def(py::init<Lights, Lines, Textures, TT>(),
            "lights"_a, "lines"_a, "textures"_a, "frame"_a)
        .def_property_readonly("frame", [](Scene s) { return s.frame.t; })
        .def_readonly("lights", &Scene::lights)
        .def_readonly("lines", &Scene::lines)
        .def_readonly("textures", &Scene::textures)
        .def_readonly("baked", &Scene::baked);

    py::class_<Render>(m, "Render", py::module_local())
        .def_property_readonly("screen", [](Render r) { return variable(r.screen); })
        .def_property_readonly("indices", [](Render r) { return variable(r.indices); })
        .def_property_readonly("locations", [](Render r) { return variable(r.locations); })
        .def_property_readonly("dots", [](Render r) { return variable(r.dots); })
        .def_property_readonly("distances", [](Render r) { return variable(r.distances); });

    m.def("initialize", &initialize, R"pbdoc(
        Initializes the state.
    )pbdoc");
    m.def("bake", &bake, py::call_guard<py::gil_scoped_release>());
    m.def("physics", &_physics, py::call_guard<py::gil_scoped_release>());
    m.def("render", &render, py::call_guard<py::gil_scoped_release>());
}