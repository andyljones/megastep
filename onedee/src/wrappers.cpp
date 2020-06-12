#include <torch/extension.h>
#include <torch/csrc/autograd/variable.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "common.h"
namespace py = pybind11;
using namespace pybind11::literals;
using namespace std::string_literals;

TT variable(TT t) { return torch::autograd::make_variable(t); }

// TODO: Is this still needed?
void _physics(const Movement movement, const Scene& scene, Drones& drones) { return physics(movement, scene, drones); }

template<typename T>
void ragged(py::module &m, std::string name) {
    py::class_<T>(m, name.c_str(), py::module_local())
        .def_readonly("vals", &T::vals)
        .def_readonly("widths", &T::widths)
        .def_readonly("starts", &T::starts)
        .def_property_readonly("inverse", [](T r) { return variable(r.inverse); });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    ragged<Textures>(m, "Textures");
    ragged<Lines>(m, "Lines");
    ragged<Baked>(m, "Baked");
    // ragged<Lights>(m, "Lights"); // Unneeded as replicates an existing type

    py::class_<Respawns>(m, "Respawns", py::module_local())
        .def(py::init<TT, TT, TT, TT, TT>(), 
            "centers"_a, "radii"_a, "lowers"_a, "uppers"_a, "widths"_a);

    //TODO: Swap out this DroneData/SceneData stuff for direct access to the arrays.
    // Will have to replicate the Ragged logic on the Python side, but it's worth it to 
    // avoid all this indirection.
    py::class_<Drones>(m, "Drones", py::module_local())
        .def(py::init<TT, TT, TT, TT>(),
            "angles"_a, "positions"_a, "angmomenta"_a, "momenta"_a)
        .def_property_readonly("angles", [](Drones d) { return d.angles.t; })
        .def_property_readonly("positions", [](Drones d) { return d.positions.t; })
        .def_property_readonly("angmomenta", [](Drones d) { return d.angmomenta.t; })
        .def_property_readonly("momenta", [](Drones d) { return d.momenta.t; });

    py::class_<Scene>(m, "Scene", py::module_local()) 
        .def(py::init<TT, TT, TT, TT, TT, TT, TT>(),
            "lights"_a, "lightwidths"_a, 
            "lines"_a, "linewidths"_a,
            "textures"_a, "texwidths"_a,
            "frame"_a)
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

    py::class_<Movement>(m, "Movement", py::module_local())
        .def(py::init<TT, TT, TT>(), "mesial"_a, "lateral"_a, "yaw"_a);

    m.def("initialize", &initialize);
    m.def("bake", &bake, py::call_guard<py::gil_scoped_release>());
    m.def("respawn", &respawn, py::call_guard<py::gil_scoped_release>());
    m.def("physics", &_physics, py::call_guard<py::gil_scoped_release>());
    m.def("render", &render, py::call_guard<py::gil_scoped_release>());
    //TODO: This should work. Come back when you're more comfortable with type inference
    // m.def("render", [](auto ...args) { return torch::autograd::make_variable(render(args...)); });
}