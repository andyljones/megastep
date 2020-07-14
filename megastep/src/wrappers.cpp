#include <torch/extension.h>
#include <torch/csrc/autograd/variable.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include "common.h"

namespace py = pybind11;
using namespace pybind11::literals;
using namespace std::string_literals;

TT variable(TT t) { return torch::autograd::make_variable(t); }

/// Proxy for converting the Tensor of `progress` into a TensorProxy
void _physics(const Scene& scene, Agents& agents, TT progress) { return physics(scene, agents, progress); }

template<typename T>
void ragged(py::module &m, std::string name) {
    py::class_<T>(m, name.c_str(), py::module_local())
        .def(py::init<TT, TT>(), 
            "vals"_a, "widths"_a)
        .def("__getitem__", [](T self, int n) { return self[n]; })
        .def("__getitem__", [](T self, py::slice slice) { return self[slice]; })
        .def("clone", &T::clone)
        .def("numpyify", &T::numpyify)
        .def_readonly("vals", &T::vals)
        .def_readonly("widths", &T::widths)
        .def_readonly("starts", &T::starts)
        .def_readonly("ends", &T::ends)
        .def_property_readonly("inverse", [](T r) { return variable(r.inverse); });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = R"pbdoc(
        This module contains all the rendering and physics CUDA kernels, and intended to operate on the state tensors held
        by :class:`megastep.core.Core`. 

        **Internals**

        This module is dynamically compiled upon import of :mod:`megastep`.

        The best explanation of how the bridge between CUDA and Python works is the `PyTorch
        C++ extension tutorial <https://pytorch.org/tutorials/advanced/cpp_extension.html>`_ .
        
        In short though, this is a `PyBind <https://pybind11.readthedocs.io/>`_ module. You can find the PyBind
        wrappers in ``wrappers.cpp``, and the actual code they call into in ``common.h`` and ``kernels.cu``.
        
        I have very limited experience with distributing binaries, so while I've _tried_ to reference the library
        paths in a platform-independent way, there is a good chance they'll turn out to be dependent after all.
        Submit an issue and explain a better way to me!
        
        The libraries listed are - I believe - the minimal possible to allow megastep's compilation. The default
        library set for PyTorch extensions is much larger and slower to compile.
    )pbdoc";

    m.def("initialize", &initialize, "agent_radius"_a, "res"_a, "fov"_a, "fps"_a, R"pbdoc(
        Initializes the CUDA kernels by setting some global constants. The constants are then used by 
        :func:`bake`, :func:`physics` and :func:`render`.

        Really, the existance of these constants is an indicator the whole CUDA side of things should be wrapped up
        in a class. But that'd make things a bit messier, and it's a rare use-case that'll have them set to different
        values in the same process.
    )pbdoc");
    m.def("bake", &bake, "scene"_a, R"pbdoc(
        Pre-computes the lighting for the static geometry, updating the :attr:`Scene.baked` tensor.

        For more details on how this works, see the :ref:`rendering <Rendering>` section.

        :param scene: The scene to compute the lighting for
        :type scene: :class:`Scene`
    )pbdoc", py::call_guard<py::gil_scoped_release>());
    m.def("physics", &_physics, "scene"_a, "agents"_a, "progress"_a, R"pbdoc(
        Advances the physics simulation, updating the :attr:`Agents`'s movement tensors based on their momenta 
        and possible collisions. It also updates the ``progress`` tensor with how far the agents moved before
        colliding with something.

        For more details on how this works, see the :ref:`physics <Physics>` section.
        
        :param scene: The scene to reference when updating the agents
        :type scene: :class:`Scene`
        :param agents: The agents to update the movement of
        :type agents: :class:`Agents`
        :param progress: A (n_env, n_agent) tensor that will be filled with the progress made by the agents. 'Progress'
            is what fraction of their intended movement they managed to complete before colliding with something. A
            value less than 1 means they did indeed hit something.
    )pbdoc", py::call_guard<py::gil_scoped_release>());
    m.def("render", &render, "scene"_a, "agents"_a, R"pbdoc(
        Returns a rendering of the scene onto the agents' cameras.

        For more details on how this works, see the :ref:`rendering <Rendering>` section.

        :param scene: The scene to reference when updating the agents
        :type scene: :class:`Scene`
        :param agents: The agents to update the movement of
        :type agents: :class:`Agents`
        :rtype: :class:`Render` 
    )pbdoc", py::call_guard<py::gil_scoped_release>());

    py::options options;
    options.disable_function_signatures();

    // PyBind won't bind two aliases of the same type, so rather than bind one of Lights or Textures, 
    // instead a generic alias gets bound for each dimension.
    ragged<Ragged<float, 1>>(m, "Ragged1D");
    ragged<Ragged<float, 2>>(m, "Ragged2D");
    ragged<Ragged<float, 3>>(m, "Ragged3D");

    py::class_<Agents>(m, "Agents", py::module_local())
        .def(py::init<TT, TT, TT, TT>(),
            "angles"_a, "positions"_a, "angmomenta"_a, "momenta"_a, R"pbdoc( 
                Holds the state of the agents. Typically accessed through :attr:`megastep.core.Core.agents`.)pbdoc")
        .def_property_readonly("angles", [](Agents a) { return a.angles.t; }, R"pbdoc(
            An (n_env, n_agent)-tensor of agents' angles relative to the positive x axis, given in degrees.)pbdoc")
        .def_property_readonly("positions", [](Agents a) { return a.positions.t; }, R"pbdoc(
            An (n_env, n_agent, 2)-tensor of agents' positions, in meters.)pbdoc")
        .def_property_readonly("angmomenta", [](Agents a) { return a.angmomenta.t; }, R"pbdoc(
            An (n_env, n_agent)-tensor of agents' angular velocity, in degrees per second.)pbdoc")
        .def_property_readonly("momenta", [](Agents a) { return a.momenta.t; }, R"pbdoc( 
            An (n_env, n_agent, 2)-tensor of agents' velocity, in meters per second.)pbdoc");

    py::class_<Scene>(m, "Scene", py::module_local()) 
        .def(py::init<int, Lights, Lines, Textures, TT>(),
            "n_agents", "lights"_a, "lines"_a, "textures"_a, "frame"_a, R"pbdoc(
                Holds the state of the scene. Typically accessed through :attr:`megastep.core.Core.scene`.)pbdoc")
        .def_property_readonly("frame", [](Scene s) { return s.frame.t; }, R"pbdoc(
            An (n_frame_line, 2, 2)-tensor giving the frame - the set of lines - that make up the agent. This will be 
            shifted and rotated according to the :class:`Agents` angles and positions, then rendered into the scene.)pbdoc")
        .def_readonly("n_agents", &Scene::n_agents, R"pbdoc(
            The number of agents in each environment)pbdoc")
        .def_readonly("lights", &Scene::lights, R"pbdoc(
            An (n_lights, 3)-tensor giving the locations of the lights in the first two columns, and their intensities 
            (typically a value between 0 and 1) in the third.)pbdoc")
        .def_readonly("lines", &Scene::lines, R"pbdoc(
            An (n_lines, 2, 2)-:class:`megastep.ragged.Ragged` tensor giving the lines in each scene.)pbdoc")
        .def_readonly("textures", &Scene::textures, R"pbdoc(
            An (n_texels, 3)-:class:`megastep.ragged.Ragged` tensor giving the texels in each line.)pbdoc")
        .def_readonly("baked", &Scene::baked, R"pbdoc(
            An (n_texels,)-:class:`megastep.ragged.Ragged` tensor giving the :func:`bake`-d illumination of each texel.)pbdoc");

    py::class_<Render>(m, "Render", py::module_local(), R"pbdoc(
            The result of a :func:`render` call, showing the scene from the agents' points of view.

            Rendering is done by casting 'rays' from the camera, through each pixel and out into the world. When a ray
            intersects a line from :attr:`Scene.lines`, that's called a 'hit'. )pbdoc")
        .def_property_readonly("screen", [](Render r) { return variable(r.screen); }, R"pbdoc(
            A (n_envs, n_agents, res, 3)-tensor giving the views of each agent. Colours are RGB with values between 0 and 1. Infinity is coloured black.)pbdoc")
        .def_property_readonly("indices", [](Render r) { return variable(r.indices); }, R"pbdoc(
            A (n_envs, n_agents, res)-tensor giving the index into :attr:`Scene.lines` of the hit. Rays which don't hit anything get a -1.)pbdoc")
        .def_property_readonly("locations", [](Render r) { return variable(r.locations); }, R"pbdoc(
            A (n_envs, n_agents, res)-tensor giving the location along each line that the hit occurs. A zero means it
            happened at the first endpoint; a one means it happened at the second.)pbdoc")
        .def_property_readonly("dots", [](Render r) { return variable(r.dots); }, R"pbdoc(
            A (n_envs, n_agents, res)-tensor giving the dot product between the ray and the line it hit.)pbdoc")
        .def_property_readonly("distances", [](Render r) { return variable(r.distances); }, R"pbdoc(
            A (n_envs, n_agents, res)-tensor giving the distance from the camera to the hit in meters.)pbdoc");
}