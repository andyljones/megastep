#include <ATen/ATen.h>
#include <variant>
#include <exception>
#include <iostream>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace pybind11::literals;

using TT = at::Tensor;

#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Define our own copy of RestrictPtrTraits here, as the at::RestrictPtrTraits is 
// only included during NVCC compilation, not plain C++. This would mess things up 
// since this file is included on both the NVCC and Clang sides. 
template <typename T>
struct RestrictPtrTraits {
  typedef T* __restrict__ PtrType;
};

template<typename T>
at::ScalarType dtype() { return at::typeMetaToScalarType(caffe2::TypeMeta::Make<T>()); }

template <typename T, size_t D>
struct TensorProxy {

    using PTA = at::PackedTensorAccessor32<T, D, RestrictPtrTraits>;
    TT t; 

    TensorProxy(const at::Tensor t) : t(t) {
        CHECK_INPUT(t);
        AT_ASSERT(t.scalar_type() == dtype<T>());
        AT_ASSERT(t.ndimension() == D);
    }

    static TensorProxy<T, D> empty(at::IntArrayRef size) { return TensorProxy(at::empty(size, at::device(at::kCUDA).dtype(dtype<T>()))); }
    static TensorProxy<T, D> zeros(at::IntArrayRef size) { return TensorProxy(at::zeros(size, at::device(at::kCUDA).dtype(dtype<T>()))); }
    static TensorProxy<T, D> ones(at::IntArrayRef size) { return TensorProxy(at::ones(size, at::device(at::kCUDA).dtype(dtype<T>()))); }

    PTA pta() const { return t.packed_accessor32<T, D, RestrictPtrTraits>(); }

    size_t size(const size_t i) const { return t.size(i); }
};

template <typename T, size_t D>
struct RaggedPackedTensorAccessor {
    using IPTA = TensorProxy<int, 1>::PTA;
    using PTA = at::PackedTensorAccessor32<T, D, RestrictPtrTraits>;
    using TA = at::TensorAccessor<T, D, RestrictPtrTraits, int32_t>;

    PTA vals;
    const IPTA widths;
    const IPTA starts;
    const IPTA inverse;
    int32_t _sizes[D];
    int32_t _strides[D];

    RaggedPackedTensorAccessor(
        TT vals, TT widths, TT starts, TT inverse) :
        vals(vals.packed_accessor32<T, D, RestrictPtrTraits>()), 
        widths(widths.packed_accessor32<int, 1, RestrictPtrTraits>()), 
        starts(starts.packed_accessor32<int, 1, RestrictPtrTraits>()), 
        inverse(inverse.packed_accessor32<int, 1, RestrictPtrTraits>()) {

        for (auto d=0; d<D; ++d) {
            _sizes[d] = vals.size(d);
            _strides[d] = vals.stride(d);
        }
        // _sizes[0] is going to be wrong, but that's not used by the accessor mechanism. 
        // Alternative is to construct PTAs dynamically on the device, which would be Not Fast. 
        // Just to make sure we notice immediately if it's ever used, let's set it to an illegal value
        _sizes[0] = -1;
    }

    C10_HOST_DEVICE TA operator[](const int n) const {
        return TA(vals.data() + starts[n]*_strides[0], _sizes, _strides);
    }
    
    C10_HOST_DEVICE int64_t size(const int d) const { 
        return (d == 0) ? widths.size(0) : vals.size(d-1);
    }

};

#if defined(__CUDACC__)
TT inverses(const TT& widths);
#else
TT inverses(const TT& widths) {
    at::AutoNonVariableTypeMode nonvar{true};
    const auto starts = widths.cumsum(0) - widths.to(at::kLong);
    const auto flags = at::ones(starts.size(0), at::dtype(at::kInt).device(widths.device()));
    auto indices = at::zeros(widths.sum(0).item<int64_t>(), at::dtype(at::kInt).device(widths.device()));
    auto inverse = indices.scatter(0, starts, flags).cumsum(0).to(at::kInt)-1;
    return inverse;
}  
#endif


template <typename T, size_t D>
struct Ragged {
    const TT vals;
    const TT widths;
    const TT starts;
    const TT ends;
    const TT inverse;

    using PTA = RaggedPackedTensorAccessor<T, D>;

    Ragged(TT vals, TT widths) : 
        vals(vals), widths(widths), 
        starts(widths.cumsum(0).toType(at::kInt) - widths),
        ends(widths.cumsum(0).toType(at::kInt)),
        inverse(inverses(widths)) { 
        
        CHECK_CONTIGUOUS(vals);
        CHECK_CONTIGUOUS(widths);

        AT_ASSERT(widths.size(0) == starts.size(0));
        AT_ASSERT(widths.scalar_type() == dtype<int>());
        AT_ASSERT(widths.ndimension() == 1);
        AT_ASSERT(widths.sum(0).item<int64_t>() == vals.size(0));
        AT_ASSERT(vals.size(0) == inverse.size(0));
        AT_ASSERT(vals.scalar_type() == dtype<T>());
        AT_ASSERT(vals.ndimension() == D);
    }

    PTA pta() const { return PTA(vals, widths, starts, inverse); }

    TT operator[](const int n) const {
        return vals.slice(0, starts[n].item<int64_t>(), ends[n].item<int64_t>());
    }

    Ragged<T, D> operator[](const py::slice slice) const {
        py::size_t start, stop, step, slicelength;
        if (!slice.compute(static_cast<py::size_t>(widths.size(0)), &start, &stop, &step, &slicelength)) {
            throw py::error_already_set();
        }
        return Ragged<T, D>(
            vals.slice(0, starts[start].item<int64_t>(), ends[stop-1].item<int64_t>()),
            widths.slice(0, start, stop));
    }

    size_t size(const size_t i) const { return vals.size(i); }

    Ragged<T, D> clone() const { return Ragged<T, D>(vals.clone(), widths.clone()); }

    py::object numpyify() const {
        const auto Ragged = py::module::import("megastep.ragged").attr("RaggedNumpy");
        const auto numpyify = py::module::import("rebar.arrdict").attr("numpyify");
        return Ragged(numpyify(vals), numpyify(widths));
    }
};

using Angles = TensorProxy<float, 2>;
using Positions = TensorProxy<float, 3>;
using AngMomenta = TensorProxy<float, 2>;
using Momenta = TensorProxy<float, 3>;

struct Agents {
    Angles angles;
    Positions positions; 
    AngMomenta angmomenta;
    Momenta momenta; 
};

using Lights = Ragged<float, 2>;
using Lines = Ragged<float, 3>;
using Textures = Ragged<float, 2>;
using Baked = Ragged<float, 1>;
using Frame = TensorProxy<float, 3>;

struct Scenery {
    const int n_agents;
    const Lights lights;
    const Lines lines;
    const Textures textures;
    const Frame frame;
    const Baked baked;

    // Weird initialization of `baked` here is to avoid having to create a `AutoNonVariableTypeMode` 
    // guard, because I still don't understand the Variable vs Tensor thing. 
    // Goal is to create a Tensor of 1s like textures.vals[:, 0]
    Scenery(int n_agents, Lights lights, Lines lines, Textures textures, TT frame) :
        n_agents(n_agents), lights(lights), lines(lines), textures(textures), frame(frame),
        baked(at::ones_like(textures.vals.select(1, 0)), textures.widths) {
    }

    py::object operator[](const size_t e) {
        const auto dotdict = py::module::import("rebar.dotdict").attr("dotdict");
        const auto se = lines.starts[e].item<int64_t>();
        const auto ee = lines.ends[e].item<int64_t>();
        return dotdict(
            "n_agents"_a=n_agents,
            "lights"_a=lights[e],
            "lines"_a=lines[e],
            "textures"_a=textures[py::slice(se, ee, 1)],
            "frame"_a=frame.t,
            "baked"_a=baked[py::slice(se, ee, 1)]);
    }
    
};

struct Render {
    const TT indices;
    const TT locations;
    const TT dots;
    const TT distances;
    const TT screen;
};

using Progress = TensorProxy<float, 2>; 

void initialize(float, int, float, float);
void bake(Scenery& scenery);
void physics(const Scenery& scenery, Agents& agents, Progress progress);
Render render(const Scenery& scenery, const Agents& agents);
