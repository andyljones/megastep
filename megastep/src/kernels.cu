#include <math_constants.h>
#include <ATen/ATen.h>
#include "common.h"
#include <ATen/cuda/CUDAContext.h>
#include <tuple>

// COMMON

const float AMBIENT = .1;
const uint BLOCK = 128;

int RES;
float FPS;
__constant__ float FPS_;
__constant__ float AGENT_RADIUS;
__constant__ float HALF_SCREEN_WIDTH;

__host__ void initialize(float agent_radius, int res, float fov, float fps) {
    RES = res;
    cudaMemcpyToSymbol(AGENT_RADIUS, &agent_radius, sizeof(float));

    const auto half_screen = tanf(CUDART_PI_F/180.f*fov/2.);
    cudaMemcpyToSymbol(HALF_SCREEN_WIDTH, &half_screen, sizeof(float));

    FPS = fps;
    cudaMemcpyToSymbol(FPS_, &fps, sizeof(float));
}


at::cuda::CUDAStream stream() { 
    return at::cuda::getCurrentCUDAStream();
}

// SIMULATOR - COLLISIONS

struct Point {
    float x;
    float y;

    __device__ Point(float x, float y) : x(x), y(y) { }
    __device__ Point(at::TensorAccessor<float, 1, RestrictPtrTraits, int> t) : x(t[0]), y(t[1]) { }

    __device__ Point operator/(float v) const { return Point(x/v, y/v); }
    __device__ Point operator*(float v) const { return Point(x*v, y*v); }
    __device__ Point operator+(const Point q) const { return Point(x + q.x, y + q.y); }
    __device__ Point operator-(const Point q) const { return Point(x - q.x, y - q.y); }

    __device__ float len2() const { return x*x + y*y; }
    __device__ float len() const { return sqrtf(len2()); }
};

struct Line {
    Point a;
    Point b;

    __device__ Line(at::TensorAccessor<float, 2, RestrictPtrTraits, int> t) : a(t[0]), b(t[1]) { }
};

__device__ inline float cross(Point V, Point W) {
    return V.x*W.y - V.y*W.x;
}

__device__ inline float dot(Point V, Point W) {
    return V.x*W.x + V.y*W.y;
}

struct Intersection { float s; float t; };
__device__ Intersection intersect(Point P, Point U, Point Q, Point V) {
    /* Finds the intersection of two infinite lines described as P+sU, Q+tV
    
    Returns:
        s: fraction from P along U to intersection
        t: fraction from Q along V to intersection
    */

    const auto UxV = cross(U, V);
    const bool distant = fabsf(UxV) < 1.e-3f;
    float s, t;
    if (distant) {
        s = CUDART_INF_F;
        t = CUDART_INF_F;
    } else {
        const auto PQ = Q - P;
        s = cross(PQ, V)/UxV;
        t = cross(PQ, U)/UxV;
    }
    return {s, t};
}
__device__ Intersection intersect(Point P, Point U, Line L) { return intersect(P, U, L.a, L.b - L.a); }

struct Projection { float s; float d; };
__device__ Projection project(Point P, Point U, Point Q) {
    /* Projects Q onto the infinite line P+sU.
    
    Returns:
        s: fraction from P along U to the projection of Q onto (P, P+U)
        d: distance from Q to the projection of Q onto (P, P+U)
    */
    const auto u = U.len() + 1e-6f;

    const auto PQ = Q - P;
    const auto s = dot(PQ, U)/(u*u);
    const auto d = fabsf(cross(PQ, U))/u;

    return {s, d};
}
__device__ Projection project(Line L, Point Q) { return project(L.a, L.b - L.a, Q); }

__device__ float sensibilize(float p) {
    // Collide a bit earlier than is exactly right, so nothing gets stuck in the next iteration
    const auto margin = .99f;
    // Sometimes the `side` value can have numerical issues, so here we clamp it to something sensible
    if (isnan(p)) {
        return 0.f;
    } else {
        return fmaxf(fminf(p*margin, 1.f), 0.f);
    }
}
__device__ float collision(Point p0, Point v0, Point p1, Point v1) {
    //Follows http://ericleong.me/research/circle-circle/#dynamic-circle-circle-collision

    // Make the agent a bit bigger so that the near vision plane doesn't go through walls
    const auto r = 1.001f*2.f*AGENT_RADIUS;
    auto x = 1.f;

    const auto a = project(p0, v0 - v1, p1);
    if ((0 < a.s) & (a.d < r)) {
        const auto backoff = sqrtf(r*r - a.d*a.d)/(v0 - v1).len();
        x = fminf(x, sensibilize(a.s - backoff));
    }

    return x;
}

__device__ float collision(Point p, Point v, Line l) {
    // Follows http://ericleong.me/research/circle-line/#moving-circle-and-static-line-segment

    // Make the agent a bit bigger so that the near vision plane doesn't go through walls
    const auto r = 1.001f*AGENT_RADIUS;
    auto x = 1.f;

    // Test for passing through `l`
    const auto mid = intersect(p, v, l);
    if ((0 < mid.s) & (mid.s < 1) & (0 < mid.t) & (mid.t < 1)) {
        x = fminf(x, sensibilize((1 - r/project(l, p).d)*mid.s));
    } 

    // Test for passing within r of `l.a`
    const auto a = project(p, v, l.a);
    if ((0 < a.s) & (a.d < r)) {
        const auto backoff = sqrtf(r*r - a.d*a.d)/v.len();
        x = fminf(x, sensibilize(a.s - backoff));
    }

    // Test for passing within r of `l.b`
    const auto b = project(p, v, l.b);
    if ((0 < b.s) & (b.d < r)) {
        const auto backoff = sqrtf(r*r - b.d*b.d)/v.len();
        x = fminf(x, sensibilize(b.s - backoff));
    }

    // Test for passing within r of the middle of `l`
    const auto side = project(l, p + v);
    if ((0 < side.s) & (side.s < 1) & (side.d < r)) {
        const auto dp = project(l, p).d;
        const auto dq = side.d;
        x = fminf(x, sensibilize((dp - r)/(dp - dq)));
    }

    return x;
}

__host__ TT normalize_degrees(TT a) {
    return (((a % 360.f) + 180.f) % 360.f) - 180.f;
}

using Progress = TensorProxy<float, 2>; 

__global__ void collision_kernel(
                int DF, Positions::PTA positions, Velocity::PTA velocity, 
                Lines::PTA lines, Progress::PTA progress) {
    const auto N = positions.size(0);
    const auto D = positions.size(1);

    const int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n < N) {
        const auto L = lines.widths[n];
        for (int d0=0; d0 < D; d0++) {
            const Point p0(positions[n][d0]);
            const Point m0(velocity[n][d0]);

            float x = 1.f;
            for (int d1=0; d1 < D; d1++) {
                if (d0 != d1) {
                    const Point p1(positions[n][d1]);
                    const Point m1(velocity[n][d1]);

                    x = fminf(x, collision(p0, m0/FPS_, p1, m1/FPS_));
                }
            }

            // Check whether it's collided with any walls
            for (int l=DF; l < L; l++) {
                x = fminf(x, collision(p0, m0/FPS_, lines[n][l]));
            }

            progress[n][d0] = x;
        }
    }
}

__host__ Physics physics(const Scenery& scenery, const Agents& agents) {
    const uint N = agents.angles.size(0);
    const uint A = scenery.n_agents;
    const uint F = scenery.model.size(0);

    auto progress(Progress::empty({N, A}));
    const uint collision_blocks = (N + BLOCK - 1)/BLOCK;
    collision_kernel<<<collision_blocks, {BLOCK,}, 0, stream()>>>(
        A*F, agents.positions.pta(), agents.velocity.pta(), scenery.lines.pta(), progress.pta());

    //TODO: Collisions should only kill the normal component of momentum
    c10::InferenceMode nonvar{true};
    agents.positions.t.set_(agents.positions.t + progress.t.unsqueeze(-1)*agents.velocity.t/FPS);
    agents.velocity.t.masked_fill_(progress.t.unsqueeze(-1) < 1, 0.f);
    agents.angles.t.set_(normalize_degrees(agents.angles.t + progress.t*agents.angvelocity.t/FPS));
    agents.angvelocity.t.masked_fill_(progress.t < 1, 0.f);

    return {progress.t};
}

// RENDERING - BAKING

__device__ float ray_y(float r, float R) {
    return (R - 2*r - 1)*HALF_SCREEN_WIDTH/R;
}

__device__ float light_intensity(Lines::PTA lines, Lights::PTA lights, Point C, int n, int af) {

    const float LUMINANCE = 2.f;
    float intensity = AMBIENT;

    const auto num_i = lights.widths[n];
    const auto num_l = lines.widths[n];
    for (int i=0; i < num_i; i++) {
        const Point I(lights[n][i]);
        const auto Ii = lights[n][i][2];
        bool unobstructed = true;

        // Ignore the dynamic lines at the front of the array
        for (int l1=af; l1 < num_l; l1++) {
            const Line L(lines[n][l1]);
            const auto p = intersect(I, C - I, L);

            // Test the length to .999 rather than 1 so that floating point errors don't end up
            // randomly darkening some texels.
            bool obstructed = (p.t > 0.f) & (p.t < 1.f) & (p.s > 0.f) & (p.s < .999f);
            unobstructed = unobstructed & !obstructed;
        }

        const auto d2 = (I - C).len2();
        if (unobstructed) {
            intensity += LUMINANCE*Ii/fmaxf(d2, 1.f);
        } 
    }
    
    return fminf(intensity, 1.f);
}

__global__ void baking_kernel(
    Lines::PTA lines, Lights::PTA lights, Textures::PTA textures, Baked::PTA baked, int af) {

    const auto t = blockDim.x*blockIdx.x + threadIdx.x;
    if (t < textures.vals.size(0)) {
        const auto l0 = textures.inverse[t];
        const auto n = lines.inverse[l0];

        const auto loc = (t - textures.starts[l0] + .5f)/textures.widths[l0];
        const auto C = Point(lines.vals[l0][0])*(1.f-loc) + Point(lines.vals[l0][1])*loc;

        const auto i = light_intensity(lines, lights, C, n, af);
        baked.vals[t] = i;
    }
}

__host__ void bake(Scenery& scenery) {
    const uint T = scenery.textures.vals.size(0);
    const uint F = scenery.model.size(0);

    const auto blocks = (T + BLOCK - 1)/BLOCK;
    baking_kernel<<<blocks, BLOCK, 0, stream()>>>(
        scenery.lines.pta(), scenery.lights.pta(), scenery.textures.pta(), scenery.baked.pta(), scenery.n_agents*F);
}

// RENDERING - KERNELS

__global__ void draw_kernel(Angles::PTA angles, Positions::PTA positions, Model::PTA model, Lines::PTA lines) {
    const auto n = blockIdx.x;

    const auto e = threadIdx.x;
    const auto m = threadIdx.y;
    const auto a = threadIdx.z;

    const auto ang = angles[n][a]/180.f;
    const auto c = cospif(ang);
    const auto s = sinpif(ang);

    const auto px = positions[n][a][0];
    const auto py = positions[n][a][1];

    // TODO: Stick these in constant memory
    const auto M = model.size(0);
    const auto mx = model[m][e][0];
    const auto my = model[m][e][1];

    lines[n][a*M + m][e][0] = c*mx - s*my + px;
    lines[n][a*M + m][e][1] = s*mx + c*my + py;
}


using Indices = TensorProxy<int, 3>;
using Locations = TensorProxy<float, 3>;
using Dots = TensorProxy<float, 3>;
using Distances = TensorProxy<float, 3>;

__global__ void raycast_kernel(
    Angles::PTA angles, Positions::PTA positions, Lines::PTA lines, 
    Indices::PTA indices, Locations::PTA locations, Dots::PTA dots, Distances::PTA distances) {

    const auto n = blockIdx.x;
    const auto r = threadIdx.x;
    const auto a = blockIdx.y;

    // Generate the ray
    const float ang = angles[n][a]/180.f;
    const auto c = cospif(ang);
    const auto s = sinpif(ang);

    const Point p(positions[n][a]);

    const float R = indices.size(2);
    const Point u(1.f, ray_y(r, R));
    const Point ru(c*u.x - s*u.y, s*u.x + c*u.y);
    const auto rlen = ru.len();

    // Cast the ray
    const auto num_l = lines.widths[n];
    float nearest_idx = -1;
    float nearest_s = CUDART_INF_F; 
    float nearest_loc = CUDART_NAN_F;
    float nearest_dot = CUDART_NAN_F;
    for (int l=0; l < num_l; l++) {
        const Line L(lines[n][l]);
        const Point v(L.b.x - L.a.x, L.b.y - L.a.y);

        const auto q = intersect(p, ru, L);

        // dot of the ray and the line
        // this is _not_ the dot of the ray and the normal. we can get that easily enough from this,
        // but not vice versa. the thing that breaks the symmetry is that we only need the absolute
        // value of the dot with the normal.
        const auto dtop = dot(ru, v);
        const auto dbot = rlen*v.len();
        const auto dot = dtop/(dbot + 1.e-6f);

        // Use the agent radius as the near plane
        const bool hit = (0 <= q.t) & (q.t <= 1);
        // 1e-4 offset here is to suppress z fighting
        const bool better = (AGENT_RADIUS/rlen < q.s) & (q.s < nearest_s - 1.e-4f);
        if (hit & better) {
            nearest_s = q.s;

            nearest_idx = l;
            nearest_loc = q.t;
            nearest_dot = dot;
        }
    }

    indices[n][a][r] = nearest_idx;
    locations[n][a][r] = nearest_loc;
    dots[n][a][r] = nearest_dot;
    distances[n][a][r] = nearest_s*rlen;
}

using Screen = TensorProxy<float, 4>;

struct Filter {
    int l;
    int r;
    float lw;
    float rw;
};

__device__ Filter filter(float x, int w) {
    const auto y = fminf(x*(w+1), w-1);
    const int l = fmaxf(y-1, 0);
    const int r = fminf(y, w-1);

    const auto ld = fabsf(y - (l+1)) + 1.e-3f;
    const auto rd = fabsf(y - (r+1)) + 1.e-3f;
    const auto lw = rd/(ld+rd);
    const auto rw = ld/(ld+rd);

    return {l, r, lw, rw};
}

__global__ void shader_kernel(
    Indices::PTA indices, Locations::PTA locations, Dots::PTA dots,
    Lines::PTA lines, Lights::PTA lights, Textures::PTA textures, Baked::PTA baked, int F,
    Screen::PTA screen) {

    const auto n = blockIdx.x;
    const auto r = threadIdx.x;
    const auto a = blockIdx.y;
    const auto AF = screen.size(1)*F;

    auto s0 = 0.f, s1 = 0.f, s2 = 0.f;
    const auto l0 = indices[n][a][r];
    if (l0 >= 0) {
        const auto loc = locations[n][a][r];

        //TODO: Stick this in texture memory
        // Use the tex object API: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-object-api
        // Use CUDA arrays becuase they're the only ones that support normalized indexing and filtering
        // Use a single texture per design, as CUDA arrays are size-limited
        // Use linear interpolation. Do I have to start thinking of texels as fenceposts rather than centers?
        const auto start = lines.starts[n] + l0;
        const auto f = filter(loc, textures.widths[start]);
        const auto tex_l = textures[start][f.l]; 
        const auto tex_r = textures[start][f.r];

        // If it's a dynamic line, calculate the intensity on the fly. Else - if it's static - use the baked version.
        float intensity;
        if (l0 < AF) {
            const auto C = Point(lines[n][l0][0])*(1-loc) + Point(lines[n][l0][1])*loc;
            intensity = light_intensity(lines, lights, C, n, AF);
        } else { 
            intensity = f.lw*baked[start][f.l] + f.rw*baked[start][f.r];
        }

        // `dots` is the dot with the line; we want the dot with the normal
        const auto dot = 1 - dots[n][a][r]*dots[n][a][r];
        s0 = dot*intensity*(f.lw*tex_l[0] + f.rw*tex_r[0]);
        s1 = dot*intensity*(f.lw*tex_l[1] + f.rw*tex_r[1]);
        s2 = dot*intensity*(f.lw*tex_l[2] + f.rw*tex_r[2]);
    }
    screen[n][a][r][0] = s0;
    screen[n][a][r][1] = s1;
    screen[n][a][r][2] = s2;
}

__host__ Render render(const Scenery& scenery, const Agents& agents) {
    const uint N = agents.angles.size(0);
    const uint A = scenery.n_agents;
    const uint F = scenery.model.size(0);

    //TODO: This gives underfull warps. But it's also not the bottleneck, so who cares
    draw_kernel<<<N, {2, F, A}, 0, stream()>>>(
        agents.angles.pta(), agents.positions.pta(), scenery.model.pta(), scenery.lines.pta()); 

    auto indices(Indices::empty({N, A, RES}));
    auto locations(Locations::empty({N, A, RES}));
    auto dots(Dots::empty({N, A, RES}));
    auto distances(Distances::empty({N, A, RES}));
    raycast_kernel<<<{N, A}, {(uint) RES}, 0, stream()>>>(
        agents.angles.pta(), agents.positions.pta(), scenery.lines.pta(), 
        indices.pta(), locations.pta(), dots.pta(), distances.pta());

    auto screen(Screen::empty({N, A, RES, 3}));
    shader_kernel<<<{N, A}, {(uint) RES}, 0, stream()>>>(
        indices.pta(), locations.pta(), dots.pta(),
        scenery.lines.pta(), scenery.lights.pta(), scenery.textures.pta(), scenery.baked.pta(), F, screen.pta()); 

    return {indices.t, locations.t, dots.t, distances.t, screen.t};
}