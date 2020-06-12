#include "common.h"
#include <c10/cuda/CUDAStream.h>

Respawns mock_respawn() {
    auto options = at::device(at::kCUDA).dtype(at::kFloat);
    auto centers = at::range(0, 5, options).reshape({3, 1, 2});
    auto radii = at::range(6, 8, options).reshape({3, 1});
    auto lowers = at::range(9, 11, options).reshape({3, 1});
    auto uppers = at::range(12, 14, options).reshape({3, 1});
    auto widths = at::tensor({2, 1}, options.dtype(at::kInt));
    return Respawns(centers, radii, lowers, uppers, widths);
}

Drones mock_drones() {
    auto options = at::device(at::kCUDA).dtype(at::kFloat);
    auto angles = at::range(0, 1, options).reshape({2, 1});
    auto positions = at::range(1, 4, options).reshape({2, 1, 2});
    return Drones(angles, positions);
}

void test_respawn() {
    auto respawns = mock_respawn();
    auto reset = at::tensor({1, 1}, at::dtype(at::kByte).device(at::kCUDA));
    auto drones = mock_drones();
    respawn(respawns, reset, drones);
    c10::cuda::getDefaultCUDAStream().synchronize();
}

int main(int argc, char **argv) {
    initialize(.15/pow(2., .5), 64, 90.f);
    test_respawn();
    return 0;
}