#include "common.h"
#include <c10/cuda/CUDAStream.h>

Agents mock_agents() {
    auto options = at::device(at::kCUDA).dtype(at::kFloat);
    auto angles = at::range(0, 1, options).reshape({2, 1});
    auto positions = at::range(1, 4, options).reshape({2, 1, 2});
    return Agents(angles, positions);
}

void test_respawn() {
    auto respawns = mock_respawn();
    auto reset = at::tensor({1, 1}, at::dtype(at::kByte).device(at::kCUDA));
    auto agents = mock_agents();
    respawn(respawns, reset, agents);
    c10::cuda::getDefaultCUDAStream().synchronize();
}

int main(int argc, char **argv) {
    initialize(.15/pow(2., .5), 64, 90.f);
    test_respawn();
    return 0;
}