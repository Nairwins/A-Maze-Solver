#include <iostream>
#include <math.h>
#include <vector>

__global__ void generateWalls(int* maze, int width, int height, unsigned int seed) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= width || y >= height) return;

    curandState state;
    curand_init(seed, x + y * width, 0, &state);

    maze[y * width + x] = curand(&state) % 2; // 0 = empty, 1 = wall
}
