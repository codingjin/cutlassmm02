#include <iostream>
#include <cuda_runtime.h>

int main() {
    int device = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Shared Memory per SM: " << prop.sharedMemPerMultiprocessor / 1024.0 << " KB" << std::endl;
    std::cout << "Shared Memory per Block: " << prop.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
    std::cout << "Max Shared Memory per Block (opt-in): " << prop.sharedMemPerBlockOptin / 1024.0 << " KB" << std::endl;
    std::cout << "Number of SMs: " << prop.multiProcessorCount << std::endl;

    return 0;
}
