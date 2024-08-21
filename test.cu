#include <cstdio>
#include <cuda_runtime.h>
#include "lib/atomicQ.cuh"
#include "lib/atomicMap.cuh"
#include <Cudacpp/cudaVector.cuh>

constexpr int queueSize = 100;
struct hash {
    __device__ size_t operator()(int key) const {
        return key;
    }
};

// カーネル関数
__global__ void test() {
    extern __shared__ int sharedMem[];
    cudacpp::cudaQueue<int, queueSize>* queue = reinterpret_cast<cudacpp::cudaQueue<int, queueSize>*>(sharedMem);

    cudacpp::cuda_unordered_map<int, int, hash,255>* map =  (cudacpp::cuda_unordered_map<int, int, hash,255>*)&queue[1];

    int tid = threadIdx.x;
    int value;
    queue->push(tid);
queue->push(tid * 2);
queue->push(tid * 4);
    __syncthreads();

    
    while (tid < queue->size()) {
        if (tid == 0)
        {
            map->insert(0, 0);
        }
        
        queue->pop_and_front(value);
        printf("bbbb\n");
        auto it =  map->find(value);
        printf("aaaaa\n");
        
        if (it== map->end())
        {
    printf("Thread %d: %d\n", tid, value);
        map->insert(value, value*2);
        }


        
        printf("Thread %d: %d\n", tid, value);
        queue->push(value);
}
}


int main() {
    int sharedMemSize = sizeof(cudacpp::cudaQueue<int, queueSize>) 
                  + sizeof(cudacpp::cuda_unordered_map<int, int, hash,256>);


    test<<<1, 8, sharedMemSize>>>();
    cudaDeviceSynchronize();

    return 0;
}
