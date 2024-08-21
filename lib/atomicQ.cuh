#include <cstdio>
#include <cuda_runtime.h>

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

namespace cudacpp {

template <typename T, size_t MAX_SIZE>
class cudaQueue {
private:
    T main_Mem[MAX_SIZE];
    unsigned int SIZE;
    unsigned int front_idx;
    unsigned int back_idx;

public:
    CUDA_CALLABLE_MEMBER cudaQueue() : SIZE(0), front_idx(0), back_idx(0) {}


    CUDA_CALLABLE_MEMBER void push(const T& value) {
        if (atomicAdd(&SIZE, 1) < MAX_SIZE) {
            main_Mem[atomicAdd(&back_idx, 1) % MAX_SIZE] = value;
        } else {
            atomicSub(&SIZE, 1);
            printf("Queue is full\n");
        }
    }

    CUDA_CALLABLE_MEMBER bool pop_and_front(T& result) {
        unsigned int cur_size = atomicSub(&SIZE, 1);
        if (cur_size > 0) {
            unsigned int idx = atomicAdd(&front_idx, 1) % MAX_SIZE;
            result = main_Mem[idx];

            return true;
        } else {
            atomicAdd(&SIZE, 1);
            return false;
        }
    }

    CUDA_CALLABLE_MEMBER bool empty() const {
        return SIZE == 0;
    }
    CUDA_CALLABLE_MEMBER void pop() {
        unsigned int cur_size = atomicSub(&SIZE, 1);
        if (cur_size > 0) {
            atomicAdd(&front_idx, 1);
        } else {
            atomicAdd(&SIZE, 1);
            printf("Queue is empty\n");
        }
    }

    CUDA_CALLABLE_MEMBER T front() const {
        unsigned int current_size = SIZE;
        unsigned int current_front_idx = front_idx;

        if (current_size > 0) {
            return main_Mem[current_front_idx % MAX_SIZE];
        } else {
            printf("Queue is empty\n");
            return T(); // Return default value
        }
    }
    CUDA_CALLABLE_MEMBER  int size()  {
        return SIZE;
    }
};

}
