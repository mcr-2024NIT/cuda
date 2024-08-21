#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <cuda/std/initializer_list>
#include <thrust/copy.h>

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

namespace cudacpp {

    // 基本の1次元ベクトル
    template <typename T>
    class cudaVector {
        T* mainMemory = nullptr;
        uint_fast64_t SIZE = 0;

    public:
        using iterator = T*;
        using const_iterator = const T* const;

        CUDA_CALLABLE_MEMBER cudaVector() = default;

        CUDA_CALLABLE_MEMBER cudaVector(uint_fast64_t allocSize)
            : mainMemory(new T[allocSize]), SIZE(allocSize) {}

        CUDA_CALLABLE_MEMBER cudaVector(uint_fast64_t allocSize, const T& initValue)
            : SIZE(allocSize), mainMemory(new T[allocSize]) {
            for (uint_fast64_t i = 0; i < SIZE; ++i) {
                mainMemory[i] = initValue;
            }
        }

        CUDA_CALLABLE_MEMBER cudaVector(cuda::std::initializer_list<T> initList)
            : mainMemory(new T[initList.size()]), SIZE(initList.size()) {
            thrust::copy(initList.begin(), initList.end(), mainMemory);
        }

        CUDA_CALLABLE_MEMBER cudaVector(const cudaVector<T>& lVector)
            : SIZE(lVector.SIZE), mainMemory(new T[lVector.SIZE]) {
            thrust::copy(lVector.mainMemory, lVector.mainMemory + SIZE, mainMemory);
        }

        CUDA_CALLABLE_MEMBER cudaVector(cudaVector<T>&& lVector) noexcept
            : SIZE(lVector.SIZE), mainMemory(lVector.mainMemory) {
            lVector.mainMemory = nullptr;
            lVector.SIZE = 0;
        }

        CUDA_CALLABLE_MEMBER cudaVector<T>& operator=(const cudaVector<T>& lVector) {
            if (this != &lVector) {
                delete[] mainMemory;
                SIZE = lVector.SIZE;
                mainMemory = new T[SIZE];
                thrust::copy(lVector.mainMemory, lVector.mainMemory + SIZE, mainMemory);
            }
            return *this;
        }

        CUDA_CALLABLE_MEMBER cudaVector<T>& operator=(cudaVector<T>&& lVector) noexcept {
            if (this != &lVector) {
                delete[] mainMemory;
                SIZE = lVector.SIZE;
                mainMemory = lVector.mainMemory;
                lVector.mainMemory = nullptr;
                lVector.SIZE = 0;
            }
            return *this;
        }

        CUDA_CALLABLE_MEMBER void operator+=(const cudaVector<T>& lVector) {
            T* tempMemory = new T[SIZE];
            thrust::copy(mainMemory, mainMemory + SIZE, tempMemory);
            delete[] mainMemory;
            mainMemory = new T[SIZE + lVector.SIZE];
            thrust::copy(tempMemory, tempMemory + SIZE, mainMemory);
            thrust::copy(lVector.mainMemory, lVector.mainMemory + lVector.SIZE, mainMemory + SIZE);
            SIZE += lVector.SIZE;
            delete[] tempMemory;
        }

        CUDA_CALLABLE_MEMBER bool operator==(const cudaVector<T>& other) const {
            if (SIZE != other.SIZE) return false;
            for (uint_fast64_t i = 0; i < SIZE; ++i) {
                if (!(mainMemory[i] == other.mainMemory[i])) return false;
            }
            return true;
        }

        CUDA_CALLABLE_MEMBER inline constexpr void clear() noexcept {
            delete[] mainMemory;
            mainMemory = nullptr;
            SIZE = 0;
        }

        CUDA_CALLABLE_MEMBER inline constexpr void erase(const_iterator eraseBegin, const_iterator eraseEnd) noexcept {
            if (eraseBegin >= eraseEnd)
                return;

            uint_fast64_t startDistance = eraseBegin - begin();
            uint_fast64_t deleteDistance = eraseEnd - eraseBegin;
            uint_fast64_t endDistance = end() - eraseEnd;

            if (deleteDistance == SIZE) {
                clear();
                return;
            }

            thrust::copy(eraseEnd, end(), mainMemory + startDistance);
            SIZE -= deleteDistance;
        }

        CUDA_CALLABLE_MEMBER inline constexpr void insert(const_iterator insertPos, T value) noexcept {
            if (insertPos > end() || insertPos < begin())
                return;

            uint_fast64_t startDistance = insertPos - begin();
            uint_fast64_t endDistance = end() - insertPos;
            T* tempMemory = new T[SIZE];
            thrust::copy(mainMemory, mainMemory + SIZE, tempMemory);
            delete[] mainMemory;
            mainMemory = new T[SIZE + 1];
            thrust::copy(tempMemory, tempMemory + startDistance, mainMemory);
            mainMemory[startDistance] = value;
            thrust::copy(tempMemory + startDistance, tempMemory + SIZE, mainMemory + startDistance + 1);
            delete[] tempMemory;
            ++SIZE;
        }

        CUDA_CALLABLE_MEMBER inline constexpr T& at(uint_fast64_t pos) noexcept {
            return mainMemory[pos];
        }

        CUDA_CALLABLE_MEMBER inline constexpr T& operator[](uint_fast64_t index) noexcept {
            return mainMemory[index];
        }
        CUDA_CALLABLE_MEMBER inline constexpr T& operator[](uint_fast64_t index) const noexcept {
            return mainMemory[index];
        }

        CUDA_CALLABLE_MEMBER inline constexpr const_iterator begin() noexcept {
            return mainMemory;
        }

        CUDA_CALLABLE_MEMBER inline constexpr const_iterator end() noexcept {
            return mainMemory + SIZE;
        }
        CUDA_CALLABLE_MEMBER inline constexpr const_iterator begin() const noexcept {
            return mainMemory;
        }

        CUDA_CALLABLE_MEMBER inline constexpr const_iterator end() const noexcept {
            return mainMemory + SIZE;
        }

        CUDA_CALLABLE_MEMBER inline constexpr T front() noexcept {
            return mainMemory[0];
        }
        CUDA_CALLABLE_MEMBER inline constexpr T front() const noexcept {
            return mainMemory[0];
        }

        CUDA_CALLABLE_MEMBER inline constexpr T back() noexcept {
            return mainMemory[SIZE - 1];
        }
        CUDA_CALLABLE_MEMBER inline constexpr T back() const noexcept {
            return mainMemory[SIZE - 1];
        }
        CUDA_CALLABLE_MEMBER inline constexpr void push_back(T value) noexcept {
            T* tempMemory = new T[SIZE];
            thrust::copy(mainMemory, mainMemory + SIZE, tempMemory);
            delete[] mainMemory;
            mainMemory = new T[SIZE + 1];
            thrust::copy(tempMemory, tempMemory + SIZE, mainMemory);
            mainMemory[SIZE] = value;
            delete[] tempMemory;
            ++SIZE;
        }

        CUDA_CALLABLE_MEMBER inline constexpr void pop_back() noexcept {
            if (SIZE == 0) {
                printf("\nPopping empty Vector\n");
                exit(-1);
            }
            T* tempMemory = new T[SIZE];
            thrust::copy(mainMemory, mainMemory + SIZE, tempMemory);
            delete[] mainMemory;
            mainMemory = new T[SIZE - 1];
            thrust::copy(tempMemory, tempMemory + SIZE - 1, mainMemory);
            delete[] tempMemory;
            --SIZE;
        }

        CUDA_CALLABLE_MEMBER inline constexpr bool empty() noexcept {
            return SIZE == 0;
        }
        CUDA_CALLABLE_MEMBER inline constexpr bool empty() const noexcept {
            return SIZE == 0;
        }
        CUDA_CALLABLE_MEMBER inline constexpr int size() noexcept {
            return SIZE;
        }
        CUDA_CALLABLE_MEMBER inline constexpr int size() const noexcept {
            return SIZE;
        }

        // resizeメソッドの追加
        CUDA_CALLABLE_MEMBER void resize(uint_fast64_t newSize, const T& initValue = T()) {
            T* tempMemory = new T[newSize];
            uint_fast64_t minSize = (newSize < SIZE) ? newSize : SIZE;
            thrust::copy(mainMemory, mainMemory + minSize, tempMemory);
            for (uint_fast64_t i = minSize; i < newSize; ++i) {
                tempMemory[i] = initValue;
            }
            delete[] mainMemory;
            mainMemory = tempMemory;
            SIZE = newSize;
        }

        CUDA_CALLABLE_MEMBER ~cudaVector() noexcept {
            delete[] mainMemory;
        }
    };

    // 多次元のベクトルを再帰的に定義
    template <typename T, size_t N>
    class cudaVectorND {
        cudaVectorND<T, N - 1>* data = nullptr;
        uint_fast64_t SIZE = 0;

    public:
        using iterator = cudaVectorND<T, N - 1>*;
        using const_iterator = const cudaVectorND<T, N - 1>*;

        CUDA_CALLABLE_MEMBER cudaVectorND() = default;

        CUDA_CALLABLE_MEMBER cudaVectorND(uint_fast64_t allocSize)
            : SIZE(allocSize) {
            data = new cudaVectorND<T, N - 1>[allocSize];
        }

        CUDA_CALLABLE_MEMBER cudaVectorND(uint_fast64_t allocSize, const T& initValue)
            : SIZE(allocSize) {
            data = new cudaVectorND<T, N - 1>[allocSize];
            for (uint_fast64_t i = 0; i < SIZE; ++i) {
                data[i] = cudaVectorND<T, N - 1>(allocSize, initValue);
            }
        }

        CUDA_CALLABLE_MEMBER cudaVectorND(cuda::std::initializer_list<cudaVectorND<T, N - 1>> initList)
            : SIZE(initList.size()) {
            data = new cudaVectorND<T, N - 1>[initList.size()];
            thrust::copy(initList.begin(), initList.end(), data);
        }

        CUDA_CALLABLE_MEMBER cudaVectorND(const cudaVectorND<T, N>& lVector)
            : SIZE(lVector.SIZE) {
            data = new cudaVectorND<T, N - 1>[SIZE];
            for (uint_fast64_t i = 0; i < SIZE; ++i) {
                data[i] = lVector.data[i];
            }
        }

        CUDA_CALLABLE_MEMBER cudaVectorND(cudaVectorND<T, N>&& lVector) noexcept
            : SIZE(lVector.SIZE), data(lVector.data) {
            lVector.data = nullptr;
            lVector.SIZE = 0;
        }

        CUDA_CALLABLE_MEMBER cudaVectorND<T, N>& operator=(const cudaVectorND<T, N>& lVector) {
            if (this != &lVector) {
                delete[] data;
                SIZE = lVector.SIZE;
                data = new cudaVectorND<T, N - 1>[SIZE];
                for (uint_fast64_t i = 0; i < SIZE; ++i) {
                    data[i] = lVector.data[i];
                }
            }
            return *this;
        }

        CUDA_CALLABLE_MEMBER cudaVectorND<T, N>& operator=(cudaVectorND<T, N>&& lVector) noexcept {
            if (this != &lVector) {
                delete[] data;
                SIZE = lVector.SIZE;
                data = lVector.data;
                lVector.data = nullptr;
                lVector.SIZE = 0;
            }
            return *this;
        }

        CUDA_CALLABLE_MEMBER cudaVectorND<T, N>& operator[](uint_fast64_t index) {
            return data[index];
        }

        CUDA_CALLABLE_MEMBER const cudaVectorND<T, N>& operator[](uint_fast64_t index) const {
            return data[index];
        }

        CUDA_CALLABLE_MEMBER uint_fast64_t size() const noexcept {
            return SIZE;
        }

        CUDA_CALLABLE_MEMBER ~cudaVectorND() noexcept {
            delete[] data;
        }
    };

    // 特化で終端のケースを定義
    template <typename T>
    class cudaVectorND<T, 1> {
        T* data = nullptr;
        uint_fast64_t SIZE = 0;

    public:
        CUDA_CALLABLE_MEMBER cudaVectorND() = default;

        CUDA_CALLABLE_MEMBER cudaVectorND(uint_fast64_t allocSize)
            : SIZE(allocSize) {
            data = new T[allocSize];
        }

        CUDA_CALLABLE_MEMBER cudaVectorND(uint_fast64_t allocSize, const T& initValue)
            : SIZE(allocSize) {
            data = new T[allocSize];
            for (uint_fast64_t i = 0; i < SIZE; ++i) {
                data[i] = initValue;
            }
        }

        CUDA_CALLABLE_MEMBER cudaVectorND(cuda::std::initializer_list<T> initList)
            : SIZE(initList.size()) {
            data = new T[initList.size()];
            thrust::copy(initList.begin(), initList.end(), data);
        }

        CUDA_CALLABLE_MEMBER cudaVectorND(const cudaVectorND<T, 1>& lVector)
            : SIZE(lVector.SIZE) {
            data = new T[SIZE];
            thrust::copy(lVector.data, lVector.data + SIZE, data);
        }

        CUDA_CALLABLE_MEMBER cudaVectorND(cudaVectorND<T, 1>&& lVector) noexcept
            : SIZE(lVector.SIZE), data(lVector.data) {
            lVector.data = nullptr;
            lVector.SIZE = 0;
        }

        CUDA_CALLABLE_MEMBER cudaVectorND<T, 1>& operator=(const cudaVectorND<T, 1>& lVector) {
            if (this != &lVector) {
                delete[] data;
                SIZE = lVector.SIZE;
                data = new T[SIZE];
                thrust::copy(lVector.data, lVector.data + SIZE, data);
            }
            return *this;
        }

        CUDA_CALLABLE_MEMBER cudaVectorND<T, 1>& operator=(cudaVectorND<T, 1>&& lVector) noexcept {
            if (this != &lVector) {
                delete[] data;
                SIZE = lVector.SIZE;
                data = lVector.data;
                lVector.data = nullptr;
                lVector.SIZE = 0;
            }
            return *this;
        }

        CUDA_CALLABLE_MEMBER T& operator[](uint_fast64_t index) {
            return data[index];
        }

        CUDA_CALLABLE_MEMBER const T& operator[](uint_fast64_t index) const {
            return data[index];
        }

        CUDA_CALLABLE_MEMBER uint_fast64_t size() const noexcept {
            return SIZE;
        }

        CUDA_CALLABLE_MEMBER ~cudaVectorND() noexcept {
            delete[] data;
        }
    };

}
