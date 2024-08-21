#ifndef ATOMICMAP_CUH
#define ATOMICMAP_CUH

#include <cstddef>
#include <cuda_runtime.h>

__device__ __inline__ bool atomicCAS_bool(int *address, bool compare, bool val) {
    int old = *address;
    int assumed;

    do {
        assumed = old;
        if ((bool)(old & 1) != compare)
            return (bool)(old & 1);
        old = atomicCAS(address, assumed, val ? 1 : 0);
    } while (assumed != old);

    return (bool)(old & 1);
}

namespace cudacpp {

// ハッシュテーブルのエントリ
template <typename T1, typename T2>
struct Entry {
    T1 first;
    T2 second;
    int occupied;
};

// CUDA対応のハッシュマップ
template <typename T1, typename T2, typename Hash, size_t Capacity>
class cuda_unordered_map {
private:
    Entry<T1, T2>* entries;
    size_t size;            // 現在のサイズ
    Hash hasher;            // ハッシュ関数

    // ハッシュ関数の計算
__device__ size_t hash_function(const T1& key) const {

    return      hasher(key) % Capacity;
    }


public:
    // イテレータのクラス
    class iterator {
    private:
        Entry<T1, T2>* entries;
        size_t index;

    public:
        __device__ iterator(Entry<T1, T2>* entries, size_t index)
            : entries(entries), index(index) {}

        __device__ Entry<T1, T2>& operator*() {
            return entries[index];
        }

        __device__ Entry<T1, T2>* operator->() {
            return &entries[index];
        }

        __device__ iterator& operator++() {
            while (++index < Capacity && !entries[index].occupied);
            return *this;
        }

        __device__ bool operator!=(const iterator& other) const {
            return index != other.index;
        }

        __device__ bool operator==(const iterator& other) const {
            return index == other.index;
        }
    };

    // コンストラクタ
    __device__ cuda_unordered_map()
        : size(0), hasher(Hash()) {
        entries = new Entry<T1, T2>[Capacity];
        for (size_t i = 0; i < Capacity; ++i) {
            entries[i].occupied = false;
        }


    }

    // デストラクタ
    __device__ ~cuda_unordered_map() {
        delete[] entries;
    }

    // 要素の挿入
    __device__ void insert(const T1& key, const T2& value) {
        size_t idx = hash_function(key);
        printf("idx %d\n", idx);
        while (true) {
            printf("Occupied:" );
            bool expected = false;
            printf("Occupied: %d\n", entries[idx].occupied);
            if (atomicCAS_bool((int*)&entries[idx].occupied, expected, true) == expected) {
                //occupiedの表示
                printf("Occupied: %d\n", entries[idx].occupied);
                entries[idx].first = key;
                entries[idx].second = value;
                atomicAdd(&size, 1);
                break;
            }
            idx = (idx + 1) % Capacity;
        }
    }

    // 要素の検索
__device__ iterator find(const T1& key) {
    size_t idx = hash_function(key);
    printf("Finding key: %d at index %llu\n", key, static_cast<unsigned long long>(idx));
    printf(entries[idx].occupied ? "Occupied\n" : "Not occupied\n");
    while (entries[idx].occupied) {
        printf("Entry occupied at index %llu, key: %d\n", idx, static_cast<unsigned long long>(entries[idx].first));

        if (entries[idx].first == key) {
            printf("bbb\n");
            return iterator(entries, idx);
        }
        idx = (idx + 1) % Capacity;
        printf("Next index: %zu\n", idx);
    }

    printf("ccc\n");
    return end();
}


    // 要素の存在確認
    __device__ size_t count(const T1& key) const {
        size_t idx = hash_function(key);
        while (entries[idx].occupied) {
            if (entries[idx].first == key) {
                return 1;
            }
            idx = (idx + 1) % Capacity;
        }
        return 0;
    }

    __device__ iterator begin() {
        size_t idx = 0;
        while (idx < Capacity && !entries[idx].occupied) {
            ++idx;
        }
        return iterator(entries, idx);
    }

    __device__ iterator end() {
        return iterator(entries, Capacity);
    }

    __device__ T2& operator[](const T1& key) {
        size_t idx = hash_function(key);
        while (true) {
            if (entries[idx].occupied && entries[idx].first == key) {
                return entries[idx].second;
            }
            bool expected = false;
            if (atomicCAS(&entries[idx].occupied, expected, true) == expected) {
                entries[idx].first = key;
                atomicAdd(&size, 1);
                return entries[idx].second;
            }
            idx = (idx + 1) % Capacity;
        }
    }

    __device__ __host__ static size_t size_of_entries() {
        return sizeof(Entry<T1, T2>);
    }
};

} // namespace cudacpp

#endif // ATOMICMAP_CUH
