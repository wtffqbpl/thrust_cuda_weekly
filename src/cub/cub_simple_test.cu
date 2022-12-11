#include <cub/cub.cuh>
#include <gtest/gtest.h>
#include <iostream>
#include <numeric>

namespace {

// Block-sorting CUDA kernel
template <typename Key,
         unsigned BLOCK_THREADS = 256,
         unsigned ITEMS_PER_THREAD = 16>
__global__ void BlockSortKernel(Key *d_in, Key *d_out) {
  using namespace cub;


  // Specialize BlockRadixSort, BlockLoad, and BlockStore for 128 threads
  // owning 16 integer items each.
  using BlockRadixSort = BlockRadixSort<Key, BLOCK_THREADS, ITEMS_PER_THREAD>;
  using BlockLoad = BlockLoad <Key, BLOCK_THREADS, ITEMS_PER_THREAD, BLOCK_LOAD_TRANSPOSE>;
  using BlockStore = BlockStore<Key, BLOCK_THREADS, ITEMS_PER_THREAD, BLOCK_STORE_TRANSPOSE>;

  // Alocate shared memory
  __shared__ union {
    typename BlockRadixSort::TempStorage sort;
    typename BlockLoad::TempStorage load;
    typename BlockStore::TempStorage store;
  } temp_storage;

  // OffsetT ofr this block's ment
  auto block_offset = blockIdx.x * BLOCK_THREADS * ITEMS_PER_THREAD;

  // Obtain a segment of 2048 consecutive keys that are blocked across threads.
  Key thread_keys[ITEMS_PER_THREAD];
  BlockLoad(temp_storage.load).Load(d_in + block_offset, thread_keys);
  __syncthreads();

  // Collectively sort the keys.
  BlockRadixSort(temp_storage.sort).Sort(thread_keys);
  __syncthreads();

  // Store the stored segment.
  BlockStore(temp_storage.store).Store(d_out + block_offset, thread_keys);
}

template <typename Key,
         unsigned BLOCK_THREADS = 256,
         unsigned ITEMS_PER_THREAD = 16>
bool cub_basic_test() {
  constexpr unsigned TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;

  Key *h_in = new Key[TILE_SIZE];
  std::iota(h_in, h_in + TILE_SIZE, 0);

  // initialize device arrays.
  Key *d_in = nullptr;
  Key *d_out = nullptr;
  cudaMalloc(&d_in, sizeof(Key) * TILE_SIZE);
  cudaMalloc(&d_out, sizeof(Key) * TILE_SIZE);

  cudaMemcpy(d_in, h_in, sizeof(Key) * TILE_SIZE, cudaMemcpyHostToDevice);
  BlockSortKernel<Key, BLOCK_THREADS, ITEMS_PER_THREAD><<<1, BLOCK_THREADS>>>(d_in, d_out);

  cudaDeviceSynchronize();
  if (cudaPeekAtLastError() != cudaSuccess) {
    std::cout << "Execute kernel function failed.\n";
    std::exit(1);
  }

  return true;
}

} // namespace

TEST(cub_test,basic_test) {
  EXPECT_TRUE(cub_basic_test<int>());
  EXPECT_TRUE(cub_basic_test<float>());
  EXPECT_TRUE(cub_basic_test<double>());
}