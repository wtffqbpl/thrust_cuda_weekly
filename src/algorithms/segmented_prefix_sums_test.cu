#include <gtest/gtest.h>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

namespace {

/// \code
/// template < typename DerivedPolicy,
///            typename InputIterator1,
///            typename InputIterator2,
///            typename OutputIterator>
/// __host__ __device__ OutputIterator
/// inclusive_scan_by_key(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
///                         InputIterator1 first,
///                         InputIterator2 last1,
///                         InputIterator2 first2,
///                         OutputIterator result) { }
/// \endcode

// `inclusive_scan_by_key` computes an inclusive key-value or `segmented` prefix sum operation.
// The term `inclusive` means that each result includes the corresponding input operand in the
// partial sum. The term `segmented` means that the partial sums are broken into distinct
// segments. In other words, within each segment a separate inclusive scan operation is computed.
// Refer to the code sample below for example usage.
//  用来判断 InputIterator2 中放的key是否连续，如果连续，则进行 inclusive_scan 操作，如果不连续，
// 则从当前index开始重新进行pre sum计算.
// `inclusive_scan_by_key assumes `equal_to` as the binary predicate used to compare adjacent keys.
// Specifically, consecutive iterators `i` and `i + 1` in the range `[first1, last1)` belong to
// the same segment if `*i == *(i + 1)`, and belong to different segments otherwise.
// This version of `inclusive_scan_by_key` assumes `plus` as the associative operator used to
// perform the prefix sum. When the input and output sequences are the same, the scan is performed
// in-place.
// Results are not deterministic for pseudo-associative operators (e.g. addition of floating-point
// types). Results for pseudo-associative operators may vary from run to run.
// The algorithm's execution is parallelized as determined by `exec`.

void inclusive_scan_by_key_test () {
  int data[10] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512};
  int keys[10] = {0, 0, 0, 1, 1,  2,  3,  3,   3,   3};

  thrust::inclusive_scan_by_key(thrust::host, keys, keys + 10, data, data); // in place.
  // data is now {1, 3, 7, 8, 24, 32, 64, 192, 448, 960}

  EXPECT_EQ(data[0], 1);
  EXPECT_EQ(data[1], 3);
  EXPECT_EQ(data[2], 7);
  EXPECT_EQ(data[3], 8);
  EXPECT_EQ(data[4], 24);
  EXPECT_EQ(data[5], 32);
  EXPECT_EQ(data[6], 64);
  EXPECT_EQ(data[7], 192);
  EXPECT_EQ(data[8], 448);
  EXPECT_EQ(data[9], 960);
}

// BinaryPredicate for the head flag segment representation equivalent to
// thrust::not2(thrust::project2nd<int, int>()));
template <typename HeadFlagType>
struct head_flag_predicate : public thrust::binary_function<HeadFlagType, HeadFlagType, bool> {
  __host__ __device__
  bool operator()(HeadFlagType, HeadFlagType right) const { return !right; }
};

template <typename Vector>
void print(const Vector &v) {
  for (unsigned i = 0; i < v.size(); ++i)
    std::cout << v[i] << ' ';
  std::cout << '\n';
}

void inclusive_scan_by_key_test2() {
  int keys[] = {0, 0, 0, 1, 1, 2, 2, 2, 3, 4, 4, 5, 5, 5}; // segments represented with keys.
  int flags[] = {1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0}; // segments represented with head flags.
  int values[] = {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2}; // values corresponding to each key.

  constexpr int N = sizeof(keys) / sizeof(int); // number of elements.

  // copy input data to device.
  thrust::device_vector<int> d_keys(keys, keys + N);
  thrust::device_vector<int> d_flags(flags, flags + N);
  thrust::device_vector<int> d_values(values, values + N);

  // allocate storage for output
  thrust::device_vector<int> d_output(N);

  // inclusive scan using keys.
  thrust::inclusive_scan_by_key(d_keys.begin(), d_keys.end(),
                                d_values.begin(), d_output.begin());
#ifndef NDEBUG
  std::cout << "Inclusive Segmented Scan w/ Key Sequence\n";
  std::cout << "keys          : "; print(d_keys);
  std::cout << "Input values  : "; print(d_values);
  std::cout << "output values : "; print(d_output);
#endif

  EXPECT_EQ(d_output[0], 2); EXPECT_EQ(d_output[1], 4); EXPECT_EQ(d_output[2], 6);
  EXPECT_EQ(d_output[3], 2); EXPECT_EQ(d_output[4], 4);
  EXPECT_EQ(d_output[5], 2); EXPECT_EQ(d_output[6], 4); EXPECT_EQ(d_output[7], 6);
  EXPECT_EQ(d_output[8], 2);
  EXPECT_EQ(d_output[9], 2); EXPECT_EQ(d_output[10], 4);
  EXPECT_EQ(d_output[11], 2); EXPECT_EQ(d_output[12], 4); EXPECT_EQ(d_output[13], 6);

  // inclusive scan using head flags.
  // default场景下，inclusive_scan_by_key 的predicate比较的是keys[i] == keys[i + 1]，
  // 如果两个values指示的值相同，那么就可以做 pre sum 操作。如果不同，则重新开始。
  // 当然 inclusive_scan_by_key 还可以重新指定比较操作。当前这里的比较操作，是判断 keys[i + 1]
  // 是否为空来作为predicate operation.
  thrust::inclusive_scan_by_key(d_flags.begin(), d_flags.end(),
                                d_values.begin(), d_output.begin(),
                                head_flag_predicate<int>());

#ifndef NDEBUG
  std::cout << "\nInclusive Segmented Scan w/ Head Flag Sequence\n";
  std::cout << "head flags    : "; print(d_flags);
  std::cout << "input values  : "; print(d_values);
  std::cout << "output values : "; print(d_output);
#endif

  EXPECT_EQ(d_output[0], 2); EXPECT_EQ(d_output[1], 4); EXPECT_EQ(d_output[2], 6);
  EXPECT_EQ(d_output[3], 2); EXPECT_EQ(d_output[4], 4);
  EXPECT_EQ(d_output[5], 2); EXPECT_EQ(d_output[6], 4); EXPECT_EQ(d_output[7], 6);
  EXPECT_EQ(d_output[8], 8);
  EXPECT_EQ(d_output[9], 2);
  EXPECT_EQ(d_output[10], 2); EXPECT_EQ(d_output[11], 4);
  EXPECT_EQ(d_output[12], 2); EXPECT_EQ(d_output[13], 4);

  // exclusive scan using keys.
  thrust::exclusive_scan_by_key(d_keys.begin(), d_keys.end(),
                                d_values.begin(), d_output.begin());
#ifndef NDEBUG
  std::cout << "\nExclusive Segmented Scan w/ Head Flag Sequence\n";
  std::cout << "keys          : "; print(d_keys);
  std::cout << "input values  : "; print(d_values);
  std::cout << "output values : "; print(d_output);
#endif
  EXPECT_EQ(d_output[0], 0); EXPECT_EQ(d_output[1], 2); EXPECT_EQ(d_output[2], 4);
  EXPECT_EQ(d_output[3], 0); EXPECT_EQ(d_output[4], 2);
  EXPECT_EQ(d_output[5], 0); EXPECT_EQ(d_output[6], 2); EXPECT_EQ(d_output[7], 4);
  EXPECT_EQ(d_output[8], 0);
  EXPECT_EQ(d_output[9], 0); EXPECT_EQ(d_output[10], 2);
  EXPECT_EQ(d_output[11], 0); EXPECT_EQ(d_output[12], 2); EXPECT_EQ(d_output[13], 4);

  // exclusive scan using head flags.
  thrust::exclusive_scan_by_key(d_flags.begin(), d_flags.end(),
                                d_values.begin(), d_output.begin(),
                                0,
                                head_flag_predicate<int>());
#ifndef NDEBUG
  std::cout << "\nExclusive Segmented Scan w/ Head Flag Sequence\n";
  std::cout << "head flags    : "; print(d_flags);
  std::cout << "input values  : "; print(d_values);
  std::cout << "output values : "; print(d_output);
#endif
  // int flags[] = {1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0}; // segments represented with head flags.
  // int values[] = {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2}; // values corresponding to each key.
  EXPECT_EQ(d_output[0], 0); EXPECT_EQ(d_output[1], 2); EXPECT_EQ(d_output[2], 4);
  EXPECT_EQ(d_output[3], 0); EXPECT_EQ(d_output[4], 2);
  EXPECT_EQ(d_output[5], 0); EXPECT_EQ(d_output[6], 2); EXPECT_EQ(d_output[7], 4);
  EXPECT_EQ(d_output[8], 6);
  EXPECT_EQ(d_output[9], 0);
  EXPECT_EQ(d_output[10], 0); EXPECT_EQ(d_output[11], 2);
  EXPECT_EQ(d_output[12], 0); EXPECT_EQ(d_output[13], 2);
}

} // namespace

TEST(inclusive_scan_by_key, basic_test) {
  inclusive_scan_by_key_test();
  inclusive_scan_by_key_test2();
}