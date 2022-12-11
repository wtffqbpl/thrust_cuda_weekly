#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>

TEST(inclusive_scan, basic_test) {
  int data[6] = {1, 0, 2, 2, 1, 3};

  thrust::inclusive_scan(data, data + 6, data); // in-place scan.

  EXPECT_EQ(data[0], 1);
  EXPECT_EQ(data[1], 1);
  EXPECT_EQ(data[2], 3);
  EXPECT_EQ(data[3], 5);
  EXPECT_EQ(data[4], 6);
  EXPECT_EQ(data[5], 9);
}

namespace scan_simple {
template <typename T>
struct max_functor {
  __host__ __device__
  T operator()(T rhs, T lhs) const { return thrust::max(rhs, lhs); }
};

template <typename Vector>
void scan_simple_test() {
  typedef typename Vector::value_type T;
  typename Vector::iterator iter;

  Vector input(5);
  Vector result(5);
  Vector output(5);

  input[0] = 1, input[1] = 3, input[2] = -2, input[3] = 4; input[4] = -5;

  Vector input_copy(input);

  // inclusive scan
  iter = thrust::inclusive_scan(input.begin(), input.end(), output.begin());
  result[0] = 1, result[1] = 4, result[2] = 2, result[3] = 6, result[4] = 1;
  EXPECT_EQ(std::size_t(iter - output.begin()), input.size());
  EXPECT_EQ(input, input_copy);
  EXPECT_EQ(output, result);

  // exclusive scan
  iter = thrust::exclusive_scan(input.begin(), input.end(), output.begin(), T(0));
  result[0] = 0, result[1] = 1, result[2] = 4, result[3] = 2, result[4] = 6;
  EXPECT_EQ(std::size_t(iter - output.begin()), input.size());
  EXPECT_EQ(input, input_copy);
  EXPECT_EQ(output, result);

  // inclusive scan with init
  iter = thrust::exclusive_scan(input.begin(), input.end(), output.begin(), T(3));
  result[0] = 3, result[1] = 4, result[2] = 7, result[3] = 5, result[4] = 9;
  EXPECT_EQ(std::size_t(iter - output.begin()), input.size());
  EXPECT_EQ(input, input_copy);
  EXPECT_EQ(output, result);

  // inclusive scan with op
  iter = thrust::inclusive_scan(input.begin(), input.end(), output.begin(),
                                thrust::plus<T>());
  result[0] = 1, result[1] = 4, result[2] = 2, result[3] = 6, result[4] = 1;
  EXPECT_EQ(std::size_t(iter - output.begin()), input.size());
  EXPECT_EQ(input, input_copy);
  EXPECT_EQ(output, result);

  // exclusive scan with init and op
  iter = thrust::exclusive_scan(input.begin(), input.end(), output.begin(), T(3),
                                thrust::plus<T>());
  result[0] = 3, result[1] = 4, result[2] = 7, result[3] = 5, result[4] = 9;
  EXPECT_EQ(std::size_t(iter - output.begin()), input.size());
  EXPECT_EQ(input, input_copy);
  EXPECT_EQ(output, result);
}

} // namespace scan_simple

TEST(scan_test, basic_test) {
  scan_simple::scan_simple_test<thrust::host_vector<int>>();
  scan_simple::scan_simple_test<thrust::host_vector<float>>();
  scan_simple::scan_simple_test<thrust::host_vector<double>>();
  scan_simple::scan_simple_test<thrust::host_vector<long>>();
  scan_simple::scan_simple_test<thrust::host_vector<long long>>();

  scan_simple::scan_simple_test<thrust::device_vector<int>>();
  scan_simple::scan_simple_test<thrust::device_vector<float>>();
  scan_simple::scan_simple_test<thrust::device_vector<double>>();
  scan_simple::scan_simple_test<thrust::device_vector<long>>();
  scan_simple::scan_simple_test<thrust::device_vector<long long>>();
}


namespace scan_matrix {

void scan_matrix_by_rows0(thrust::device_vector<int>& u, int n, int m) {
  // launch a separate scan for each row in the matrix.
  for (int i = 0; i < n; ++i)
    thrust::inclusive_scan(u.begin() + m * i, u.begin() + m * (i + 1),
                           u.begin() + m * i);
}

} // namespace scan_matrix