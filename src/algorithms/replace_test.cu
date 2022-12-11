#include <gtest/gtest.h>
#include <thrust/replace.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

namespace {

template <typename Vector>
void testReplaceSimple() {
  using T = typename Vector::value_type;

  Vector data(5);
  data[0] = 1, data[1] = 2, data[2] = 1, data[3] = 3, data[4] = 2;

  thrust::replace(data.begin(), data.end(), T{1}, T{4});
  thrust::replace(data.begin(), data.end(), T{2}, T{5});

  Vector result(5);
  result[0] = 4, result[1] = 5, result[2] = 4, result[3] = 3, result[4] = 5;

#ifndef NDEBUG
  std::cout << "Expected: \n";
  for (auto val : data)
    std::cout << val << std::endl;

  std::cout << "Actual: \n";
  for (auto val : data)
    std::cout << val << std::endl;
#endif

  EXPECT_EQ(data, result);
}

} // namespace

TEST(replace_test, basic_test) {
  testReplaceSimple<thrust::host_vector<int>>();
  testReplaceSimple<thrust::host_vector<float>>();
  testReplaceSimple<thrust::host_vector<double>>();

  testReplaceSimple<thrust::device_vector<int>>();
  testReplaceSimple<thrust::device_vector<float>>();
  testReplaceSimple<thrust::device_vector<double>>();
}