#include <algorithm>
#include <gtest/gtest.h>
#include <iostream>
#include <list>
#include <thrust/device_vector.h>

TEST(hello_test, test) {
  std::cout << "Hello world" << std::endl;
}

namespace {

template <typename ContainerTy, typename T>
struct check_res {
  unsigned idx = 0;
  const ContainerTy &coll_;

  explicit check_res(const ContainerTy &coll) : coll_(coll) {}
  bool operator()(const T val) { return val == coll_[idx++]; }
};

bool check_vector() {
  // create an STL list with 4 values
  std::list<int> stl_list;
  stl_list.push_back(10);
  stl_list.push_back(20);
  stl_list.push_back(30);
  stl_list.push_back(40);

  // initialize a device_vector with the list
  thrust::device_vector<int> D(stl_list.begin(), stl_list.end());

  // copy a device_vector into an STL vector.
  std::vector<int> stl_vector(D.size());
  thrust::copy(D.begin(), D.end(), stl_vector.begin());

  std::vector<int> expects(stl_list.begin(), stl_list.end());

  return std::all_of(stl_list.begin(), stl_list.end(),
                     check_res<std::vector<int>, int>(stl_vector));
}

} // namespace

TEST(vector_test, device_vector_test) {
  EXPECT_TRUE(check_vector());
}