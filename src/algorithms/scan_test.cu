#include <gtest/gtest.h>
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