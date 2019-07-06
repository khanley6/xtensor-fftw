/*
 * xtensor-fftw
 * Copyright (c) 2017, Patrick Bos
 *
 * Distributed under the terms of the BSD 3-Clause License.
 *
 * The full license is in the file LICENSE, distributed with this software.
 */

#include <xtensor-fftw/basic.hpp>
#include <xtensor/xio.hpp>
#include "basic_interface.hpp"

///////////////////////////////////////////////////////////////////////////////
// Setup
///////////////////////////////////////////////////////////////////////////////

// GoogleTest fixture class
template <typename T>
class TransformAndInvert_realFFT : public ::testing::Test {};

TYPED_TEST_CASE(TransformAndInvert_realFFT, MyTypes);
//TYPED_TEST_CASE(TransformAndInvert_realFFT, double);


///////////////////////////////////////////////////////////////////////////////
// Real FFT (real input)
///////////////////////////////////////////////////////////////////////////////

////
// Real FFT: xarray
////
TYPED_TEST(TransformAndInvert_realFFT, realFFT_1D_xtensor) {
  xt::xtensor<TypeParam, 1> a = generate_data<TypeParam, 1>(4);
  auto a_fourier = xt::fftw::rfft(a);
  // std::cout << "fourier transform of input before ifft (which is destructive!): " << a_fourier << std::endl;
  auto should_be_a = xt::fftw::irfft(a_fourier);
  std::cout << "even a = " << a << std::endl;
  std::cout << "even a_fourier = " << a_fourier << std::endl;
  std::cout << "even should_be_a = " << should_be_a << std::endl;
  assert_results(a, a_fourier, should_be_a);
}

TYPED_TEST(TransformAndInvert_realFFT, realFFT_2D_xarray) {
  xt::xarray<TypeParam> a = generate_data<TypeParam, 2>(4);
  auto a_fourier = xt::fftw::rfft<2>(a);
  // std::cout << "fourier transform of input before ifft (which is destructive!): " << a_fourier << std::endl;
  auto should_be_a = xt::fftw::irfft<2>(a_fourier);
  assert_results(a, a_fourier, should_be_a);
}


TYPED_TEST(TransformAndInvert_realFFT, realFFT_3D_xtensor_deduced_dim) {
  xt::xtensor<TypeParam, 3> a = generate_data<TypeParam, 3>(4);
  auto a_fourier = xt::fftw::rfft(a);
  // std::cout << "fourier transform of input before ifft (which is destructive!): " << a_fourier << std::endl;
  auto should_be_a = xt::fftw::irfft(a_fourier);
  assert_results(a, a_fourier, should_be_a);
}

TYPED_TEST(TransformAndInvert_realFFT, realFFT_nD_n_equals_4_xtensor) {
  xt::xtensor<TypeParam, 4> a = generate_data<TypeParam, 4>(4);
  auto a_fourier = xt::fftw::rfft<4>(a);
  // std::cout << "fourier transform of input before ifft (which is destructive!): " << a_fourier << std::endl;
  auto should_be_a = xt::fftw::irfft<4>(a_fourier);
  assert_results(a, a_fourier, should_be_a);
}


// odd data sizes

TYPED_TEST(TransformAndInvert_realFFT, realFFT_1D_xtensor_odd_size) {
  xt::xtensor<TypeParam, 1> a = generate_data<TypeParam, 1>(5);
  auto a_fourier = xt::fftw::rfft(a);
  // std::cout << "fourier transform of input before ifft (which is destructive!): " << a_fourier << std::endl;
  auto should_be_a = xt::fftw::irfft(a_fourier, true);
  assert_results(a, a_fourier, should_be_a);
}

TYPED_TEST(TransformAndInvert_realFFT, realFFT_2D_xarray_odd_size) {
  xt::xarray<TypeParam> a = generate_data<TypeParam, 2>(5);
  auto a_fourier = xt::fftw::rfft<2>(a);
  // std::cout << "fourier transform of input before ifft (which is destructive!): " << a_fourier << std::endl;
  auto should_be_a = xt::fftw::irfft<2>(a_fourier, true);
  assert_results(a, a_fourier, should_be_a);
}

TYPED_TEST(TransformAndInvert_realFFT, realFFT_3D_xtensor_deduced_dim_odd_size) {
  xt::xtensor<TypeParam, 3> a = generate_data<TypeParam, 3>(5);
  auto a_fourier = xt::fftw::rfft(a);
  // std::cout << "fourier transform of input before ifft (which is destructive!): " << a_fourier << std::endl;
  auto should_be_a = xt::fftw::irfft(a_fourier, true);
  assert_results(a, a_fourier, should_be_a);
}

TYPED_TEST(TransformAndInvert_realFFT, realFFT_nD_n_equals_4_xtensor_odd_size) {
  xt::xtensor<TypeParam, 4> a = generate_data<TypeParam, 4>(5);
  auto a_fourier = xt::fftw::rfft<4>(a);
  // std::cout << "fourier transform of input before ifft (which is destructive!): " << a_fourier << std::endl;
  auto should_be_a = xt::fftw::irfft<4>(a_fourier, true);
  assert_results(a, a_fourier, should_be_a);
}
