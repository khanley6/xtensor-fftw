/*
 * xtensor-fftw
 * Copyright (c) 2017, Patrick Bos
 *
 * Distributed under the terms of the BSD 3-Clause License.
 *
 * The full license is in the file LICENSE, distributed with this software.
 */

#include <xtensor/xadapt.hpp>
#include <xtensor-fftw/basic.hpp>
#include "basic_interface.hpp"

///////////////////////////////////////////////////////////////////////////////
// Setup
///////////////////////////////////////////////////////////////////////////////

// GoogleTest fixture class
template <typename T>
class TransformAndInvert_FFT : public ::testing::Test {};

TYPED_TEST_CASE(TransformAndInvert_FFT, MyTypes);

///////////////////////////////////////////////////////////////////////////////
// Regular FFT (complex to complex)
///////////////////////////////////////////////////////////////////////////////

////
// Regular FFT: xarray
////

TYPED_TEST(TransformAndInvert_FFT, FFT_1D_xtensor) {
  xt::xtensor<std::complex<TypeParam>, 1> a = generate_complex_data<TypeParam, 1>(4);
  auto a_fourier = xt::fftw::fft<1>(a);
  // std::cout << "fourier transform of input before ifft (which is destructive!): " << a_fourier << std::endl;
  auto should_be_a = xt::fftw::ifft<1>(a_fourier);
  assert_results_complex(a, a_fourier, should_be_a);
}

TYPED_TEST(TransformAndInvert_FFT, FFT_2D_xarray) {
  //xt::xtensor<std::complex<TypeParam>, 2> a = generate_complex_data<TypeParam, 2>(4);
  xt::xarray<std::complex<TypeParam>> a = generate_complex_data<TypeParam, 2>(4);
  auto a_fourier = xt::fftw::fft<2>(a);
  // std::cout << "fourier transform of input before ifft (which is destructive!): " << a_fourier << std::endl;
  auto should_be_a = xt::fftw::ifft<2>(a_fourier);
  assert_results_complex(a, a_fourier, should_be_a);
}

TYPED_TEST(TransformAndInvert_FFT, FFT_3D_xtensor_deduced_dim) {
  xt::xtensor<std::complex<TypeParam>, 3> a = generate_complex_data<TypeParam, 3>(4);
  auto a_fourier = xt::fftw::fft(a);
  // std::cout << "fourier transform of input before ifft (which is destructive!): " << a_fourier << std::endl;
  auto should_be_a = xt::fftw::ifft(a_fourier);
  assert_results_complex(a, a_fourier, should_be_a);
}

TYPED_TEST(TransformAndInvert_FFT, FFT_nD_n_equals_4_xtensor) {
  xt::xtensor<std::complex<TypeParam>, 4> a = generate_complex_data<TypeParam, 4>(4);
  auto a_fourier = xt::fftw::fft<4>(a);
  // std::cout << "fourier transform of input before ifft (which is destructive!): " << a_fourier << std::endl;
  auto should_be_a = xt::fftw::ifft<4>(a_fourier);
  assert_results_complex(a, a_fourier, should_be_a);
}
