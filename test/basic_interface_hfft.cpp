/*
 * xtensor-fftw
 * Copyright (c) 2017, Patrick Bos
 *
 * Distributed under the terms of the BSD 3-Clause License.
 *
 * The full license is in the file LICENSE, distributed with this software.
 *
 * Modifications:
 * Copyright (c) 2019, Kenneth Hanley
 */

#include <xtensor-fftw/basic.hpp>
#include "basic_interface.hpp"

///////////////////////////////////////////////////////////////////////////////
// Setup
///////////////////////////////////////////////////////////////////////////////

// GoogleTest fixture class
template <typename T>
class TransformAndInvert_hermFFT : public ::testing::Test {};

TYPED_TEST_CASE(TransformAndInvert_hermFFT, MyTypes);


///////////////////////////////////////////////////////////////////////////////
// Hermitian FFT (real spectrum)
///////////////////////////////////////////////////////////////////////////////

////
// Hermitian FFT: xarray
////

TYPED_TEST(TransformAndInvert_hermFFT, hermFFT_1D_xtensor) {
  xt::xtensor<std::complex<TypeParam>, 1> a = generate_hermitian_data<TypeParam, 1>(4);
  auto a_fourier = xt::fftw::hfft(a);
  // std::cout << "fourier transform of input before ifft (which is destructive!): " << a_fourier << std::endl;
  auto should_be_a = xt::fftw::ihfft(a_fourier);
  assert_results_complex(a, a_fourier, should_be_a);
}

TYPED_TEST(TransformAndInvert_hermFFT, hermFFT_2D_xarray) {
  xt::xarray<std::complex<TypeParam>> a = generate_hermitian_data<TypeParam, 2>(4);
  auto a_fourier = xt::fftw::hfft<2>(a);
  // std::cout << "fourier transform of input before ifft (which is destructive!): " << a_fourier << std::endl;
  auto should_be_a = xt::fftw::ihfft<2>(a_fourier);
  assert_results_complex(a, a_fourier, should_be_a);
}

TYPED_TEST(TransformAndInvert_hermFFT, hermFFT_3D_xtensor_deduced_dim) {
  xt::xtensor<std::complex<TypeParam>, 3> a = generate_hermitian_data<TypeParam, 3>(4);
  auto a_fourier = xt::fftw::hfft(a);
  // std::cout << "fourier transform of input before ifft (which is destructive!): " << a_fourier << std::endl;
  auto should_be_a = xt::fftw::ihfft(a_fourier);
  assert_results_complex(a, a_fourier, should_be_a);
}

TYPED_TEST(TransformAndInvert_hermFFT, hermFFT_nD_n_equals_4_xtensor) {
  xt::xtensor<std::complex<TypeParam>, 4> a = generate_hermitian_data<TypeParam, 4>(4);
  auto a_fourier = xt::fftw::hfft<4>(a);
  // std::cout << "fourier transform of input before ifft (which is destructive!): " << a_fourier << std::endl;
  auto should_be_a = xt::fftw::ihfft<4>(a_fourier);
  assert_results_complex(a, a_fourier, should_be_a);
}
