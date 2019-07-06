/*
 * xtensor-fftw
 * Copyright (c) 2017, Patrick Bos
 *
 * Distributed under the terms of the BSD 3-Clause License.
 *
 * The full license is in the file LICENSE, distributed with this software.
 */

#include <complex>
#include <xtensor-fftw/basic.hpp>
#include "basic_interface.hpp"

///////////////////////////////////////////////////////////////////////////////
// Setup
///////////////////////////////////////////////////////////////////////////////

// GoogleTest fixture class
template <typename T>
class TransformAndInvert_hermFFT : public ::testing::Test {};

//TYPED_TEST_CASE(TransformAndInvert_hermFFT, MyTypes);
TYPED_TEST_CASE(TransformAndInvert_hermFFT, float);


///////////////////////////////////////////////////////////////////////////////
// Hermitian FFT (real spectrum)
///////////////////////////////////////////////////////////////////////////////

////
// Hermitian FFT: xarray
////
using namespace std::complex_literals;

TYPED_TEST(TransformAndInvert_hermFFT, hermFFT_1D_xtensor) {
  //xt::xtensor<std::complex<TypeParam>, 1> a = generate_hermitian_data<TypeParam, 1,  xt::fftw::hfft<TypeParam,1>, xt::fftw::ihfft<TypeParam,1>>(4);
  xt::xtensor<std::complex<TypeParam>, 1> a{ {4.007308e+36,0.000000e+00}, {1.153787e+36,4.222010e+36}, {6.085475e+34,1.938944e+36}, {2.911800e+35,0.000000e+00}};
  auto a_fourier = xt::fftw::hfft(a);
  // std::cout << "fourier transform of input before ifft (which is destructive!): " << a_fourier << std::endl;
  auto should_be_a = xt::fftw::ihfft(a_fourier);
  assert_results_complex(a, a_fourier, should_be_a);
}

/*
TYPED_TEST(TransformAndInvert_hermFFT, hermFFT_2D_xarray) {
  xt::xarray<std::complex<TypeParam>, xt::layout_type::row_major> a = generate_hermitian_data<TypeParam, 2, xt::fftw::hfft2, xt::fftw::ihfft2>(4);
  auto a_fourier = xt::fftw::hfft2(a);
  // std::cout << "fourier transform of input before ifft (which is destructive!): " << a_fourier << std::endl;
  auto should_be_a = xt::fftw::ihfft2(a_fourier);
  assert_results_complex(a, a_fourier, should_be_a);
}

TYPED_TEST(TransformAndInvert_hermFFT, hermFFT_3D_xarray) {
  xt::xarray<std::complex<TypeParam>, xt::layout_type::row_major> a = generate_hermitian_data<TypeParam, 3, xt::fftw::hfft3, xt::fftw::ihfft3>(4);
  auto a_fourier = xt::fftw::hfft3(a);
  // std::cout << "fourier transform of input before ifft (which is destructive!): " << a_fourier << std::endl;
  auto should_be_a = xt::fftw::ihfft3(a_fourier);
  assert_results_complex(a, a_fourier, should_be_a);
}

TYPED_TEST(TransformAndInvert_hermFFT, hermFFT_nD_n_equals_4_xarray) {
  xt::xarray<std::complex<TypeParam>, xt::layout_type::row_major> a = generate_hermitian_data<TypeParam, 4,  xt::fftw::hfftn<4>, xt::fftw::ihfftn<4>>(4);
  auto a_fourier = xt::fftw::hfftn<4>(a);
  // std::cout << "fourier transform of input before ifft (which is destructive!): " << a_fourier << std::endl;
  auto should_be_a = xt::fftw::ihfftn<4>(a_fourier);
  assert_results_complex(a, a_fourier, should_be_a);
}

TYPED_TEST(TransformAndInvert_hermFFT, hermFFT_nD_n_equals_1_xarray) {
  xt::xarray<std::complex<TypeParam>, xt::layout_type::row_major> a = generate_hermitian_data<TypeParam, 1, xt::fftw::hfft, xt::fftw::ihfft>(4);
  auto a_fourier = xt::fftw::hfftn<1>(a);
  // std::cout << "fourier transform of input before ifft (which is destructive!): " << a_fourier << std::endl;
  auto should_be_a = xt::fftw::ihfftn<1>(a_fourier);
  assert_results_complex(a, a_fourier, should_be_a);
}
*/
