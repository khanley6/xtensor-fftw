/*
 * xtensor-fftw
 * Copyright (c) 2017, Patrick Bos
 * Distributed under the terms of the BSD 3-Clause License.
 *
 * The full license is in the file LICENSE, distributed with this software.
 *
 * Modifications:
 * Copyright (c) 2019, Kenneth Hanley
 */

#ifndef XTENSOR_FFTW_HELPER_HPP
#define XTENSOR_FFTW_HELPER_HPP

#define _USE_MATH_DEFINES  // for MSVC ("Math Constants are not defined in Standard C/C++")
#include <cmath>           // M_PI
#include <type_traits>

#include <xtensor/xtensor.hpp>

namespace xt {
  namespace fftw {

    //template <typename T, typename = std::enable_if_t<std::is_floating_point<T>::value>>
    template <class E, typename T = typename std::decay_t<E>::value_type>
    xt::xtensor<T, 1> fftshift(const xt::xexpression<E>& e) {
      // partly mimic np.fftshift (only 1D arrays)
      auto de = xt::eval(e.derived_cast());
      xt::xtensor<T, 1> shifted(de);
      auto it_in = de.begin();
      for (std::size_t ix_shifted = de.size()/2; it_in != de.end(); ++it_in, ++ix_shifted) {
        shifted[ix_shifted % de.size()] = *it_in;
      }
      return shifted;
    }

    template <class E, typename T = typename std::decay_t<E>::value_type>
    xt::xtensor<T, 1> ifftshift(const xt::xexpression<E>& e) {
      auto shifted = e.derived_cast();
      // partly mimic np.ifftshift (only 1D arrays)
      xt::xtensor<T, 1> out = shifted;
      auto it_out = out.begin();
      for (std::size_t ix_shifted = out.size()/2; it_out != out.end(); ++ix_shifted, ++it_out) {
        *it_out = shifted[ix_shifted % out.size()];
      }
      return out;
    }


    template <typename T>
    xt::xtensor<T, 1> fftfreq(unsigned long n, T d=1.0) {
      // mimic np.fftfreq
      T df = 1 / (d * static_cast<T>(n));
      xt::xtensor<T, 1> frequencies;
      if (n % 2 == 0) {
        frequencies = xt::arange<T>(-static_cast<long>(n/2), n/2) * df;
      } else {
        frequencies = xt::arange<T>(-static_cast<long>(n/2), n/2 + 1) * df;
      }
      frequencies = ifftshift(frequencies);
      return frequencies;
    }

    template <typename T>
    xt::xtensor<T, 1> fftscale(unsigned long n, T d=1.0) {
      // mimic np.fftfreq, but in scale space instead of frequency space (dk = 2\pi df)
      return 2 * M_PI * fftfreq(n, d);
    }


    template <typename T>
    xt::xtensor<T, 1> rfftfreq(unsigned long n, T d=1.0) {
      // mimic np.rfftfreq
      T df = 1 / (d * static_cast<T>(n));
      xt::xtensor<T, 1> frequencies = xt::arange<T>(0., n/2 + 1) * df;
      return frequencies;
    }

    template <typename T>
    xt::xtensor<T, 1> rfftscale(unsigned long n, T d=1.0) {
      // mimic np.rfftfreq, but in scale space instead of frequency space (dk = 2\pi df)
      return 2 * M_PI * rfftfreq(n, d);
    }

  }
}

#endif //XTENSOR_FFTW_HELPER_HPP
