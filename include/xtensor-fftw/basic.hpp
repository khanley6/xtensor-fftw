#ifndef XTENSOR_FFTW_BASIC_HPP
#define XTENSOR_FFTW_BASIC_HPP

#include <xtensor/xtensor.hpp>
#include "xtensor/xcomplex.hpp"
#include "xtensor/xeval.hpp"
#include <xtl/xcomplex.hpp>
#include <complex>
#include <tuple>
#include <type_traits>
#include <exception>
#include <mutex>

// for product accumulate:
#include <numeric>
#include <functional>

#include <fftw3.h>

#ifdef __CLING__
#pragma cling load("fftw3")
#endif

namespace xt
{
    namespace fftw
    {
        namespace detail
        {
            // FFTW is not thread-safe, so we need to guard around its functions (except fftw_execute).
            inline std::mutex& fftw_global_mutex() {
                static std::mutex m;
                return m;
            }

            enum class dir {
                forward  = FFTW_FORWARD,
                backward = FFTW_BACKWARD,
                nil      = 0
            };

            template <typename T>
            struct is_floating_point : std::is_floating_point<T> {};
            template <typename T>
            struct is_floating_point<std::complex<T>> : std::is_floating_point<T> {};
            //template <typename T>
            //inline constexpr bool is_floating_point_v = is_floating_point<T>::value;

            // input vs output shape conversion
            template <class E>
            inline auto output_shape_from_input(const xexpression<E>& e, bool half_plus_one_out, bool half_plus_one_in, bool odd_last_dim = false) {
                auto de = e.derived_cast();
                auto output_shape = de.shape();
                if (half_plus_one_out) {        // r2c
                    auto n = output_shape.size();
                    output_shape[n-1] = output_shape[n-1]/2 + 1;
                } else if (half_plus_one_in) {  // c2r
                    auto n = output_shape.size();
                    if (!odd_last_dim) {
                        output_shape[n - 1] = (output_shape[n - 1] - 1) * 2;
                    } else {
                        output_shape[n - 1] = (output_shape[n - 1] - 1) * 2 + 1;
                    }
                }
                return output_shape;
            }

            // aliases for the fftw precision-dependent types:
            template <typename T> struct fftw_t {
                static_assert(sizeof(T) == 0, "Only specializations of fftw_t can be used");
            };
            template <> struct fftw_t<float> {
                using plan = fftwf_plan;
                using complex = fftwf_complex;
                constexpr static void (&execute)(plan) = fftwf_execute;
                constexpr static void (&destroy_plan)(plan) = fftwf_destroy_plan;
            };
            template <> struct fftw_t<double> {
                using plan = fftw_plan;
                using complex = fftw_complex;
                constexpr static void (&execute)(plan) = fftw_execute;
                constexpr static void (&destroy_plan)(plan) = fftw_destroy_plan;
            };
            template <> struct fftw_t<long double> {
                using plan = fftwl_plan;
                using complex = fftwl_complex;
                constexpr static void (&execute)(plan) = fftwl_execute;
                constexpr static void (&destroy_plan)(plan) = fftwl_destroy_plan;
            };
            // and subclass alias for when calling with a complex type:
            template <typename T> struct fftw_t<std::complex<T>> : public fftw_t<T> {};

            // convert std::complex to fftwX_complex with right precision X; non-complex floats stay themselves:
            template <typename T>
            using fftw_number_t = std::conditional_t<xtl::is_complex<T>::value,
                typename fftw_t<xtl::complex_value_type_t<T>>::complex,
                xtl::complex_value_type_t<T>
                >;

            // dimension-dependent function signatures of fftw planning functions
            template <typename in_t, typename out_t, std::size_t dim, dir fftw_direction, bool fftw_123dim>
            struct fftw_plan_dft_signature {};

            template <typename in_t, typename out_t, std::size_t dim>
            struct fftw_plan_dft_signature<in_t, out_t, dim, dir::nil, false> {
                using type = typename fftw_t<in_t>::plan (&)(int rank, const int *n, fftw_number_t<in_t> *, fftw_number_t<out_t> *, unsigned int);
            };
            template <typename in_t, typename out_t>
            struct fftw_plan_dft_signature<in_t, out_t, 1, dir::nil, true> {
                using type = typename fftw_t<in_t>::plan (&)(int n1, fftw_number_t<in_t> *, fftw_number_t<out_t> *, unsigned int);
            };
            template <typename in_t, typename out_t>
            struct fftw_plan_dft_signature<in_t, out_t, 2, dir::nil, true> {
                using type = typename fftw_t<in_t>::plan (&)(int n1, int n2, fftw_number_t<in_t> *, fftw_number_t<out_t> *, unsigned int);
            };
            template <typename in_t, typename out_t>
            struct fftw_plan_dft_signature<in_t, out_t, 3, dir::nil, true> {
                using type = typename fftw_t<in_t>::plan (&)(int n1, int n2, int n3, fftw_number_t<in_t> *, fftw_number_t<out_t> *, unsigned int);
            };

            template <typename in_t, typename out_t, std::size_t dim, dir fftw_direction>
            struct fftw_plan_dft_signature<in_t, out_t, dim, fftw_direction, false> {
                using type = typename fftw_t<in_t>::plan (&)(int rank, const int *n, fftw_number_t<in_t> *, fftw_number_t<out_t> *, int, unsigned int);
            };
            template <typename in_t, typename out_t, dir fftw_direction>
            struct fftw_plan_dft_signature<in_t, out_t, 1, fftw_direction, true> {
                using type = typename fftw_t<in_t>::plan (&)(int n1, fftw_number_t<in_t> *, fftw_number_t<out_t> *, int, unsigned int);
            };
            template <typename in_t, typename out_t, dir fftw_direction>
            struct fftw_plan_dft_signature<in_t, out_t, 2, fftw_direction, true> {
                using type = typename fftw_t<in_t>::plan (&)(int n1, int n2, fftw_number_t<in_t> *, fftw_number_t<out_t> *, int, unsigned int);
            };
            template <typename in_t, typename out_t, dir fftw_direction>
            struct fftw_plan_dft_signature<in_t, out_t, 3, fftw_direction, true> {
                using type = typename fftw_t<in_t>::plan (&)(int n1, int n2, int n3, fftw_number_t<in_t> *, fftw_number_t<out_t> *, int, unsigned int);
            };


            // output to DFT-dimensions conversion
            template <class E>
            inline auto dft_dims_from_output(const xexpression<E>& e, bool half_plus_one_out, bool odd_last_dim = false) {
                auto de = e.derived_cast();
                auto dft_dims = de.shape();
                if (half_plus_one_out) {        // r2c
                    auto n = dft_dims.size();
                    if (!odd_last_dim) {
                        dft_dims[n - 1] = (dft_dims[n - 1] - 1) * 2;
                    } else {
                        dft_dims[n - 1] = (dft_dims[n - 1] - 1) * 2 + 1;
                    }
                }
                return dft_dims;
            }


            template <std::size_t Dim, dir fftw_direction, class E1, class E2,
                      typename T1 = typename std::decay_t<E1>::value_type,
                      typename T2 = typename std::decay_t<E2>::value_type>
                      //typename fftw_plan_dft_signature<T1, T2, Dim, fftw_direction, true>::type fftw_plan_dft>
            auto fftw_plan_factory(const xexpression<E1>& input_expr, xexpression<E2>& output_expr, bool half_plus_one_out) {
                using in_value_type = typename std::decay_t<E1>::value_type;
                using out_value_type = typename std::decay_t<E2>::value_type;
                using fftw_input_t = fftw_number_t<in_value_type>;
                using fftw_output_t = fftw_number_t<out_value_type>;

                auto input = input_expr.derived_cast();
                auto output = output_expr.derived_cast();

                auto dft_dimensions_unsigned = dft_dims_from_output(output, half_plus_one_out);

                using plan_t = typename fftw_t<T1>::plan (&)(int n1, fftw_number_t<T1> *, fftw_number_t<T2> *, int, unsigned int);

                std::lock_guard<std::mutex> guard(detail::fftw_global_mutex());
                //return fftw_plan_dft(static_cast<int>(dft_dimensions_unsigned[0]),
                return fftw_plan_dft_1d(static_cast<int>(dft_dimensions_unsigned[0]),
                                     const_cast<fftw_input_t *>(reinterpret_cast<const fftw_input_t *>(input.data())),
                                     reinterpret_cast<fftw_output_t *>(output.data()),
                                     static_cast<int>(fftw_direction),
                                     FFTW_ESTIMATE);
            }

            template <std::size_t Dim, class E, typename T = typename std::decay_t<E>::value_type>
            inline auto fft_impl(const xexpression<E>& input_expr, bool half_plus_one_out, bool half_plus_one_in) {
                auto input = input_expr.derived_cast();
                auto output_shape = output_shape_from_input(input, half_plus_one_out, half_plus_one_in, false);
                //xt::xarray<T> output(output_shape);
                xt::xtensor<T, Dim> output(output_shape);

                auto ax = normalize_axis(input.dimension(), -1);
                bool odd_last_dim = (input.shape(ax) % 2 != 0);

                //auto plan = nullptr;
                auto plan = fftw_plan_factory<Dim, dir::forward>(input, output, half_plus_one_out);
                if (plan == nullptr) {
                    throw std::runtime_error("Plan creation returned nullptr.");
                }

                fftw_execute(plan);
                {
                    std::lock_guard<std::mutex> guard(detail::fftw_global_mutex());
                    fftw_destroy_plan(plan);
                }
                return output;
            }

            inline auto ifft_impl() {
                return 0;
            }
        } // namespace detail

        ////
        // Regular FFT: 1D
        ////
        template <std::size_t Dim = 1, class E,
                  typename T = typename std::decay_t<E>::value_type,
                  typename = std::enable_if_t<detail::is_floating_point<T>::value>>
            inline auto fft(const xexpression<E> &e) {
                auto de = eval(e.derived_cast());
                return detail::fft_impl<Dim>(de, false, false);
            }

        template <std::size_t Dim = 1, class E,
                  typename T = typename std::decay_t<E>::value_type,
                  typename = std::enable_if_t<detail::is_floating_point<T>::value>>
            inline auto ifft(const xt::xexpression<E> &e) {
                auto de = xt::eval(e.derived_cast());
            }

    } // namespace fftw
} // namespace xt

#endif // XTENSOR_FFTW_BASIC_HPP
