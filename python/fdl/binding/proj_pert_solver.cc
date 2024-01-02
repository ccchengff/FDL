#include <tuple>
#include <vector>
#include <cmath>
#include <random>
#include <numeric>
#include <algorithm>
#include <assert.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace fdl {

#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define ABS(x) ((x) < 0 ? -(x) : (x))
#define SQUARE(x) ((x) * (x))

template <typename scalar_t>
inline scalar_t ComputeSumKL1d(const scalar_t& std0, const scalar_t& std1, 
                               const scalar_t& mean_diff) {
  static_assert(std::is_floating_point<scalar_t>::value,
    "Can only be used with floating point types");
  scalar_t ratio = std0 / std1;
  if (ABS(std0 / std1 - 1) < 0.1) {
    return SQUARE(mean_diff / std0);
  } else {
    auto v1 = SQUARE(std0 / std1);
    auto v2 = SQUARE(std1 / std0);
    auto v3 = SQUARE(mean_diff / std0);
    auto v4 = SQUARE(mean_diff / std1);
    return ((v1 + v2 - 2) + (v3 + v4)) / 2;
  }
}

template <typename scalar_t>
std::tuple<scalar_t, scalar_t> 
SolveProjection1d(const scalar_t& std0, const scalar_t& std1, 
                  const scalar_t& mean_diff, 
                  const scalar_t& sum_kl_bound) {
  static_assert(std::is_floating_point<scalar_t>::value,
    "Can only be used with floating point types");
  
  if (ComputeSumKL1d(std0, std1, mean_diff) <= sum_kl_bound) {
    return std::tuple<scalar_t, scalar_t>(std0, std1);
  } else if (ABS(std0 / std1 - 1) < 0.1) {
    auto ret = static_cast<scalar_t>(
      ABS(mean_diff) / std::sqrt(sum_kl_bound));
    return std::tuple<scalar_t, scalar_t>(ret, ret);
  }

  auto x0 = static_cast<scalar_t>(MAX(std0, std1));
  auto y0 = static_cast<scalar_t>(MIN(std0, std1));
  auto scale = static_cast<scalar_t>(1.0 / y0);
  x0 = x0 * scale;
  y0 = y0 * scale;
  auto c1 = static_cast<scalar_t>(ABS(mean_diff) * scale);
  auto c1_sqr = static_cast<scalar_t>(SQUARE(c1));
  auto c2 = static_cast<scalar_t>(sum_kl_bound);
  auto c2_plus_one_times_two = static_cast<scalar_t>(2.0 * (c2 + 1.0));

  auto compute_y_and_dy_dx = [&](scalar_t x_val) {
    // pre-compute
    auto x_sqr_val = x_val * x_val;
    auto x_cube_val = x_sqr_val * x_val;
    auto c1_div_x_val = c1 / x_val;
    auto c1_sqr_div_x_sqr_val = c1_div_x_val * c1_div_x_val;

    // A and dA_dx
    auto A_val = 1 / x_sqr_val;
    auto dA_dx_val = -2 / x_cube_val;
    // B and dB_dx
    auto B_val = c1_sqr_div_x_sqr_val - c2_plus_one_times_two;
    auto dB_dx_val = -2 * c1_sqr_div_x_sqr_val / x_val;
    // C and dC_dx
    auto C_val = x_sqr_val + c1_sqr;
    auto dC_dx_val = 2 * x_val;
    // sqrt(B^2 - 4AC)
    auto B_sqr_minus_four_A_C_val = B_val * B_val - 4 * A_val * C_val;
    auto sqrt_B_sqr_minus_four_A_C_val = std::sqrt(B_sqr_minus_four_A_C_val);
    // Upper (-B - sqrt(B^2 - 4AC)) and Lower (2*A)
    auto Upper_val = -B_val - sqrt_B_sqr_minus_four_A_C_val;
    auto dUpper_dx_val = -dB_dx_val - \
      (1 / (2 * sqrt_B_sqr_minus_four_A_C_val)) * \
      (2 * B_val * dB_dx_val - 4 * (A_val * dC_dx_val + dA_dx_val * C_val));
    auto Lower_val = 2 * A_val;
    auto dLower_dx_val = 2 * dA_dx_val;
    
    auto y_val = std::sqrt(Upper_val / Lower_val);
    auto tmp1 = 0.5 / y_val;
    auto tmp2 = dUpper_dx_val * Lower_val - Upper_val * dLower_dx_val;
    tmp2 = tmp2 / (Lower_val * Lower_val);
    auto dy_dx_val = tmp1 * tmp2;
    return std::tuple<scalar_t, scalar_t>(y_val, dy_dx_val);
  };

  scalar_t x_ret, y_ret;
  scalar_t eps = 1e-3;
  auto x_iso = c1 / std::sqrt(c2);
  if (x0 < x_iso) {
    // binary search
    auto x_l = x_iso;
    auto x_r = x_iso * static_cast<scalar_t>(std::sqrt(
      0.5 + (c2 + 1) / std::sqrt(2 * (c2 + 2))));
    scalar_t x_proj, y_proj;
    bool found = false;
    while (x_l < x_r - eps) {
      auto x_mid = (x_l + x_r) / 2;
      auto y_and_dy_dx_mid = compute_y_and_dy_dx(x_mid);
      auto y_mid = std::get<0>(y_and_dy_dx_mid);
      auto dy_dx_mid = std::get<1>(y_and_dy_dx_mid);
      auto slope = (y_mid - y0) / (x_mid - x0);
      auto tmp = slope * dy_dx_mid;
      if (tmp < -1 - eps) {
        x_l = x_mid;
      } else if (tmp > -1 + eps) {
        x_r = x_mid;
      } else {
        x_proj = x_mid;
        y_proj = y_mid;
        found = true;
        break;
      }
    }
    if (!found) {
      x_proj = (x_l + x_r) / 2;
      y_proj = std::get<0>(compute_y_and_dy_dx(x_proj));
    }
    x_ret = x_proj / scale;
    y_ret = y_proj / scale;
  } else {
    // In practice, this case corner would not happen
    x_ret = x0 / scale;
    y_ret = std::get<0>(compute_y_and_dy_dx(x0)) / scale;
  }

  if (std0 > std1) {
    return std::tuple<scalar_t, scalar_t>(x_ret, y_ret);
  } else {
    return std::tuple<scalar_t, scalar_t>(y_ret, x_ret);
  }
}

template <typename scalar_t>
std::tuple<std::vector<scalar_t>, std::vector<scalar_t>> 
SolveProjection(const std::vector<scalar_t>& std0, 
                const std::vector<scalar_t>& std1, 
                const std::vector<scalar_t>& mean_diff, 
                scalar_t sum_kl_bound) {
  static_assert(std::is_floating_point<scalar_t>::value,
    "Can only be used with floating point types");
  size_t dim = mean_diff.size();
  std::vector<scalar_t> proj_std0;
  std::vector<scalar_t> proj_std1;
  proj_std0.resize(dim);
  proj_std1.resize(dim);
  #pragma omp parallel for schedule(static)
  for (size_t i = 0; i < dim; i++) {
    auto proj = SolveProjection1d(std0[i], std1[i], mean_diff[i], 
      sum_kl_bound / dim);
    proj_std0[i] = std::get<0>(proj);
    proj_std1[i] = std::get<1>(proj);
  }
  return std::tuple<std::vector<scalar_t>, std::vector<scalar_t>>(
    proj_std0, proj_std1);
}
} // namespace fdl


namespace py = pybind11;

template <typename IN, typename OUT>
inline py::array_t<OUT> ToPyArray(std::vector<IN>& vec) {
  auto result = py::array_t<OUT>(vec.size());
  py::buffer_info res_buff = result.request();
  auto* ptr = reinterpret_cast<OUT*>(res_buff.ptr);
  std::copy(vec.begin(), vec.end(), ptr);
  return result;
}

template <typename IN, typename OUT>
inline std::vector<OUT> FromPyArray(py::array_t<IN> arr) {
  py::buffer_info arr_buf = arr.request();
  assert(arr_buf.ndim == 1 && 
    "Currently we only support one-dimensional arrays");
  auto arr_ptr = reinterpret_cast<IN*>(arr_buf.ptr);
  std::vector<OUT> result(arr_ptr, arr_ptr + arr_buf.shape[0]);
  return std::move(result);
}

typedef double value_t; // for numerical stability

template <typename scalar_t>
std::tuple<py::array_t<scalar_t>, py::array_t<scalar_t>> 
SolveProjectionPy(py::array_t<scalar_t> std0, py::array_t<scalar_t> std1, 
                  py::array_t<scalar_t> mean_diff, scalar_t sum_kl_bound) {
  auto std0_vec = FromPyArray<scalar_t, value_t>(std0);
  auto std1_vec = FromPyArray<scalar_t, value_t>(std1);
  auto mean_diff_vec = FromPyArray<scalar_t, value_t>(mean_diff);
  auto proj = fdl::SolveProjection<value_t>(std0_vec, std1_vec, 
    mean_diff_vec, static_cast<value_t>(sum_kl_bound));
  auto proj_std0 = ToPyArray<value_t, scalar_t>(std::get<0>(proj));
  auto proj_std1 = ToPyArray<value_t, scalar_t>(std::get<1>(proj));
  return std::tuple<py::array_t<scalar_t>, py::array_t<scalar_t>>(
    proj_std0, proj_std1);
}

PYBIND11_MODULE(proj_pert_c, m) {
  m.def("SolveProjection", &(SolveProjectionPy<value_t>), "Solve projection");
}
