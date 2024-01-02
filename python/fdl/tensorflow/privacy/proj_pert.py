#-*- coding:utf-8 -*-

import tensorflow.compat.v1 as tf
import numpy as np
import logging

def projection_perturb_fn(x, y, sum_kl_bound=1.0, iso_proj=True):
    if sum_kl_bound <= 0:
        raise ValueError("Upper bound of sumKL should be positive")
    logging.info(
        f"iso_proj[{iso_proj}], "
        f"sum_kl_bound[{sum_kl_bound}]")
    
    if len(y.shape) == 2 and (y.shape[0] == 1 or y.shape[1] == 1):
        y = tf.reshape(y, [-1])
    elif len(y.shape) != 1:
        raise ValueError(f"Unsupported shape for y: {y.shape}")
    if y.dtype != tf.float32:
        y = tf.cast(y, tf.float32)
    
    x_shape = tf.shape(x)
    pos_indices = tf.where(y)
    neg_indices = tf.where(1 - y)
    pos_x = tf.gather_nd(x, pos_indices)
    neg_x = tf.gather_nd(x, neg_indices)
    pos_mean, pos_cov = tf.nn.moments(pos_x, [0])
    neg_mean, neg_cov = tf.nn.moments(neg_x, [0])
    mean_diff = pos_mean - neg_mean
    pos_std = tf.math.sqrt(pos_cov)
    neg_std = tf.math.sqrt(neg_cov)

    if iso_proj:
        avg_sum_kl_bound = sum_kl_bound / tf.cast(x_shape[1], tf.float32)
        iso_perturbed_cov = tf.math.pow(mean_diff, 2) / avg_sum_kl_bound
        pos_perturbed_cov = neg_perturbed_cov = iso_perturbed_cov
    else:
        try:
            import proj_pert_solver_c
            solve_projection_C = proj_pert_solver_c.SolveProjection
        except:
            logging.warn(
                "Failed to import the proj_pert_solver_c library. "
                "Will use the pythonic version to search for the projection, "
                "which might be slow.")
            solve_projection_C = None
        def _fn(pos_std, neg_std, mean_diff):
            pos_std_np = pos_std.numpy()
            neg_std_np = neg_std.numpy()
            mean_diff_np = mean_diff.numpy()
            if solve_projection_C is not None:
                return solve_projection_C(
                    pos_std_np, neg_std_np, 
                    mean_diff_np, 
                    sum_kl_bound)
            else:
                dim = mean_diff_np.shape[0]
                avg_sum_kl_bound = sum_kl_bound / dim
                pos_perturbed_std_np = np.zeros(dim)
                neg_perturbed_std_np = np.zeros(dim)
                for i in range(dim):
                    pos_perturbed_std_np[i], neg_perturbed_std_np[i] = \
                        solve_projection_1d(
                            pos_std_np[i], neg_std_np[i], 
                            mean_diff_np[i], 
                            avg_sum_kl_bound)
                return pos_perturbed_std_np, neg_perturbed_std_np
        
        pos_perturbed_std, neg_perturbed_std = tf.py_function(
            func=_fn, inp=[pos_std, neg_std, mean_diff], 
            Tout=[pos_std.dtype, neg_std.dtype])
        pos_perturbed_cov = tf.math.pow(pos_perturbed_std, 2)
        neg_perturbed_cov = tf.math.pow(neg_perturbed_std, 2)
    
    pos_noise_cov = tf.math.maximum(pos_perturbed_cov - pos_cov, 0)
    neg_noise_cov = tf.math.maximum(neg_perturbed_cov - neg_cov, 0)
    pos_noise_std = tf.math.sqrt(pos_noise_cov)
    neg_noise_std = tf.math.sqrt(neg_noise_cov)
    pos_noise_mean = tf.zeros([x_shape[1]])
    neg_noise_mean = tf.zeros([x_shape[1]])

    reshaped_y = tf.reshape(y, [-1, 1])
    pos_noise_dist = tf.distributions.Normal(pos_noise_mean, pos_noise_std)
    neg_noise_dist = tf.distributions.Normal(neg_noise_mean, neg_noise_std)
    pos_noise = pos_noise_dist.sample([x_shape[0]]) * reshaped_y
    neg_noise = neg_noise_dist.sample([x_shape[0]]) * (1 - reshaped_y)
    
    perturbed_x = x + pos_noise + neg_noise
    return perturbed_x


def solve_projection_1d(std0, std1, mean_diff, sum_kl_bound):
    if _compute_sum_kl_1d_np(std0, std1, mean_diff) <= sum_kl_bound:
        return std0, std1
    if np.abs(std0 / std1 - 1.0) < 0.1:
        ret = np.abs(mean_diff) / (sum_kl_bound ** 0.5)
        return ret, ret

    x0 = max(std0, std1)
    y0 = min(std0, std1)
    scale = 1.0 / y0
    x0 = x0 * scale
    y0 = y0 * scale
    c1 = np.abs(mean_diff) * scale
    c1_sqr = c1 * c1
    c2 = sum_kl_bound
    c2_plus_one_times_two = 2.0 * (c2 + 1.0)

    def A(x_val, x_sqr_val):
        return 1.0 / x_sqr_val

    def dA_dx(x_val, x_cube_val):
        return -2.0 / x_cube_val

    def B(x_val, c1_sqr_div_x_sqr_val):
        return c1_sqr_div_x_sqr_val - c2_plus_one_times_two

    def dB_dx(x_val, c1_sqr_div_x_sqr_val):
        return -2.0 * c1_sqr_div_x_sqr_val / x_val

    def C(x_val, x_sqr_val):
        return x_sqr_val + c1_sqr

    def dC_dx(x_val):
        return 2.0 * x_val

    def Upper(A_val, B_val, C_val, sqrt_B_sqr_minus_four_A_C_val):
        # -B - sqrt(B^2 - 4AC)
        return -B_val - sqrt_B_sqr_minus_four_A_C_val

    def dUpper_dx(A_val, dA_dx_val, B_val, dB_dx_val, C_val, dC_dx_val, sqrt_B_sqr_minus_four_A_C_val):
        tmp1 = 0.5 / sqrt_B_sqr_minus_four_A_C_val
        tmp2 = 2.0 * B_val * dB_dx_val - 4.0 * (A_val * dC_dx_val + dA_dx_val * C_val)
        return -dB_dx_val - tmp1 * tmp2

    def Lower(A_val):
        return 2.0 * A_val

    def dLower_dx(dA_dx_val):
        return 2.0 * dA_dx_val

    def y(Upper_val, Lower_val):
        # sqrt((-B - sqrt(B^2 - 4AC)) / (2A))
        y_sqr_val = Upper_val / Lower_val
        return y_sqr_val ** 0.5

    def dy_dx(Upper_val, dUpper_dx_val, Lower_val, dLower_dx_val, y_val):
        tmp1 = 0.5 / y_val
        tmp2 = dUpper_dx_val * Lower_val - Upper_val * dLower_dx_val
        tmp2 = tmp2 / (Lower_val ** 2)
        return tmp1 * tmp2

    def compute_y_and_dy_dx(x_val):
        # pre-compute
        x_sqr_val = x_val * x_val
        x_cube_val = x_sqr_val * x_val
        c1_div_x_val = c1 / x_val
        c1_sqr_div_x_sqr_val = c1_div_x_val * c1_div_x_val

        A_val, dA_dx_val = A(x_val, x_sqr_val), dA_dx(x_val, x_cube_val)
        B_val, dB_dx_val = B(x_val, c1_sqr_div_x_sqr_val), dB_dx(x_val, c1_sqr_div_x_sqr_val)
        C_val, dC_dx_val = C(x_val, x_sqr_val), dC_dx(x_val)
        # sqrt(B^2 - 4AC)
        B_sqr_minus_four_A_C_val = B_val * B_val - 4.0 * A_val * C_val
        sqrt_B_sqr_minus_four_A_C_val = B_sqr_minus_four_A_C_val ** 0.5
        
        Upper_val = Upper(A_val, B_val, C_val, sqrt_B_sqr_minus_four_A_C_val)
        dUpper_dx_val = dUpper_dx(A_val, dA_dx_val, B_val, dB_dx_val, C_val, dC_dx_val, sqrt_B_sqr_minus_four_A_C_val)
        Lower_val = Lower(A_val)
        dLower_dx_val = dLower_dx(dA_dx_val)
        
        y_val = y(Upper_val, Lower_val)
        dy_dx_val = dy_dx(Upper_val, dUpper_dx_val, Lower_val, dLower_dx_val, y_val)
        return y_val, dy_dx_val

    x_iso = c1 / (c2 ** 0.5)
    if x0 < x_iso:
        x_l = x_iso
        x_r = c1 * ((0.5 * (1 + (c2 + 1) / (c2 + 2) * ((2 * c2 + 4) ** 0.5)) / c2) ** 0.5)
        x_proj = y_proj = None
        while x_l < x_r - 1e-3:
            x_mid = 0.5 * (x_l + x_r)
            y_mid, dy_dx_mid = compute_y_and_dy_dx(x_mid)
            slope = (y_mid - y0) / (x_mid - x0)
            tmp = slope * dy_dx_mid
            if np.abs(tmp + 1) < 1e-3:
                x_proj = x_mid
                y_proj = y_mid
                break
            elif tmp > -1:
                x_r = x_mid
            else:
                x_l = x_mid
        if x_proj is None:
            x_proj = 0.5 * (x_l + x_r)
            y_proj, _ = compute_y_and_dy_dx(x_proj)
        x_ret = x_proj / scale
        y_ret = y_proj / scale
    else:
        # In practice, this case corner should not happen
        x_ret = x0 / scale
        y_ret = compute_y_and_dy_dx(x0)[0] / scale
    
    if std0 > std1:
        return x_ret, y_ret
    else:
        return y_ret, x_ret


def _compute_sum_kl(std0, std1, mean_diff):
    v1 = tf.math.pow(std0 / std1, 2)
    v2 = tf.math.pow(std1 / std0, 2)
    v3 = tf.math.pow(mean_diff / std0, 2)
    v4 = tf.math.pow(mean_diff / std1, 2)
    return 0.5 * tf.reduce_sum((v1 + v2 - 2.0) + (v3 + v4))


def _compute_sum_kl_1d_np(std0, std1, mean_diff):
    if np.abs(std0 / std1 - 1.0) < 0.1:
        return np.power(mean_diff / std0, 2)
    else:
        v1 = np.power(std0 / std1, 2)
        v2 = np.power(std1 / std0, 2)
        v3 = np.power(mean_diff / std0, 2)
        v4 = np.power(mean_diff / std1, 2)
        return 0.5 * ((v1 + v2 - 2.0) + (v3 + v4))
