/*
 * Copyright (c) 2018 Arm Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include "OpticalFlow.h"

#include "GaussianPyramidHalf.h"
#include "Scharr.h"
#include "Utils.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace reference
{
namespace
{
using KeyPointArray         = std::vector<KeyPoint>;
using InternalKeyPointArray = std::vector<InternalKeyPoint>;

// Constants used for Lucas-Kanade Algorithm
constexpr int   W_BITS                = 14;
constexpr float D0                    = 1 << W_BITS;
constexpr float DETERMINANT_THRESHOLD = 1.0e-07f;
constexpr float EIGENVALUE_THRESHOLD  = 1.0e-04f;
constexpr float FLT_SCALE             = 1.0f / (1 << 20);

// Creates an InternalKeyPointArray for tracking non-integral pixel coordinates
InternalKeyPointArray create_internal_keypoints(const KeyPointArray &keypoints)
{
    InternalKeyPointArray internal_keypoints;

    for(auto keypoint : keypoints)
    {
        InternalKeyPoint internal_keypoint;

        internal_keypoint.x               = static_cast<float>(keypoint.x);
        internal_keypoint.y               = static_cast<float>(keypoint.y);
        internal_keypoint.tracking_status = static_cast<bool>(keypoint.tracking_status);

        internal_keypoints.push_back(internal_keypoint);
    }

    return internal_keypoints;
}

// Scale tracked points based on Pyramid level
void scale_tracked_points(size_t level, size_t num_levels, bool use_initial_estimate,
                          InternalKeyPointArray &old_points_internal, InternalKeyPointArray &new_points_internal,
                          const KeyPointArray &old_points, const KeyPointArray &new_points_estimates)
{
    if(level == num_levels - 1) // lowest resolution
    {
        const float scale = std::pow(SCALE_PYRAMID_HALF, level);

        for(size_t i = 0; i < old_points.size(); ++i)
        {
            old_points_internal.at(i).x               = old_points.at(i).x * scale;
            old_points_internal.at(i).y               = old_points.at(i).y * scale;
            old_points_internal.at(i).tracking_status = true;

            InternalKeyPoint keypoint_to_track;

            if(use_initial_estimate)
            {
                keypoint_to_track.x               = new_points_estimates.at(i).x * scale;
                keypoint_to_track.y               = new_points_estimates.at(i).y * scale;
                keypoint_to_track.tracking_status = (new_points_estimates.at(i).tracking_status == 1);
            }
            else
            {
                keypoint_to_track.x               = old_points_internal.at(i).x;
                keypoint_to_track.y               = old_points_internal.at(i).y;
                keypoint_to_track.tracking_status = true;
            }

            new_points_internal.at(i) = keypoint_to_track;
        }
    }
    else
    {
        for(size_t i = 0; i < old_points.size(); ++i)
        {
            old_points_internal.at(i).x /= SCALE_PYRAMID_HALF;
            old_points_internal.at(i).y /= SCALE_PYRAMID_HALF;
            new_points_internal.at(i).x /= SCALE_PYRAMID_HALF;
            new_points_internal.at(i).y /= SCALE_PYRAMID_HALF;
        }
    }
}

bool is_invalid_keypoint(const InternalKeyPoint &keypoint, const ValidRegion &valid_region, size_t window_dimension)
{
    const int half_window = window_dimension / 2;
    const int x           = std::floor(keypoint.x);
    const int y           = std::floor(keypoint.y);

    return (x - half_window < valid_region.start(0)) || (x + half_window >= valid_region.end(0) - 1) || (y - half_window < valid_region.start(1)) || (y + half_window >= valid_region.end(1) - 1);
}

template <typename T>
constexpr int INT_ROUND(T x, int n)
{
    return (x + (1 << (n - 1))) >> n;
}

// Return the bilinear value at a specified coordinate with different border modes
template <typename T>
int bilinear_interpolate(const SimpleTensor<T> &in, Coordinates id, float wx, float wy, BorderMode border_mode, T constant_border_value, int scale)
{
    const int level = id.x();
    const int idy   = id.y();

    const float dx   = wx;
    const float dy   = wy;
    const float dx_1 = 1.0f - dx;
    const float dy_1 = 1.0f - dy;

    const T border_value = constant_border_value;

    id.set(0, level);
    id.set(1, idy);
    const T tl = tensor_elem_at(in, id, border_mode, border_value);
    id.set(0, level + 1);
    id.set(1, idy);
    const T tr = tensor_elem_at(in, id, border_mode, border_value);
    id.set(0, level);
    id.set(1, idy + 1);
    const T bl = tensor_elem_at(in, id, border_mode, border_value);
    id.set(0, level + 1);
    id.set(1, idy + 1);
    const T br = tensor_elem_at(in, id, border_mode, border_value);

    // weights
    const int w00 = roundf(dx_1 * dy_1 * D0);
    const int w01 = roundf(dx * dy_1 * D0);
    const int w10 = roundf(dx_1 * dy * D0);
    const int w11 = D0 - w00 - w01 - w10;

    return static_cast<int>(INT_ROUND(tl * w00 + tr * w01 + bl * w10 + br * w11, scale));
}

template <typename T>
std::vector<int> compute_derivative(const SimpleTensor<T> &input, const InternalKeyPoint &keypoint,
                                    BorderMode border_mode, uint8_t constant_border_value, size_t window_dimension, int scale)
{
    std::vector<int> bilinear_values;

    const int half_window = window_dimension / 2;

    float keypoint_int_x = 0;
    float keypoint_int_y = 0;

    const float wx = std::modf(keypoint.x, &keypoint_int_x);
    const float wy = std::modf(keypoint.y, &keypoint_int_y);

    Coordinates tl_window(static_cast<int>(keypoint_int_x) - half_window, static_cast<int>(keypoint_int_y) - half_window);
    Coordinates br_window(static_cast<int>(keypoint_int_x) + half_window, static_cast<int>(keypoint_int_y) + half_window);

    for(int y = tl_window.y(); y <= br_window.y(); ++y)
    {
        for(int x = tl_window.x(); x <= br_window.x(); ++x)
        {
            bilinear_values.push_back(bilinear_interpolate(input, Coordinates(x, y), wx, wy, border_mode, static_cast<T>(constant_border_value), scale));
        }
    }

    return bilinear_values;
}

std::tuple<float, float, float> compute_spatial_gradient_matrix(const std::vector<int> &bilinear_ix, const std::vector<int> &bilinear_iy)
{
    ARM_COMPUTE_ERROR_ON(bilinear_ix.size() != bilinear_iy.size());

    int iA11 = 0;
    int iA12 = 0;
    int iA22 = 0;

    for(size_t i = 0; i < bilinear_ix.size(); ++i)
    {
        int ixval = bilinear_ix[i];
        int iyval = bilinear_iy[i];

        iA11 += ixval * ixval;
        iA12 += ixval * iyval;
        iA22 += iyval * iyval;
    }

    return std::make_tuple(iA11 * FLT_SCALE, iA12 * FLT_SCALE, iA22 * FLT_SCALE);
}

std::tuple<double, double> compute_temporal_gradient_vector(const std::vector<int> &bilinear_it_old,
                                                            const std::vector<int> &bilinear_it_new,
                                                            const std::vector<int> &bilinear_ix,
                                                            const std::vector<int> &bilinear_iy)
{
    ARM_COMPUTE_ERROR_ON(bilinear_ix.size() != bilinear_iy.size());
    ARM_COMPUTE_ERROR_ON(bilinear_it_old.size() != bilinear_it_new.size());

    int ib1 = 0;
    int ib2 = 0;

    for(size_t i = 0; i < bilinear_ix.size(); ++i)
    {
        int ixval = bilinear_ix[i];
        int iyval = bilinear_iy[i];
        int ival  = bilinear_it_old[i];
        int jval  = bilinear_it_new[i];

        const int diff = jval - ival;

        ib1 += diff * ixval;
        ib2 += diff * iyval;
    }

    const double b1 = ib1 * FLT_SCALE;
    const double b2 = ib2 * FLT_SCALE;

    return std::make_tuple(b1, b2);
}
} // namespace

template <typename T>
std::vector<KeyPoint> optical_flow(const SimpleTensor<T> &old_input, const SimpleTensor<T> &new_input,
                                   const OpticalFlowParameters &params, size_t num_levels,
                                   const std::vector<KeyPoint> &old_points, const std::vector<KeyPoint> &new_points_estimates,
                                   BorderMode border_mode, uint8_t constant_border_value)
{
    const int    filter_size      = 3;    // scharr filter size
    const size_t max_iterations   = 1000; // fixed by kernel
    const size_t window_dimension = params.window_dimension;
    const size_t num_iterations   = (params.termination == Termination::TERM_CRITERIA_EPSILON) ? max_iterations : params.num_iterations;

    KeyPointArray new_points(old_points.size());

    InternalKeyPointArray old_points_internal = create_internal_keypoints(old_points);
    InternalKeyPointArray new_points_internal = create_internal_keypoints(new_points_estimates);

    SimpleTensor<int16_t> scharr_gx;
    SimpleTensor<int16_t> scharr_gy;

    // Create pyramids
    std::vector<SimpleTensor<T>> old_pyramid = gaussian_pyramid_half(old_input, border_mode, constant_border_value, num_levels);
    std::vector<SimpleTensor<T>> new_pyramid = gaussian_pyramid_half(new_input, border_mode, constant_border_value, num_levels);

    // Iterate over each level of the pyramid
    for(size_t idx = num_levels; idx > 0; --idx)
    {
        const size_t level = idx - 1;

        // Calculate scharr gradients
        std::tie(scharr_gx, scharr_gy) = scharr<int16_t, T>(old_pyramid[level], filter_size, border_mode, constant_border_value, GradientDimension::GRAD_XY);

        scale_tracked_points(level, num_levels, params.use_initial_estimate, old_points_internal, new_points_internal, old_points, new_points_estimates);

        // Calculate valid region based on image dimensions of current pyramid level
        const ValidRegion valid_region = shape_to_valid_region(old_pyramid[level].shape(), (border_mode == BorderMode::UNDEFINED), BorderSize(filter_size / 2));

        for(size_t i = 0; i < old_points.size(); ++i)
        {
            InternalKeyPoint &old_keypoint = old_points_internal.at(i);
            InternalKeyPoint &new_keypoint = new_points_internal.at(i);

            // Helper function for untracking keypoints when on the lowest pyramid level (high resolution)
            const auto untrack_keypoint = [&](bool predicate)
            {
                if(predicate && (level == 0))
                {
                    new_keypoint.tracking_status = false;
                    return true;
                }
                return predicate;
            };

            if(!old_keypoint.tracking_status)
            {
                continue;
            }

            // Check if tracked coordinate is outside image coordinate
            if(untrack_keypoint(is_invalid_keypoint(old_keypoint, valid_region, window_dimension)))
            {
                continue;
            }

            // Compute spatial derivative
            std::vector<int> bilinear_ix = compute_derivative(scharr_gx, old_keypoint, border_mode, constant_border_value, window_dimension, W_BITS);
            std::vector<int> bilinear_iy = compute_derivative(scharr_gy, old_keypoint, border_mode, constant_border_value, window_dimension, W_BITS);

            float A11 = 0.f;
            float A12 = 0.f;
            float A22 = 0.f;
            std::tie(A11, A12, A22) = compute_spatial_gradient_matrix(bilinear_ix, bilinear_iy);

            // Calculate criteria for lost tracking : Matrix A is invertible
            // 1. The determinant of the matrix is less than DETERMINANT_THRESHOLD
            // 2. The minimum eigenvalue of the matrix is less than EIGENVALUE_THRESHOLD
            const float trace_A      = A11 + A22;
            const float determinant  = A11 * A22 - A12 * A12;
            const float discriminant = (trace_A * trace_A) - 4.0f * (determinant);
            const float eigenvalue_A = (trace_A - std::sqrt(discriminant)) / 2.0f;

            // Divide by window_dimension squared to reduce the floating point accummulation error
            const float eigenvalue = eigenvalue_A / (window_dimension * window_dimension);

            // Check if it is a good point to track
            if(untrack_keypoint(eigenvalue < EIGENVALUE_THRESHOLD || determinant < DETERMINANT_THRESHOLD))
            {
                continue;
            }

            float prev_delta_x = 0.f;
            float prev_delta_y = 0.f;

            for(size_t j = 0; j < num_iterations; ++j)
            {
                // Check if tracked coordinate is outside image coordinate
                if(untrack_keypoint(is_invalid_keypoint(new_keypoint, valid_region, window_dimension)))
                {
                    break;
                }

                // Compute temporal derivative
                std::vector<int> bilinear_it_old = compute_derivative(old_pyramid[level], old_keypoint, border_mode, constant_border_value, window_dimension, W_BITS - 5);
                std::vector<int> bilinear_it_new = compute_derivative(new_pyramid[level], new_keypoint, border_mode, constant_border_value, window_dimension, W_BITS - 5);

                double b1 = 0.f;
                double b2 = 0.f;
                std::tie(b1, b2) = compute_temporal_gradient_vector(bilinear_it_old, bilinear_it_new, bilinear_ix, bilinear_iy);

                // Compute motion vector -> A^-1 * -b
                const float delta_x = (A12 * b2 - A22 * b1) / determinant;
                const float delta_y = (A12 * b1 - A11 * b2) / determinant;

                // Update the new position
                new_keypoint.x += delta_x;
                new_keypoint.y += delta_y;

                const float magnitude_squared = delta_x * delta_x + delta_y * delta_y;

                // Check if termination criteria is EPSILON and if it is satisfied
                if(magnitude_squared <= params.epsilon && (params.termination == Termination::TERM_CRITERIA_EPSILON || params.termination == Termination::TERM_CRITERIA_BOTH))
                {
                    break;
                }

                // Check convergence analyzing the previous delta
                if(j > 0 && (std::fabs(delta_x + prev_delta_x) < 0.01f && std::fabs(delta_y + prev_delta_y) < 0.01f))
                {
                    new_keypoint.x -= delta_x * SCALE_PYRAMID_HALF;
                    new_keypoint.y -= delta_y * SCALE_PYRAMID_HALF;

                    break;
                }

                prev_delta_x = delta_x;
                prev_delta_y = delta_y;
            }
        }
    }

    // Copy optical flow coordinates to output vector
    for(size_t i = 0; i < old_points.size(); ++i)
    {
        const InternalKeyPoint &new_keypoint = new_points_internal.at(i);

        new_points.at(i).x               = roundf(new_keypoint.x);
        new_points.at(i).y               = roundf(new_keypoint.y);
        new_points.at(i).tracking_status = new_keypoint.tracking_status ? 1 : 0;
    }

    return new_points;
}

template std::vector<KeyPoint> optical_flow(const SimpleTensor<uint8_t> &old_input, const SimpleTensor<uint8_t> &new_input,
                                            const OpticalFlowParameters &params, size_t num_levels,
                                            const std::vector<KeyPoint> &old_points, const std::vector<KeyPoint> &new_points_estimates,
                                            BorderMode border_mode, uint8_t constant_border_value);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
