/*
 * Copyright (c) 2016, 2017 ARM Limited.
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
#include "arm_compute/core/NEON/kernels/NELKTrackerKernel.h"

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/Coordinates.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include <arm_neon.h>
#include <cmath>

using namespace arm_compute;

/** Constants used for Lucas-Kanade Algorithm */
constexpr int   W_BITS                = 14;
constexpr float D0                    = 1 << W_BITS;
constexpr float DETERMINANT_THRESHOLD = 1.0e-07f; // Threshold for the determinant. Used for lost tracking criteria
constexpr float EIGENVALUE_THRESHOLD  = 1.0e-04f; // Thresholds for minimum eigenvalue. Used for lost tracking criteria
constexpr float FLT_SCALE             = 1.0f / (1 << 20);

namespace
{
enum class BilinearInterpolation
{
    BILINEAR_OLD_NEW,
    BILINEAR_SCHARR
};

template <typename T>
constexpr int INT_ROUND(T x, int n)
{
    return (x + (1 << (n - 1))) >> n;
}

template <typename T>
inline int get_pixel(const ITensor *tensor, int xi, int yi, int iw00, int iw01, int iw10, int iw11, int scale)
{
    const auto px00 = *reinterpret_cast<const T *>(tensor->buffer() + tensor->info()->offset_element_in_bytes(Coordinates(xi, yi)));
    const auto px01 = *reinterpret_cast<const T *>(tensor->buffer() + tensor->info()->offset_element_in_bytes(Coordinates(xi + 1, yi)));
    const auto px10 = *reinterpret_cast<const T *>(tensor->buffer() + tensor->info()->offset_element_in_bytes(Coordinates(xi, yi + 1)));
    const auto px11 = *reinterpret_cast<const T *>(tensor->buffer() + tensor->info()->offset_element_in_bytes(Coordinates(xi + 1, yi + 1)));

    return INT_ROUND(px00 * iw00 + px01 * iw01 + px10 * iw10 + px11 * iw11, scale);
}

inline int32x4_t compute_bilinear_interpolation(int16x8_t top_row, int16x8_t bottom_row, int16x4_t w00, int16x4_t w01, int16x4_t w10, int16x4_t w11, int32x4_t shift)
{
    // Get the left column of upper row
    const int16x4_t px00 = vget_low_s16(top_row);

    // Get the right column of upper row
    const int16x4_t px01 = vext_s16(px00, vget_high_s16(top_row), 1);

    // Get the left column of lower row
    const int16x4_t px10 = vget_low_s16(bottom_row);

    // Get the right column of right row
    const int16x4_t px11 = vext_s16(px10, vget_high_s16(bottom_row), 1);

    // Apply the bilinear filter
    return vqrshlq_s32(vmull_s16(px00, w00) + vmull_s16(px01, w01) + vmull_s16(px10, w10) + vmull_s16(px11, w11), shift);
}
} // namespace

void NELKTrackerKernel::init_keypoints(int start, int end)
{
    if(_level == _num_levels - 1)
    {
        const float level_scale = pow(_pyramid_scale, _level);

        for(int i = start; i < end; ++i)
        {
            _old_points_internal->at(i).x               = _old_points->at(i).x * level_scale;
            _old_points_internal->at(i).y               = _old_points->at(i).y * level_scale;
            _old_points_internal->at(i).tracking_status = true;

            NELKInternalKeypoint keypoint_to_track;

            if(_use_initial_estimate)
            {
                keypoint_to_track.x               = _new_points_estimates->at(i).x * level_scale;
                keypoint_to_track.y               = _new_points_estimates->at(i).y * level_scale;
                keypoint_to_track.tracking_status = (_new_points_estimates->at(i).tracking_status == 1);
            }
            else
            {
                keypoint_to_track.x               = _old_points_internal->at(i).x;
                keypoint_to_track.y               = _old_points_internal->at(i).y;
                keypoint_to_track.tracking_status = true;
            }

            _new_points_internal->at(i) = keypoint_to_track;
        }
    }
    else
    {
        for(int i = start; i < end; ++i)
        {
            _old_points_internal->at(i).x /= _pyramid_scale;
            _old_points_internal->at(i).y /= _pyramid_scale;
            _new_points_internal->at(i).x /= _pyramid_scale;
            _new_points_internal->at(i).y /= _pyramid_scale;
        }
    }
}

std::tuple<int, int, int> NELKTrackerKernel::compute_spatial_gradient_matrix(const NELKInternalKeypoint &keypoint, int32_t *bilinear_ix, int32_t *bilinear_iy)
{
    int iA11 = 0;
    int iA12 = 0;
    int iA22 = 0;

    int32x4_t nA11 = vdupq_n_s32(0);
    int32x4_t nA12 = vdupq_n_s32(0);
    int32x4_t nA22 = vdupq_n_s32(0);

    float keypoint_int_x = 0;
    float keypoint_int_y = 0;

    const float wx = std::modf(keypoint.x, &keypoint_int_x);
    const float wy = std::modf(keypoint.y, &keypoint_int_y);

    const int iw00 = roundf((1.0f - wx) * (1.0f - wy) * D0);
    const int iw01 = roundf(wx * (1.0f - wy) * D0);
    const int iw10 = roundf((1.0f - wx) * wy * D0);
    const int iw11 = D0 - iw00 - iw01 - iw10;

    const int16x4_t nw00 = vdup_n_s16(iw00);
    const int16x4_t nw01 = vdup_n_s16(iw01);
    const int16x4_t nw10 = vdup_n_s16(iw10);
    const int16x4_t nw11 = vdup_n_s16(iw11);

    // Convert stride from uint_t* to int16_t*
    const size_t           row_stride = _old_scharr_gx->info()->strides_in_bytes()[1] / 2;
    const Coordinates      top_left_window_corner(static_cast<int>(keypoint_int_x) - _window_dimension / 2, static_cast<int>(keypoint_int_y) - _window_dimension / 2);
    auto                   idx             = reinterpret_cast<const int16_t *>(_old_scharr_gx->buffer() + _old_scharr_gx->info()->offset_element_in_bytes(top_left_window_corner));
    auto                   idy             = reinterpret_cast<const int16_t *>(_old_scharr_gy->buffer() + _old_scharr_gy->info()->offset_element_in_bytes(top_left_window_corner));
    static const int32x4_t nshifter_scharr = vdupq_n_s32(-W_BITS);

    for(int ky = 0; ky < _window_dimension; ++ky, idx += row_stride, idy += row_stride)
    {
        int kx = 0;

        // Calculate elements in blocks of four as long as possible
        for(; kx <= _window_dimension - 4; kx += 4)
        {
            // Interpolation X
            const int16x8_t ndx_row1 = vld1q_s16(idx + kx);
            const int16x8_t ndx_row2 = vld1q_s16(idx + kx + row_stride);

            const int32x4_t nxval = compute_bilinear_interpolation(ndx_row1, ndx_row2, nw00, nw01, nw10, nw11, nshifter_scharr);

            // Interpolation Y
            const int16x8_t ndy_row1 = vld1q_s16(idy + kx);
            const int16x8_t ndy_row2 = vld1q_s16(idy + kx + row_stride);

            const int32x4_t nyval = compute_bilinear_interpolation(ndy_row1, ndy_row2, nw00, nw01, nw10, nw11, nshifter_scharr);

            // Store the intermediate data so that we don't need to recalculate them in later stage
            vst1q_s32(bilinear_ix + kx + ky * _window_dimension, nxval);
            vst1q_s32(bilinear_iy + kx + ky * _window_dimension, nyval);

            // Accumulate Ix^2
            nA11 = vmlaq_s32(nA11, nxval, nxval);
            // Accumulate Ix * Iy
            nA12 = vmlaq_s32(nA12, nxval, nyval);
            // Accumulate Iy^2
            nA22 = vmlaq_s32(nA22, nyval, nyval);
        }

        // Calculate the leftover elements
        for(; kx < _window_dimension; ++kx)
        {
            const int32_t ixval = get_pixel<int16_t>(_old_scharr_gx, top_left_window_corner.x() + kx, top_left_window_corner.y() + ky,
                                                     iw00, iw01, iw10, iw11, W_BITS);
            const int32_t iyval = get_pixel<int16_t>(_old_scharr_gy, top_left_window_corner.x() + kx, top_left_window_corner.y() + ky,
                                                     iw00, iw01, iw10, iw11, W_BITS);

            iA11 += ixval * ixval;
            iA12 += ixval * iyval;
            iA22 += iyval * iyval;

            bilinear_ix[kx + ky * _window_dimension] = ixval;
            bilinear_iy[kx + ky * _window_dimension] = iyval;
        }
    }

    iA11 += vgetq_lane_s32(nA11, 0) + vgetq_lane_s32(nA11, 1) + vgetq_lane_s32(nA11, 2) + vgetq_lane_s32(nA11, 3);
    iA12 += vgetq_lane_s32(nA12, 0) + vgetq_lane_s32(nA12, 1) + vgetq_lane_s32(nA12, 2) + vgetq_lane_s32(nA12, 3);
    iA22 += vgetq_lane_s32(nA22, 0) + vgetq_lane_s32(nA22, 1) + vgetq_lane_s32(nA22, 2) + vgetq_lane_s32(nA22, 3);

    return std::make_tuple(iA11, iA12, iA22);
}

std::pair<int, int> NELKTrackerKernel::compute_image_mismatch_vector(const NELKInternalKeypoint &old_keypoint, const NELKInternalKeypoint &new_keypoint, const int32_t *bilinear_ix,
                                                                     const int32_t *bilinear_iy)
{
    int ib1 = 0;
    int ib2 = 0;

    int32x4_t nb1 = vdupq_n_s32(0);
    int32x4_t nb2 = vdupq_n_s32(0);

    // Compute weights for the old keypoint
    float old_keypoint_int_x = 0;
    float old_keypoint_int_y = 0;

    const float old_wx = std::modf(old_keypoint.x, &old_keypoint_int_x);
    const float old_wy = std::modf(old_keypoint.y, &old_keypoint_int_y);

    const int iw00_old = roundf((1.0f - old_wx) * (1.0f - old_wy) * D0);
    const int iw01_old = roundf(old_wx * (1.0f - old_wy) * D0);
    const int iw10_old = roundf((1.0f - old_wx) * old_wy * D0);
    const int iw11_old = D0 - iw00_old - iw01_old - iw10_old;

    const int16x4_t nw00_old = vdup_n_s16(iw00_old);
    const int16x4_t nw01_old = vdup_n_s16(iw01_old);
    const int16x4_t nw10_old = vdup_n_s16(iw10_old);
    const int16x4_t nw11_old = vdup_n_s16(iw11_old);

    // Compute weights for the new keypoint
    float new_keypoint_int_x = 0;
    float new_keypoint_int_y = 0;

    const float new_wx = std::modf(new_keypoint.x, &new_keypoint_int_x);
    const float new_wy = std::modf(new_keypoint.y, &new_keypoint_int_y);

    const int iw00_new = roundf((1.0f - new_wx) * (1.0f - new_wy) * D0);
    const int iw01_new = roundf(new_wx * (1.0f - new_wy) * D0);
    const int iw10_new = roundf((1.0f - new_wx) * new_wy * D0);
    const int iw11_new = D0 - iw00_new - iw01_new - iw10_new;

    const int16x4_t nw00_new = vdup_n_s16(iw00_new);
    const int16x4_t nw01_new = vdup_n_s16(iw01_new);
    const int16x4_t nw10_new = vdup_n_s16(iw10_new);
    const int16x4_t nw11_new = vdup_n_s16(iw11_new);

    const int              row_stride = _input_new->info()->strides_in_bytes()[1];
    const Coordinates      top_left_window_corner_old(static_cast<int>(old_keypoint_int_x) - _window_dimension / 2, static_cast<int>(old_keypoint_int_y) - _window_dimension / 2);
    const Coordinates      top_left_window_corner_new(static_cast<int>(new_keypoint_int_x) - _window_dimension / 2, static_cast<int>(new_keypoint_int_y) - _window_dimension / 2);
    const uint8_t         *old_ptr         = _input_old->buffer() + _input_old->info()->offset_element_in_bytes(top_left_window_corner_old);
    const uint8_t         *new_ptr         = _input_new->buffer() + _input_new->info()->offset_element_in_bytes(top_left_window_corner_new);
    static const int32x4_t nshifter_tensor = vdupq_n_s32(-(W_BITS - 5));

    for(int ky = 0; ky < _window_dimension; ++ky, new_ptr += row_stride, old_ptr += row_stride)
    {
        int kx = 0;

        // Calculate elements in blocks of four as long as possible
        for(; kx <= _window_dimension - 4; kx += 4)
        {
            // Interpolation old tensor
            const int16x8_t nold_row1 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(old_ptr + kx)));
            const int16x8_t nold_row2 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(old_ptr + kx + row_stride)));

            const int32x4_t noldval = compute_bilinear_interpolation(nold_row1, nold_row2, nw00_old, nw01_old, nw10_old, nw11_old, nshifter_tensor);

            // Interpolation new tensor
            const int16x8_t nnew_row1 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(new_ptr + kx)));
            const int16x8_t nnew_row2 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(new_ptr + kx + row_stride)));

            const int32x4_t nnewval = compute_bilinear_interpolation(nnew_row1, nnew_row2, nw00_new, nw01_new, nw10_new, nw11_new, nshifter_tensor);

            // Calculate It gradient, i.e. pixelwise difference between old and new tensor
            const int32x4_t diff = vsubq_s32(nnewval, noldval);

            // Load the Ix and Iy gradient computed in the previous stage
            const int32x4_t nxval = vld1q_s32(bilinear_ix + kx + ky * _window_dimension);
            const int32x4_t nyval = vld1q_s32(bilinear_iy + kx + ky * _window_dimension);

            // Caculate Ix * It and Iy * It, and accumulate the results
            nb1 = vmlaq_s32(nb1, diff, nxval);
            nb2 = vmlaq_s32(nb2, diff, nyval);
        }

        // Calculate the leftover elements
        for(; kx < _window_dimension; ++kx)
        {
            const int32_t ival = get_pixel<uint8_t>(_input_old, top_left_window_corner_old.x() + kx, top_left_window_corner_old.y() + ky,
                                                    iw00_old, iw01_old, iw10_old, iw11_old, W_BITS - 5);
            const int32_t jval = get_pixel<uint8_t>(_input_new, top_left_window_corner_new.x() + kx, top_left_window_corner_new.y() + ky,
                                                    iw00_new, iw01_new, iw10_new, iw11_new, W_BITS - 5);

            const int32_t diff = jval - ival;

            ib1 += diff * bilinear_ix[kx + ky * _window_dimension];
            ib2 += diff * bilinear_iy[kx + ky * _window_dimension];
        }
    }

    ib1 += vgetq_lane_s32(nb1, 0) + vgetq_lane_s32(nb1, 1) + vgetq_lane_s32(nb1, 2) + vgetq_lane_s32(nb1, 3);
    ib2 += vgetq_lane_s32(nb2, 0) + vgetq_lane_s32(nb2, 1) + vgetq_lane_s32(nb2, 2) + vgetq_lane_s32(nb2, 3);

    return std::make_pair(ib1, ib2);
}

NELKTrackerKernel::NELKTrackerKernel()
    : _input_old(nullptr), _input_new(nullptr), _old_scharr_gx(nullptr), _old_scharr_gy(nullptr), _new_points(nullptr), _new_points_estimates(nullptr), _old_points(nullptr), _old_points_internal(),
      _new_points_internal(), _termination(Termination::TERM_CRITERIA_EPSILON), _use_initial_estimate(false), _pyramid_scale(0.0f), _epsilon(0.0f), _num_iterations(0), _window_dimension(0), _level(0),
      _num_levels(0), _valid_region()
{
}

BorderSize NELKTrackerKernel::border_size() const
{
    return BorderSize(1);
}

void NELKTrackerKernel::configure(const ITensor *input_old, const ITensor *input_new, const ITensor *old_scharr_gx, const ITensor *old_scharr_gy,
                                  const IKeyPointArray *old_points, const IKeyPointArray *new_points_estimates, IKeyPointArray *new_points,
                                  INELKInternalKeypointArray *old_points_internal, INELKInternalKeypointArray *new_points_internal,
                                  Termination termination, bool use_initial_estimate, float epsilon, unsigned int num_iterations, size_t window_dimension,
                                  size_t level, size_t num_levels, float pyramid_scale)

{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input_old, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input_new, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(old_scharr_gx, 1, DataType::S16);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(old_scharr_gy, 1, DataType::S16);

    _input_old            = input_old;
    _input_new            = input_new;
    _old_scharr_gx        = old_scharr_gx;
    _old_scharr_gy        = old_scharr_gy;
    _old_points           = old_points;
    _new_points_estimates = new_points_estimates;
    _new_points           = new_points;
    _old_points_internal  = old_points_internal;
    _new_points_internal  = new_points_internal;
    _termination          = termination;
    _use_initial_estimate = use_initial_estimate;
    _epsilon              = epsilon;
    _num_iterations       = num_iterations;
    _window_dimension     = window_dimension;
    _level                = level;
    _num_levels           = num_levels;
    _pyramid_scale        = pyramid_scale;
    _num_levels           = num_levels;

    Window window;
    window.set(Window::DimX, Window::Dimension(0, old_points->num_values()));
    window.set(Window::DimY, Window::Dimension(0, 1));

    _valid_region = intersect_valid_regions(
                        input_old->info()->valid_region(),
                        input_new->info()->valid_region(),
                        old_scharr_gx->info()->valid_region(),
                        old_scharr_gy->info()->valid_region());

    update_window_and_padding(window,
                              AccessWindowStatic(input_old->info(), _valid_region.start(0), _valid_region.start(1),
                                                 _valid_region.end(0), _valid_region.end(1)),
                              AccessWindowStatic(input_new->info(), _valid_region.start(0), _valid_region.start(1),
                                                 _valid_region.end(0), _valid_region.end(1)),
                              AccessWindowStatic(old_scharr_gx->info(), _valid_region.start(0), _valid_region.start(1),
                                                 _valid_region.end(0), _valid_region.end(1)),
                              AccessWindowStatic(old_scharr_gy->info(), _valid_region.start(0), _valid_region.start(1),
                                                 _valid_region.end(0), _valid_region.end(1)));

    INEKernel::configure(window);
}

void NELKTrackerKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    ARM_COMPUTE_ERROR_ON(_input_old->buffer() == nullptr);
    ARM_COMPUTE_ERROR_ON(_input_new->buffer() == nullptr);
    ARM_COMPUTE_ERROR_ON(_old_scharr_gx->buffer() == nullptr);
    ARM_COMPUTE_ERROR_ON(_old_scharr_gy->buffer() == nullptr);

    const int list_end   = window.x().end();
    const int list_start = window.x().start();

    init_keypoints(list_start, list_end);

    const int buffer_size = _window_dimension * _window_dimension;
    int32_t   bilinear_ix[buffer_size];
    int32_t   bilinear_iy[buffer_size];

    const int half_window = _window_dimension / 2;

    auto is_invalid_keypoint = [&](const NELKInternalKeypoint & keypoint)
    {
        const int x = std::floor(keypoint.x);
        const int y = std::floor(keypoint.y);

        return (x - half_window < _valid_region.start(0)) || (x + half_window >= _valid_region.end(0) - 1) || (y - half_window < _valid_region.start(1)) || (y + half_window >= _valid_region.end(1) - 1);
    };

    for(int list_indx = list_start; list_indx < list_end; ++list_indx)
    {
        NELKInternalKeypoint &old_keypoint = _old_points_internal->at(list_indx);
        NELKInternalKeypoint &new_keypoint = _new_points_internal->at(list_indx);

        if(!old_keypoint.tracking_status)
        {
            continue;
        }

        if(is_invalid_keypoint(old_keypoint))
        {
            if(_level == 0)
            {
                new_keypoint.tracking_status = false;
            }

            continue;
        }

        // Compute spatial gradient matrix
        int iA11 = 0;
        int iA12 = 0;
        int iA22 = 0;

        std::tie(iA11, iA12, iA22) = compute_spatial_gradient_matrix(old_keypoint, bilinear_ix, bilinear_iy);

        const float A11 = iA11 * FLT_SCALE;
        const float A12 = iA12 * FLT_SCALE;
        const float A22 = iA22 * FLT_SCALE;

        // Calculate minimum eigenvalue
        const float sum_A11_A22  = A11 + A22;
        const float discriminant = sum_A11_A22 * sum_A11_A22 - 4.0f * (A11 * A22 - A12 * A12);
        // Divide by _window_dimension^2 to reduce the floating point accummulation error
        const float minimum_eigenvalue = (sum_A11_A22 - std::sqrt(discriminant)) / (2.0f * _window_dimension * _window_dimension);

        // Determinant
        const double D = A11 * A22 - A12 * A12;

        // Check if it is a good point to track
        if(minimum_eigenvalue < EIGENVALUE_THRESHOLD || D < DETERMINANT_THRESHOLD)
        {
            // Invalidate tracked point
            if(_level == 0)
            {
                new_keypoint.tracking_status = false;
            }

            continue;
        }

        float prev_delta_x = 0.0f;
        float prev_delta_y = 0.0f;

        for(unsigned int j = 0; j < _num_iterations || _termination == Termination::TERM_CRITERIA_EPSILON; ++j)
        {
            if(is_invalid_keypoint(new_keypoint))
            {
                if(_level == 0)
                {
                    new_keypoint.tracking_status = false;
                }

                break;
            }

            // Compute image mismatch vector
            int ib1 = 0;
            int ib2 = 0;

            std::tie(ib1, ib2) = compute_image_mismatch_vector(old_keypoint, new_keypoint, bilinear_ix, bilinear_iy);

            double b1 = ib1 * FLT_SCALE;
            double b2 = ib2 * FLT_SCALE;

            // Compute motion vector -> A^-1 * -b
            const float delta_x = (A12 * b2 - A22 * b1) / D;
            const float delta_y = (A12 * b1 - A11 * b2) / D;

            // Update the new position
            new_keypoint.x += delta_x;
            new_keypoint.y += delta_y;

            const float mag2 = delta_x * delta_x + delta_y * delta_y;

            // Check if termination criteria is EPSILON and if it is satisfied
            if(mag2 <= _epsilon && (_termination == Termination::TERM_CRITERIA_EPSILON || _termination == Termination::TERM_CRITERIA_BOTH))
            {
                break;
            }

            // Check convergence analyzing the previous delta
            if(j > 0 && std::fabs(delta_x + prev_delta_x) < 0.01f && std::fabs(delta_y + prev_delta_y) < 0.01f)
            {
                new_keypoint.x -= delta_x * _pyramid_scale;
                new_keypoint.y -= delta_y * _pyramid_scale;
                break;
            }

            prev_delta_x = delta_x;
            prev_delta_y = delta_y;
        }
    }

    if(_level == 0)
    {
        for(int list_indx = list_start; list_indx < list_end; ++list_indx)
        {
            const NELKInternalKeypoint &new_keypoint = _new_points_internal->at(list_indx);

            _new_points->at(list_indx).x               = roundf(new_keypoint.x);
            _new_points->at(list_indx).y               = roundf(new_keypoint.y);
            _new_points->at(list_indx).tracking_status = new_keypoint.tracking_status ? 1 : 0;
        }
    }
}
