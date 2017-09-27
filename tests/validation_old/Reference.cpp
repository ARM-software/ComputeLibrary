/*
 * Copyright (c) 2017 ARM Limited.
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
#include "Reference.h"

#include "Helpers.h"
#include "ReferenceCPP.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/validation_old/Helpers.h"

#include <random>
#include <vector>

using namespace arm_compute::test;

#ifndef DOXYGEN_SKIP_THIS
namespace arm_compute
{
namespace test
{
namespace validation
{
KeyPointArray Reference::compute_reference_harris_corners(const TensorShape &shape, float threshold, float min_dist, float sensitivity,
                                                          int32_t gradient_size, int32_t block_size, BorderMode border_mode, uint8_t constant_border_value)
{
    // Create reference
    RawTensor ref_src(shape, Format::U8);
    RawTensor raw_Gx(shape, (gradient_size == 7) ? Format::S32 : Format::S16);
    RawTensor raw_Gy(shape, (gradient_size == 7) ? Format::S32 : Format::S16);
    RawTensor raw_candidates(shape, Format::F32);
    RawTensor raw_non_maxima(shape, Format::F32);

    KeyPointArray corners(shape.total_size());

    // Fill reference
    library->fill_tensor_uniform(ref_src, 0);

    // Compute reference
    ReferenceCPP::harris_corners(ref_src, raw_Gx, raw_Gy, raw_candidates, raw_non_maxima, threshold, min_dist, sensitivity, gradient_size, block_size, corners, border_mode, constant_border_value);

    return corners;
}

RawTensor Reference::compute_reference_absolute_difference(const TensorShape &shape, DataType dt_in0, DataType dt_in1, DataType dt_out)
{
    // Create reference
    RawTensor ref_src1(shape, dt_in0);
    RawTensor ref_src2(shape, dt_in1);
    RawTensor ref_dst(shape, dt_out);

    // Fill reference
    library->fill_tensor_uniform(ref_src1, 0);
    library->fill_tensor_uniform(ref_src2, 1);

    // Compute reference
    ReferenceCPP::absolute_difference(ref_src1, ref_src2, ref_dst);

    return ref_dst;
}

RawTensor Reference::compute_reference_accumulate(const TensorShape &shape)
{
    // Create reference
    RawTensor ref_src(shape, DataType::U8);
    RawTensor ref_dst(shape, DataType::S16);

    // Fill reference
    library->fill_tensor_uniform(ref_src, 0);
    library->fill_tensor_uniform(ref_dst, 1);

    // Compute reference
    ReferenceCPP::accumulate(ref_src, ref_dst);

    return ref_dst;
}

RawTensor Reference::compute_reference_accumulate_squared(const TensorShape &shape, uint32_t shift)
{
    // Create reference
    RawTensor ref_src(shape, DataType::U8);
    RawTensor ref_dst(shape, DataType::S16);

    // Fill reference
    // ref_dst tensor filled with non-negative values
    library->fill_tensor_uniform(ref_src, 0);
    library->fill_tensor_uniform(ref_dst, 1, static_cast<int16_t>(0), std::numeric_limits<int16_t>::max());

    // Compute reference
    ReferenceCPP::accumulate_squared(ref_src, ref_dst, shift);

    return ref_dst;
}

RawTensor Reference::compute_reference_accumulate_weighted(const TensorShape &shape, float alpha)
{
    // Create reference
    RawTensor ref_src(shape, DataType::U8);
    RawTensor ref_dst(shape, DataType::U8);

    // Fill reference
    library->fill_tensor_uniform(ref_src, 0);
    library->fill_tensor_uniform(ref_dst, 1);

    // Compute reference
    ReferenceCPP::accumulate_weighted(ref_src, ref_dst, alpha);

    return ref_dst;
}

RawTensor Reference::compute_reference_pixel_wise_multiplication(const TensorShape &shape, DataType dt_in0, DataType dt_in1, DataType dt_out, float scale, ConvertPolicy convert_policy,
                                                                 RoundingPolicy rounding_policy)
{
    // Create reference
    RawTensor ref_src1(shape, dt_in0);
    RawTensor ref_src2(shape, dt_in1);
    RawTensor ref_dst(shape, dt_out);

    // Fill reference
    library->fill_tensor_uniform(ref_src1, 0);
    library->fill_tensor_uniform(ref_src2, 1);

    // Compute reference
    ReferenceCPP::pixel_wise_multiplication(ref_src1, ref_src2, ref_dst, scale, convert_policy, rounding_policy);

    return ref_dst;
}

RawTensor Reference::compute_reference_fixed_point_pixel_wise_multiplication(const TensorShape &shape, DataType dt_in0, DataType dt_in1, DataType dt_out, float scale, int fixed_point_position,
                                                                             ConvertPolicy convert_policy, RoundingPolicy rounding_policy)
{
    // Create reference
    RawTensor ref_src1(shape, dt_in0, 1, fixed_point_position);
    RawTensor ref_src2(shape, dt_in1, 1, fixed_point_position);
    RawTensor ref_dst(shape, dt_out, 1, fixed_point_position);

    // Fill reference
    library->fill_tensor_uniform(ref_src1, 0);
    library->fill_tensor_uniform(ref_src2, 1);

    // Compute reference
    ReferenceCPP::fixed_point_pixel_wise_multiplication(ref_src1, ref_src2, ref_dst, scale, convert_policy, rounding_policy);

    return ref_dst;
}

RawTensor Reference::compute_reference_roi_pooling_layer(const TensorShape &shape, DataType dt, const std::vector<ROI> &rois, const ROIPoolingLayerInfo &pool_info)
{
    TensorShape shape_dst;
    shape_dst.set(0, pool_info.pooled_width());
    shape_dst.set(1, pool_info.pooled_height());
    shape_dst.set(2, shape.z());
    shape_dst.set(3, rois.size());

    // Create reference
    RawTensor ref_src(shape, dt);
    RawTensor ref_dst(shape_dst, dt);

    // Fill reference
    std::uniform_real_distribution<> distribution(-1, 1);
    library->fill(ref_src, distribution, 0.0);

    // Compute reference
    ReferenceCPP::roi_pooling_layer(ref_src, ref_dst, rois, pool_info);

    return ref_dst;
}

} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* DOXYGEN_SKIP_THIS */
