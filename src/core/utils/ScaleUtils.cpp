/*
 * Copyright (c) 2020, 2022 Arm Limited.
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

#include "src/core/utils/ScaleUtils.h"

#include "arm_compute/core/CPP/CPPTypes.h"
#include "arm_compute/core/TensorInfo.h"

#include "src/common/cpuinfo/CpuIsaInfo.h"

float arm_compute::scale_utils::calculate_resize_ratio(size_t input_size, size_t output_size, bool align_corners)
{
    const size_t offset = (align_corners && output_size > 1) ? 1 : 0;
    const auto   in     = input_size - offset;
    const auto   out    = output_size - offset;

    ARM_COMPUTE_ERROR_ON((input_size == 0 || output_size == 0) && offset == 1);
    ARM_COMPUTE_ERROR_ON(out == 0);

    return static_cast<float>(in) / static_cast<float>(out);
}

bool arm_compute::scale_utils::is_precomputation_required(DataLayout          data_layout,
                                                          DataType            data_type,
                                                          InterpolationPolicy policy,
                                                          BorderMode          border_mode)
{
    // Do not calculate precomputed weights and indices if kernel code doesn't use them
    if (data_layout == DataLayout::NHWC)
    {
        switch (data_type)
        {
            case DataType::F32:
            case DataType::F16:
                return (CPUInfo::get().get_isa().sve == true && policy == InterpolationPolicy::NEAREST_NEIGHBOR);
            case DataType::U8:
            case DataType::S8:
            case DataType::QASYMM8:
            case DataType::QASYMM8_SIGNED:
                return (border_mode != BorderMode::REPLICATE) || (policy == InterpolationPolicy::NEAREST_NEIGHBOR);
            default:
                return true;
        }
    }

    return true;
}
