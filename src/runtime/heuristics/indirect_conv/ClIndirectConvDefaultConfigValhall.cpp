/*
 * Copyright (c) 2022 Arm Limited.
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
#include "src/runtime/heuristics/indirect_conv/ClIndirectConvDefaultConfigValhall.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/GPUTarget.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"

namespace arm_compute
{
namespace cl_indirect_conv
{
using namespace arm_compute::misc::shape_calculator;

ClIndirectConvDefaultConfigValhall::ClIndirectConvDefaultConfigValhall(GPUTarget gpu) : IClIndirectConvKernelConfig(gpu)
{
}

DirectConvComputeKernelInfo ClIndirectConvDefaultConfigValhall::configure(const ITensorInfo   *src,
                                                                          const ITensorInfo   *wei,
                                                                          const PadStrideInfo &conv_info)
{
    using ConfigurationFunctionExecutorPtr = DirectConvComputeKernelInfo (ClIndirectConvDefaultConfigValhall::*)(
        const ITensorInfo *src, const ITensorInfo *wei, const PadStrideInfo &conv_info);

    ClIndirectConvConfigArray<ConfigurationFunctionExecutorPtr> configs_G77(
        &ClIndirectConvDefaultConfigValhall::configure_G77_f32, &ClIndirectConvDefaultConfigValhall::configure_G77_f16);

    // Important note: Indirect convolution should not be used when the kernel size is 1x1 (pointwise). The reason is because the indirect buffer makes
    // indirect convolution less efficient than direct convolution or gemm. For this reason, the heuristic of indirect convolution has not been tuned
    // for the pointwise convolution cases.

    ConfigurationFunctionExecutorPtr func = configs_G77.get_function(src->data_type());

    ARM_COMPUTE_ERROR_ON_MSG(func == nullptr, "Data type not supported for indirect convolution");
    return (this->*func)(src, wei, conv_info);
}

DirectConvComputeKernelInfo ClIndirectConvDefaultConfigValhall::configure_G77_f32(const ITensorInfo   *src,
                                                                                  const ITensorInfo   *wei,
                                                                                  const PadStrideInfo &conv_info)
{
    DirectConvComputeKernelInfo desc;

    if (src->data_layout() == DataLayout::NHWC)
    {
        const TensorShape dst_shape = misc::shape_calculator::compute_deep_convolution_shape(*src, *wei, conv_info);
        const bool        export_weights_to_cl_image = export_to_cl_image(wei);
        const int32_t     stride_x                   = conv_info.stride().first;
        const int32_t     stride_y                   = conv_info.stride().second;
        const int32_t     ofm                        = dst_shape[0];
        const int32_t     m                          = (dst_shape[1] / stride_x) * (dst_shape[2] / stride_y);

        desc.export_weights_to_cl_image = export_weights_to_cl_image;

        if (ofm <= 4)
        {
            desc.m0 = 1;
            desc.n0 = 2;
            desc.k0 = 16;
        }
        else
        {
            // The 16000 threshold value has been identified as the right
            // one for using the biggest block size allowed on F32: 5x4x4
            if (m < 16000)
            {
                desc.m0 = 4;
                desc.n0 = 4;
                desc.k0 = 4;
            }
            else
            {
                desc.m0 = 5;
                desc.n0 = 4;
                desc.k0 = 4;
            }
        }
    }

    return desc;
}

DirectConvComputeKernelInfo ClIndirectConvDefaultConfigValhall::configure_G77_f16(const ITensorInfo   *src,
                                                                                  const ITensorInfo   *wei,
                                                                                  const PadStrideInfo &conv_info)
{
    DirectConvComputeKernelInfo desc;

    if (src->data_layout() == DataLayout::NHWC)
    {
        const TensorShape wei_shape = wei->tensor_shape();
        const TensorShape dst_shape = misc::shape_calculator::compute_deep_convolution_shape(*src, *wei, conv_info);
        const bool        export_weights_to_cl_image = export_to_cl_image(wei);

        const int32_t ofm = dst_shape[0];
        const int32_t m   = dst_shape[1] * dst_shape[2];
        const int32_t k   = wei_shape[0];

        desc.export_weights_to_cl_image = export_weights_to_cl_image;

        if (ofm <= 4)
        {
            // k0 should be as larger as possible. However, we should avoid
            // having left-over for loops that make the implementation slower.
            if ((k % 16) == 0)
            {
                desc.k0 = 16;
            }
            else if ((k % 8) == 0)
            {
                desc.k0 = 8;
            }
            else
            {
                desc.k0 = 4;
            }

            desc.m0 = 1;
            desc.n0 = ofm;
        }
        else
        {
            // The 16000 threshold value has been identified as the right
            // one for using the biggest block size allowed on F16: 8x4
            if (m >= 16000 && k < 4)
            {
                desc.m0 = 8;
                desc.n0 = 4;
                desc.k0 = 4; // k0 is clamped to k inside the kernel when k is less than 4
            }
            else
            {
                desc.m0 = 5;
                desc.n0 = 4;
                desc.k0 = 8;
            }
        }
    }

    return desc;
}
} // namespace cl_indirect_conv
} // namespace arm_compute
