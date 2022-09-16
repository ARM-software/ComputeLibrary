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
#include "src/gpu/cl/kernels/direct_conv/ClDirectConvDefaultConfigValhall.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/GPUTarget.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include <utility>

namespace arm_compute
{
namespace cl_direct_conv
{
using namespace arm_compute::misc::shape_calculator;

ClDirectConvDefaultConfigValhall::ClDirectConvDefaultConfigValhall(GPUTarget gpu)
    : IClDirectConvKernelConfig(gpu)
{
}

DirectConvComputeKernelInfo ClDirectConvDefaultConfigValhall::configure(const ITensorInfo *src, const ITensorInfo *wei, const PadStrideInfo &conv_info)
{
    using ConfigurationFunctionExecutorPtr = DirectConvComputeKernelInfo (ClDirectConvDefaultConfigValhall::*)(const ITensorInfo * src, const ITensorInfo * wei, const PadStrideInfo & conv_info);

    ClDirectConvConfigArray<ConfigurationFunctionExecutorPtr> configs_G78(&ClDirectConvDefaultConfigValhall::configure_G78_f32,
                                                                          &ClDirectConvDefaultConfigValhall::configure_G78_f16,
                                                                          &ClDirectConvDefaultConfigValhall::configure_G78_u8);

    ClDirectConvConfigArray<ConfigurationFunctionExecutorPtr> configs_G57(&ClDirectConvDefaultConfigValhall::configure_G57_f32,
                                                                          &ClDirectConvDefaultConfigValhall::configure_G57_f16,
                                                                          &ClDirectConvDefaultConfigValhall::configure_G78_u8);

    ConfigurationFunctionExecutorPtr func = nullptr;
    switch(_target)
    {
        case GPUTarget::G57:
            func = configs_G57.get_function(src->data_type());
            break;
        case GPUTarget::G78:
        default:
            func = configs_G78.get_function(src->data_type());
            break;
    }

    ARM_COMPUTE_ERROR_ON_MSG(func == nullptr, "Data type not supported for direct convolution");
    return (this->*func)(src, wei, conv_info);
}

DirectConvComputeKernelInfo ClDirectConvDefaultConfigValhall::configure_G78_f32(const ITensorInfo *src, const ITensorInfo *wei, const PadStrideInfo &conv_info)
{
    DirectConvComputeKernelInfo desc;

    if(src->data_layout() == DataLayout::NHWC)
    {
        // Get the output shape
        const TensorShape wei_shape                  = wei->tensor_shape();
        const TensorShape dst_shape                  = misc::shape_calculator::compute_deep_convolution_shape(*src, *wei, conv_info);
        const bool        export_weights_to_cl_image = export_to_cl_image(wei);

        const int32_t ofm          = dst_shape[0];
        const int32_t m            = dst_shape[1] * dst_shape[2];
        const bool    is_pointwise = (wei_shape[1] == wei_shape[2]) && wei_shape[1] == 1;

        desc.export_weights_to_cl_image = export_weights_to_cl_image;

        if(dst_shape[0] <= 4)
        {
            if(is_pointwise)
            {
                if(ofm == 4)
                {
                    desc.m0 = 1;
                    desc.n0 = 4;
                    desc.k0 = 16;
                }
                else
                {
                    desc.m0 = 1;
                    desc.n0 = 1;
                    desc.k0 = 16;
                }
            }
            else
            {
                desc.m0 = 1;
                desc.n0 = 2;
                desc.k0 = 16;
            }
        }
        else
        {
            if(m < 64)
            {
                desc.m0 = 1;
                desc.n0 = 1;
                desc.k0 = 16;
            }
            else
            {
                desc.m0 = 4;
                desc.n0 = 4;
                desc.k0 = 4;
            }
        }
    }

    return desc;
}

DirectConvComputeKernelInfo ClDirectConvDefaultConfigValhall::configure_G78_f16(const ITensorInfo *src, const ITensorInfo *wei, const PadStrideInfo &conv_info)
{
    DirectConvComputeKernelInfo desc;

    if(src->data_layout() == DataLayout::NHWC)
    {
        // Get the output shape
        const TensorShape wei_shape                  = wei->tensor_shape();
        const TensorShape dst_shape                  = misc::shape_calculator::compute_deep_convolution_shape(*src, *wei, conv_info);
        const bool        export_weights_to_cl_image = export_to_cl_image(wei);

        const int32_t ofm          = dst_shape[0];
        const int32_t m            = dst_shape[1] * dst_shape[2];
        const bool    is_pointwise = (wei_shape[1] == wei_shape[2]) && wei_shape[1] == 1;

        desc.export_weights_to_cl_image = export_weights_to_cl_image;

        if(dst_shape[0] <= 4)
        {
            if(is_pointwise)
            {
                if(ofm == 4)
                {
                    desc.m0 = 1;
                    desc.n0 = 4;
                    desc.k0 = 16;
                }
                else
                {
                    desc.m0 = 1;
                    desc.n0 = 1;
                    desc.k0 = 16;
                }
            }
            else
            {
                desc.m0 = 1;
                desc.n0 = dst_shape[0];
                desc.k0 = 16;
            }
        }
        else
        {
            if(m < 64)
            {
                desc.m0 = 1;
                desc.n0 = 1;
                desc.k0 = 16;
            }
            else
            {
                if(ofm > 16)
                {
                    desc.m0 = 4;
                    desc.n0 = 4;
                    desc.k0 = 8;
                }
                else
                {
                    desc.m0 = 4;
                    desc.n0 = 4;
                    desc.k0 = 16;
                }
            }
        }
    }

    return desc;
}

DirectConvComputeKernelInfo ClDirectConvDefaultConfigValhall::configure_G78_u8(const ITensorInfo *src, const ITensorInfo *wei, const PadStrideInfo &conv_info)
{
    DirectConvComputeKernelInfo desc;

    if(src->data_layout() == DataLayout::NHWC)
    {
        // Get the output shape
        TensorShape output_shape = misc::shape_calculator::compute_deep_convolution_shape(*src, *wei, conv_info);

        desc.n0 = 4;

        if(output_shape[0] > 16)
        {
            desc.m0 = 4;
        }

        desc.k0 = 16;

        desc.export_weights_to_cl_image = false;
    }

    return desc;
}

DirectConvComputeKernelInfo ClDirectConvDefaultConfigValhall::configure_G57_f32(const ITensorInfo *src, const ITensorInfo *wei, const PadStrideInfo &conv_info)
{
    DirectConvComputeKernelInfo desc;

    if(src->data_layout() == DataLayout::NHWC)
    {
        // Get the output shape
        const TensorShape wei_shape                  = wei->tensor_shape();
        const TensorShape dst_shape                  = misc::shape_calculator::compute_deep_convolution_shape(*src, *wei, conv_info);
        const bool        export_weights_to_cl_image = export_to_cl_image(wei);

        const int32_t m            = dst_shape[1] * dst_shape[2];
        const bool    is_pointwise = (wei_shape[1] == wei_shape[2]) && wei_shape[1] == 1;

        desc.export_weights_to_cl_image = export_weights_to_cl_image;

        if(dst_shape[0] <= 4)
        {
            if(is_pointwise)
            {
                desc.m0 = 1;
                desc.n0 = 1;
                desc.k0 = 16;
            }
            else
            {
                desc.m0 = 1;
                desc.n0 = dst_shape[0];
                desc.k0 = 16;
            }
        }
        else
        {
            if(m < 64)
            {
                if(m == 1)
                {
                    desc.m0 = 1;
                    desc.n0 = 1;
                    desc.k0 = 16;
                }
                else
                {
                    desc.m0 = 4;
                    desc.n0 = 2;
                    desc.k0 = 8;
                }
            }
            else
            {
                desc.m0 = 4;
                desc.n0 = 4;
                desc.k0 = 4;
            }
        }
    }

    return desc;
}

DirectConvComputeKernelInfo ClDirectConvDefaultConfigValhall::configure_G57_f16(const ITensorInfo *src, const ITensorInfo *wei, const PadStrideInfo &conv_info)
{
    DirectConvComputeKernelInfo desc;

    if(src->data_layout() == DataLayout::NHWC)
    {
        // Get the output shape
        const TensorShape wei_shape                  = wei->tensor_shape();
        const TensorShape dst_shape                  = misc::shape_calculator::compute_deep_convolution_shape(*src, *wei, conv_info);
        const bool        export_weights_to_cl_image = export_to_cl_image(wei);

        const int32_t ofm          = dst_shape[0];
        const int32_t m            = dst_shape[1] * dst_shape[2];
        const bool    is_pointwise = (wei_shape[1] == wei_shape[2]) && wei_shape[1] == 1;

        desc.export_weights_to_cl_image = export_weights_to_cl_image;

        if(dst_shape[0] <= 4)
        {
            if(is_pointwise)
            {
                desc.m0 = 2;
                desc.n0 = 1;
                desc.k0 = 16;
            }
            else
            {
                desc.m0 = 1;
                desc.n0 = dst_shape[0];
                desc.k0 = 16;
            }
        }
        else
        {
            if(m < 64)
            {
                if(m == 1)
                {
                    desc.m0 = 1;
                    desc.n0 = 1;
                    desc.k0 = 16;
                }
                else
                {
                    desc.m0 = 4;
                    desc.n0 = 2;
                    desc.k0 = 8;
                }
            }
            else
            {
                if(ofm > 16)
                {
                    desc.m0 = 4;
                    desc.n0 = 8;
                    desc.k0 = 8;
                }
                else
                {
                    desc.m0 = 8;
                    desc.n0 = 4;
                    desc.k0 = 4;
                }
            }
        }
    }

    return desc;
}
} // namespace opencl
} // namespace arm_compute
