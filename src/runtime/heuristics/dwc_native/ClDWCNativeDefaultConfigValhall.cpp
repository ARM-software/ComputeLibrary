/*
 * Copyright (c) 2022-2023 Arm Limited.
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
#include "src/runtime/heuristics/dwc_native/ClDWCNativeDefaultConfigValhall.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/GPUTarget.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/utils/helpers/AdjustVecSize.h"

#include "src/runtime/heuristics/dwc_native/ClDWCNativeHeuristicsHelpers.h"

namespace arm_compute
{
namespace cl_dwc
{
ClDWCNativeDefaultConfigValhall::ClDWCNativeDefaultConfigValhall(GPUTarget gpu) : IClDWCNativeKernelConfig(gpu)
{
}

DWCComputeKernelInfo ClDWCNativeDefaultConfigValhall::configure(const ITensorInfo   *src,
                                                                const ITensorInfo   *wei,
                                                                const PadStrideInfo &conv_info,
                                                                const Size2D        &dilation,
                                                                unsigned int         depth_multiplier)
{
    using ConfigurationFunctionExecutorPtr = DWCComputeKernelInfo (ClDWCNativeDefaultConfigValhall::*)(
        const ITensorInfo *src, const ITensorInfo *wei, const PadStrideInfo &conv_info, const Size2D &dilation,
        unsigned int depth_multiplier);

    ClDWCNativeConfigArray<ConfigurationFunctionExecutorPtr> configs_G78(
        &ClDWCNativeDefaultConfigValhall::configure_G78_f32, &ClDWCNativeDefaultConfigValhall::configure_G78_f16,
        &ClDWCNativeDefaultConfigValhall::configure_G78_u8);

    ClDWCNativeConfigArray<ConfigurationFunctionExecutorPtr> configs_G77(
        &ClDWCNativeDefaultConfigValhall::configure_G78_f32, &ClDWCNativeDefaultConfigValhall::configure_G77_f16,
        &ClDWCNativeDefaultConfigValhall::configure_G78_u8);

    ConfigurationFunctionExecutorPtr func = nullptr;
    switch (_target)
    {
        case GPUTarget::G77:
            func = configs_G77.get_function(src->data_type());
            break;
        case GPUTarget::G78:
        default:
            func = configs_G78.get_function(src->data_type());
            break;
    }

    ARM_COMPUTE_ERROR_ON_MSG(func == nullptr, "Data type not supported for depthwise convolution");
    return (this->*func)(src, wei, conv_info, dilation, depth_multiplier);
}

DWCComputeKernelInfo ClDWCNativeDefaultConfigValhall::configure_G78_f32(const ITensorInfo   *src,
                                                                        const ITensorInfo   *wei,
                                                                        const PadStrideInfo &conv_info,
                                                                        const Size2D        &dilation,
                                                                        unsigned int         depth_multiplier)
{
    DWCComputeKernelInfo desc;

    if (src->data_layout() == DataLayout::NHWC)
    {
        const size_t      idx_c     = get_data_layout_dimension_index(wei->data_layout(), DataLayoutDimension::CHANNEL);
        const size_t      idx_w     = get_data_layout_dimension_index(wei->data_layout(), DataLayoutDimension::WIDTH);
        const TensorShape wei_shape = wei->tensor_shape();
        const size_t      kernel_c  = wei_shape[idx_c];
        const size_t      kernel_w  = wei_shape[idx_w];

        desc.export_input_to_cl_image   = false;
        desc.export_weights_to_cl_image = use_cl_image_for_weights(wei, depth_multiplier);

        if (depth_multiplier == 1)
        {
            desc.n0 = 4;
        }
        else
        {
            if ((depth_multiplier % 4) == 0)
            {
                desc.n0 = 4;
            }
            else if ((depth_multiplier % 2) == 0)
            {
                desc.n0 = 2;
            }
            else
            {
                desc.n0 = 1;
            }
        }

        // Note: If we reduce n0, export to cl_image must be false
        ARM_COMPUTE_ERROR_ON((adjust_vec_size(desc.n0, kernel_c) != desc.n0) &&
                             (desc.export_weights_to_cl_image == true));

        desc.n0 = adjust_vec_size(desc.n0, kernel_c);

        // Set m0 only if stride_x == 1 and dilation_x == 1
        if (conv_info.stride().first == 1 && dilation.x() == 1)
        {
            if ((kernel_w >= 9) || (kernel_w == 1))
            {
                desc.m0 = 1;
            }
            else
            {
                desc.m0 = 2;
            }
        }
        else
        {
            desc.m0 = 1;
        }
    }

    return desc;
}

DWCComputeKernelInfo ClDWCNativeDefaultConfigValhall::configure_G78_f16(const ITensorInfo   *src,
                                                                        const ITensorInfo   *wei,
                                                                        const PadStrideInfo &conv_info,
                                                                        const Size2D        &dilation,
                                                                        unsigned int         depth_multiplier)
{
    DWCComputeKernelInfo desc;

    if (src->data_layout() == DataLayout::NHWC)
    {
        // Src and weights have the same dimension indices
        const size_t      idx_c     = get_data_layout_dimension_index(wei->data_layout(), DataLayoutDimension::CHANNEL);
        const size_t      idx_w     = get_data_layout_dimension_index(wei->data_layout(), DataLayoutDimension::WIDTH);
        const TensorShape src_shape = src->tensor_shape();
        const TensorShape wei_shape = wei->tensor_shape();
        const size_t      src_w     = src_shape[idx_w];
        const size_t      kernel_c  = wei_shape[idx_c];
        const size_t      kernel_w  = wei_shape[idx_w];

        desc.export_input_to_cl_image   = false;
        desc.export_weights_to_cl_image = use_cl_image_for_weights(wei, depth_multiplier);

        if (depth_multiplier == 1)
        {
            if (desc.export_weights_to_cl_image == false)
            {
                desc.n0 = 8;
            }
            else
            {
                desc.n0 = 4;
            }
        }
        else
        {
            if ((depth_multiplier % 4) == 0)
            {
                desc.n0 = 4;
            }
            else if ((depth_multiplier % 2) == 0)
            {
                desc.n0 = 2;
            }
            else
            {
                desc.n0 = 1;
            }
        }

        // Note: If we reduce n0, export to cl_image must be false
        ARM_COMPUTE_ERROR_ON((adjust_vec_size(desc.n0, kernel_c) != desc.n0) &&
                             (desc.export_weights_to_cl_image == true));

        desc.n0 = adjust_vec_size(desc.n0, kernel_c);

        // Set m0 only if stride_x == 1 and dilation_x == 1
        if (conv_info.stride().first == 1 && dilation.x() == 1)
        {
            if ((kernel_w >= 9) || (kernel_w == 1))
            {
                desc.m0 = 1;
            }
            else
            {
                if ((src_w % 5) == 0)
                {
                    desc.m0 = 5;
                }
                else
                {
                    desc.m0 = 4;
                }
            }
        }
        else
        {
            desc.m0 = 1;
        }
    }

    return desc;
}

DWCComputeKernelInfo ClDWCNativeDefaultConfigValhall::configure_G78_u8(const ITensorInfo   *src,
                                                                       const ITensorInfo   *wei,
                                                                       const PadStrideInfo &conv_info,
                                                                       const Size2D        &dilation,
                                                                       unsigned int         depth_multiplier)
{
    ARM_COMPUTE_UNUSED(wei);

    DWCComputeKernelInfo desc;

    if (src->data_layout() == DataLayout::NHWC)
    {
        desc.export_input_to_cl_image   = false;
        desc.export_weights_to_cl_image = false;
        desc.n0                         = (depth_multiplier == 1) ? 4 : 1;
        if (conv_info.stride().first == 1 && dilation.x() == 1 && depth_multiplier == 1)
        {
            desc.m0 = 2;
        }
        else
        {
            desc.m0 = 1;
        }
    }

    return desc;
}

DWCComputeKernelInfo ClDWCNativeDefaultConfigValhall::configure_G77_f16(const ITensorInfo   *src,
                                                                        const ITensorInfo   *wei,
                                                                        const PadStrideInfo &conv_info,
                                                                        const Size2D        &dilation,
                                                                        unsigned int         depth_multiplier)
{
    DWCComputeKernelInfo desc;

    if (src->data_layout() == DataLayout::NHWC)
    {
        const size_t      idx_c     = get_data_layout_dimension_index(wei->data_layout(), DataLayoutDimension::CHANNEL);
        const size_t      idx_w     = get_data_layout_dimension_index(wei->data_layout(), DataLayoutDimension::WIDTH);
        const TensorShape wei_shape = wei->tensor_shape();
        const size_t      kernel_c  = wei_shape[idx_c];
        const size_t      kernel_w  = wei_shape[idx_w];

        desc.export_input_to_cl_image   = false;
        desc.export_weights_to_cl_image = use_cl_image_for_weights(wei, depth_multiplier);

        if (depth_multiplier == 1)
        {
            if (desc.export_weights_to_cl_image == false)
            {
                desc.n0 = 8;
            }
            else
            {
                desc.n0 = 4;
            }
        }
        else
        {
            if ((depth_multiplier % 4) == 0)
            {
                desc.n0 = 4;
            }
            else if ((depth_multiplier % 2) == 0)
            {
                desc.n0 = 2;
            }
            else
            {
                desc.n0 = 1;
            }
        }

        // Note: If we reduce n0, export to cl_image must be false
        ARM_COMPUTE_ERROR_ON((adjust_vec_size(desc.n0, kernel_c) != desc.n0) &&
                             (desc.export_weights_to_cl_image == true));

        desc.n0 = adjust_vec_size(desc.n0, kernel_c);

        // Set m0 only if stride_x == 1 and dilation_x == 1
        if (conv_info.stride().first == 1 && dilation.x() == 1)
        {
            if ((kernel_w >= 9) || (kernel_w == 1))
            {
                desc.m0 = 1;
            }
            else
            {
                desc.m0 = 2;
            }
        }
        else
        {
            desc.m0 = 1;
        }
    }

    return desc;
}
} // namespace cl_dwc
} // namespace arm_compute
