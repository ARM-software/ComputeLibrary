/*
 * Copyright (c) 2021 Arm Limited.
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
#include "src/core/utils/AssemblyUtils.h"

namespace arm_compute
{
namespace assembly_utils
{
arm_gemm::Activation map_to_arm_gemm_activation(const ActivationLayerInfo &act)
{
    arm_gemm::Activation gemm_act;

    // Early exit in case lower bound is other than 0, as it's not yet supported
    if(act.b() != 0.f)
    {
        return gemm_act;
    }

    switch(act.activation())
    {
        case ActivationLayerInfo::ActivationFunction::RELU:
            gemm_act.type = arm_gemm::Activation::Type::ReLU;
            break;
        case ActivationLayerInfo::ActivationFunction::BOUNDED_RELU:
            gemm_act.type   = arm_gemm::Activation::Type::BoundedReLU;
            gemm_act.param1 = act.a();
            gemm_act.param2 = 0.f;
            break;
        case ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU:
            gemm_act.type   = arm_gemm::Activation::Type::BoundedReLU;
            gemm_act.param1 = act.a();
            gemm_act.param2 = act.b();
            break;
        default:
            gemm_act.type = arm_gemm::Activation::Type::None;
    }

    return gemm_act;
}

arm_conv::PaddingValues map_to_arm_conv_padding(const PadStrideInfo &pad_stride_info)
{
    return arm_conv::PaddingValues{ pad_stride_info.pad_left(),
                                    pad_stride_info.pad_top(),
                                    pad_stride_info.pad_right(),
                                    pad_stride_info.pad_bottom() };
}
} // namespace assembly_utils
} // namespace arm_compute
