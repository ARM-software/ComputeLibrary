/*
 * Copyright (c) 2021-2023 Arm Limited.
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

#include "arm_compute/function_info/ActivationLayerInfo.h"

namespace arm_compute
{
namespace assembly_utils
{
arm_gemm::Activation map_to_arm_gemm_activation(const ActivationLayerInfo &act)
{
    arm_gemm::Activation gemm_act;

    // Early exit in case lower bound is other than 0, as it's not yet supported
    if (act.b() != 0.f)
    {
        return gemm_act;
    }

    switch (act.activation())
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
    return arm_conv::PaddingValues{pad_stride_info.pad_left(), pad_stride_info.pad_top(), pad_stride_info.pad_right(),
                                   pad_stride_info.pad_bottom()};
}

arm_gemm::WeightFormat map_to_arm_gemm_weight_format(const arm_compute::WeightFormat &weight_format)
{
    arm_gemm::WeightFormat gemm_weight_fromat;

    switch (weight_format)
    {
        case arm_compute::WeightFormat::UNSPECIFIED:
            gemm_weight_fromat = arm_gemm::WeightFormat::UNSPECIFIED;
            break;
        case arm_compute::WeightFormat::ANY:
            gemm_weight_fromat = arm_gemm::WeightFormat::ANY;
            break;
        case arm_compute::WeightFormat::OHWI:
            gemm_weight_fromat = arm_gemm::WeightFormat::OHWI;
            break;
        case arm_compute::WeightFormat::OHWIo2:
            gemm_weight_fromat = arm_gemm::WeightFormat::OHWIo2;
            break;
        case arm_compute::WeightFormat::OHWIo4:
            gemm_weight_fromat = arm_gemm::WeightFormat::OHWIo4;
            break;
        case arm_compute::WeightFormat::OHWIo8:
            gemm_weight_fromat = arm_gemm::WeightFormat::OHWIo8;
            break;
        case arm_compute::WeightFormat::OHWIo16:
            gemm_weight_fromat = arm_gemm::WeightFormat::OHWIo16;
            break;
        case arm_compute::WeightFormat::OHWIo32:
            gemm_weight_fromat = arm_gemm::WeightFormat::OHWIo32;
            break;
        case arm_compute::WeightFormat::OHWIo64:
            gemm_weight_fromat = arm_gemm::WeightFormat::OHWIo64;
            break;
        case arm_compute::WeightFormat::OHWIo128:
            gemm_weight_fromat = arm_gemm::WeightFormat::OHWIo128;
            break;
        case arm_compute::WeightFormat::OHWIo4i2:
            gemm_weight_fromat = arm_gemm::WeightFormat::OHWIo4i2;
            break;
        case arm_compute::WeightFormat::OHWIo4i2_bf16:
            gemm_weight_fromat = arm_gemm::WeightFormat::OHWIo4i2_bf16;
            break;
        case arm_compute::WeightFormat::OHWIo8i2:
            gemm_weight_fromat = arm_gemm::WeightFormat::OHWIo8i2;
            break;
        case arm_compute::WeightFormat::OHWIo8i2_bf16:
            gemm_weight_fromat = arm_gemm::WeightFormat::OHWIo8i2_bf16;
            break;
        case arm_compute::WeightFormat::OHWIo16i2:
            gemm_weight_fromat = arm_gemm::WeightFormat::OHWIo16i2;
            break;
        case arm_compute::WeightFormat::OHWIo16i2_bf16:
            gemm_weight_fromat = arm_gemm::WeightFormat::OHWIo16i2_bf16;
            break;
        case arm_compute::WeightFormat::OHWIo32i2:
            gemm_weight_fromat = arm_gemm::WeightFormat::OHWIo32i2;
            break;
        case arm_compute::WeightFormat::OHWIo32i2_bf16:
            gemm_weight_fromat = arm_gemm::WeightFormat::OHWIo32i2_bf16;
            break;
        case arm_compute::WeightFormat::OHWIo64i2:
            gemm_weight_fromat = arm_gemm::WeightFormat::OHWIo64i2;
            break;
        case arm_compute::WeightFormat::OHWIo64i2_bf16:
            gemm_weight_fromat = arm_gemm::WeightFormat::OHWIo64i2_bf16;
            break;
        case arm_compute::WeightFormat::OHWIo4i4:
            gemm_weight_fromat = arm_gemm::WeightFormat::OHWIo4i4;
            break;
        case arm_compute::WeightFormat::OHWIo4i4_bf16:
            gemm_weight_fromat = arm_gemm::WeightFormat::OHWIo4i4_bf16;
            break;
        case arm_compute::WeightFormat::OHWIo8i4:
            gemm_weight_fromat = arm_gemm::WeightFormat::OHWIo8i4;
            break;
        case arm_compute::WeightFormat::OHWIo8i4_bf16:
            gemm_weight_fromat = arm_gemm::WeightFormat::OHWIo8i4_bf16;
            break;
        case arm_compute::WeightFormat::OHWIo16i4:
            gemm_weight_fromat = arm_gemm::WeightFormat::OHWIo16i4;
            break;
        case arm_compute::WeightFormat::OHWIo16i4_bf16:
            gemm_weight_fromat = arm_gemm::WeightFormat::OHWIo16i4_bf16;
            break;
        case arm_compute::WeightFormat::OHWIo32i4:
            gemm_weight_fromat = arm_gemm::WeightFormat::OHWIo32i4;
            break;
        case arm_compute::WeightFormat::OHWIo32i4_bf16:
            gemm_weight_fromat = arm_gemm::WeightFormat::OHWIo32i4_bf16;
            break;
        case arm_compute::WeightFormat::OHWIo64i4:
            gemm_weight_fromat = arm_gemm::WeightFormat::OHWIo64i4;
            break;
        case arm_compute::WeightFormat::OHWIo64i4_bf16:
            gemm_weight_fromat = arm_gemm::WeightFormat::OHWIo64i4_bf16;
            break;
        case arm_compute::WeightFormat::OHWIo2i8:
            gemm_weight_fromat = arm_gemm::WeightFormat::OHWIo2i8;
            break;
        case arm_compute::WeightFormat::OHWIo4i8:
            gemm_weight_fromat = arm_gemm::WeightFormat::OHWIo4i8;
            break;
        case arm_compute::WeightFormat::OHWIo8i8:
            gemm_weight_fromat = arm_gemm::WeightFormat::OHWIo8i8;
            break;
        case arm_compute::WeightFormat::OHWIo16i8:
            gemm_weight_fromat = arm_gemm::WeightFormat::OHWIo16i8;
            break;
        case arm_compute::WeightFormat::OHWIo32i8:
            gemm_weight_fromat = arm_gemm::WeightFormat::OHWIo32i8;
            break;
        case arm_compute::WeightFormat::OHWIo64i8:
            gemm_weight_fromat = arm_gemm::WeightFormat::OHWIo64i8;
            break;
        default:
            gemm_weight_fromat = arm_gemm::WeightFormat::UNSPECIFIED;
    }
    return gemm_weight_fromat;
}

arm_compute::WeightFormat map_to_arm_compute_weight_format(const arm_gemm::WeightFormat &weight_format)
{
    arm_compute::WeightFormat acl_weight_fromat;

    switch (weight_format)
    {
        case arm_gemm::WeightFormat::UNSPECIFIED:
            acl_weight_fromat = arm_compute::WeightFormat::UNSPECIFIED;
            break;
        case arm_gemm::WeightFormat::ANY:
            acl_weight_fromat = arm_compute::WeightFormat::ANY;
            break;
        case arm_gemm::WeightFormat::OHWI:
            acl_weight_fromat = arm_compute::WeightFormat::OHWI;
            break;
        case arm_gemm::WeightFormat::OHWIo2:
            acl_weight_fromat = arm_compute::WeightFormat::OHWIo2;
            break;
        case arm_gemm::WeightFormat::OHWIo4:
            acl_weight_fromat = arm_compute::WeightFormat::OHWIo4;
            break;
        case arm_gemm::WeightFormat::OHWIo8:
            acl_weight_fromat = arm_compute::WeightFormat::OHWIo8;
            break;
        case arm_gemm::WeightFormat::OHWIo16:
            acl_weight_fromat = arm_compute::WeightFormat::OHWIo16;
            break;
        case arm_gemm::WeightFormat::OHWIo32:
            acl_weight_fromat = arm_compute::WeightFormat::OHWIo32;
            break;
        case arm_gemm::WeightFormat::OHWIo64:
            acl_weight_fromat = arm_compute::WeightFormat::OHWIo64;
            break;
        case arm_gemm::WeightFormat::OHWIo128:
            acl_weight_fromat = arm_compute::WeightFormat::OHWIo128;
            break;
        case arm_gemm::WeightFormat::OHWIo4i2:
            acl_weight_fromat = arm_compute::WeightFormat::OHWIo4i2;
            break;
        case arm_gemm::WeightFormat::OHWIo4i2_bf16:
            acl_weight_fromat = arm_compute::WeightFormat::OHWIo4i2_bf16;
            break;
        case arm_gemm::WeightFormat::OHWIo8i2:
            acl_weight_fromat = arm_compute::WeightFormat::OHWIo8i2;
            break;
        case arm_gemm::WeightFormat::OHWIo8i2_bf16:
            acl_weight_fromat = arm_compute::WeightFormat::OHWIo8i2_bf16;
            break;
        case arm_gemm::WeightFormat::OHWIo16i2:
            acl_weight_fromat = arm_compute::WeightFormat::OHWIo16i2;
            break;
        case arm_gemm::WeightFormat::OHWIo16i2_bf16:
            acl_weight_fromat = arm_compute::WeightFormat::OHWIo16i2_bf16;
            break;
        case arm_gemm::WeightFormat::OHWIo32i2:
            acl_weight_fromat = arm_compute::WeightFormat::OHWIo32i2;
            break;
        case arm_gemm::WeightFormat::OHWIo32i2_bf16:
            acl_weight_fromat = arm_compute::WeightFormat::OHWIo32i2_bf16;
            break;
        case arm_gemm::WeightFormat::OHWIo64i2:
            acl_weight_fromat = arm_compute::WeightFormat::OHWIo64i2;
            break;
        case arm_gemm::WeightFormat::OHWIo64i2_bf16:
            acl_weight_fromat = arm_compute::WeightFormat::OHWIo64i2_bf16;
            break;
        case arm_gemm::WeightFormat::OHWIo4i4:
            acl_weight_fromat = arm_compute::WeightFormat::OHWIo4i4;
            break;
        case arm_gemm::WeightFormat::OHWIo4i4_bf16:
            acl_weight_fromat = arm_compute::WeightFormat::OHWIo4i4_bf16;
            break;
        case arm_gemm::WeightFormat::OHWIo8i4:
            acl_weight_fromat = arm_compute::WeightFormat::OHWIo8i4;
            break;
        case arm_gemm::WeightFormat::OHWIo8i4_bf16:
            acl_weight_fromat = arm_compute::WeightFormat::OHWIo8i4_bf16;
            break;
        case arm_gemm::WeightFormat::OHWIo16i4:
            acl_weight_fromat = arm_compute::WeightFormat::OHWIo16i4;
            break;
        case arm_gemm::WeightFormat::OHWIo16i4_bf16:
            acl_weight_fromat = arm_compute::WeightFormat::OHWIo16i4_bf16;
            break;
        case arm_gemm::WeightFormat::OHWIo32i4:
            acl_weight_fromat = arm_compute::WeightFormat::OHWIo32i4;
            break;
        case arm_gemm::WeightFormat::OHWIo32i4_bf16:
            acl_weight_fromat = arm_compute::WeightFormat::OHWIo32i4_bf16;
            break;
        case arm_gemm::WeightFormat::OHWIo64i4:
            acl_weight_fromat = arm_compute::WeightFormat::OHWIo64i4;
            break;
        case arm_gemm::WeightFormat::OHWIo64i4_bf16:
            acl_weight_fromat = arm_compute::WeightFormat::OHWIo64i4_bf16;
            break;
        case arm_gemm::WeightFormat::OHWIo2i8:
            acl_weight_fromat = arm_compute::WeightFormat::OHWIo2i8;
            break;
        case arm_gemm::WeightFormat::OHWIo4i8:
            acl_weight_fromat = arm_compute::WeightFormat::OHWIo4i8;
            break;
        case arm_gemm::WeightFormat::OHWIo8i8:
            acl_weight_fromat = arm_compute::WeightFormat::OHWIo8i8;
            break;
        case arm_gemm::WeightFormat::OHWIo16i8:
            acl_weight_fromat = arm_compute::WeightFormat::OHWIo16i8;
            break;
        case arm_gemm::WeightFormat::OHWIo32i8:
            acl_weight_fromat = arm_compute::WeightFormat::OHWIo32i8;
            break;
        case arm_gemm::WeightFormat::OHWIo64i8:
            acl_weight_fromat = arm_compute::WeightFormat::OHWIo64i8;
            break;
        default:
            acl_weight_fromat = arm_compute::WeightFormat::UNSPECIFIED;
    }
    return acl_weight_fromat;
}
} // namespace assembly_utils
} // namespace arm_compute
