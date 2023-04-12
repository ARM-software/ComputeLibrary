/*
 * Copyright (c) 2018-2023 Arm Limited.
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
#include "src/cpu/kernels/CpuElementwiseUnaryKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "src/core/CPP/Validate.h"
#include "src/core/common/Registrars.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"
#include "src/cpu/kernels/elementwise_unary/list.h"
#include "support/ToolchainSupport.h"

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
namespace
{
#ifdef __aarch64__

std::unique_ptr<uint8_t[]> q8_prepare_lut(ElementWiseUnary op, const ITensorInfo *src, const ITensorInfo *dst)
{
    ARM_COMPUTE_ERROR_ON(src->data_type() != dst->data_type());
    ARM_COMPUTE_ERROR_ON(!is_data_type_quantized(src->data_type()));
    ARM_COMPUTE_ERROR_ON(src->element_size() != 1);

    auto       lut       = std::unique_ptr<uint8_t[]>(new uint8_t[256]);
    const auto is_signed = src->data_type() == DataType::QASYMM8_SIGNED;
    const auto src_qi    = src->quantization_info().uniform();
    const auto dst_qi    = dst->quantization_info().uniform();

    const auto dst_min_fp = (((is_signed) ? -128 : 0) - dst_qi.offset) * dst_qi.scale;
    const auto dst_max_fp = (((is_signed) ? 127 : 255) - dst_qi.offset) * dst_qi.scale;

    for(int i = 0; i < 256; ++i)
    {
        const auto in     = (is_signed) ? dequantize_qasymm8_signed(static_cast<int8_t>(i), src_qi) : dequantize_qasymm8(i, src_qi);
        float      result = 0;

        switch(op)
        {
            case ElementWiseUnary::RSQRT:
                result = 1 / sqrt(in);
                break;

            case ElementWiseUnary::EXP:
                result = std::exp(in);
                break;

            case ElementWiseUnary::NEG:
                result = -in;
                break;

            case ElementWiseUnary::LOG:
                result = std::log(in);
                break;

            case ElementWiseUnary::ABS:
                result = std::abs(in);
                break;

            case ElementWiseUnary::ROUND:
                result = support::cpp11::nearbyint(in);
                break;

            case ElementWiseUnary::SIN:
                result = std::sin(in);
                break;

            default:
                ARM_COMPUTE_ERROR("NOT_SUPPORTED!");
        }

        result = utility::clamp(result, dst_min_fp, dst_max_fp);

        const auto out = (is_signed) ? static_cast<uint8_t>(quantize_qasymm8_signed(result, dst_qi)) : quantize_qasymm8(result, dst_qi);
        lut[i]         = out;
    }

    return lut;
}

#endif // __aarch64__

static const std::vector<CpuElementwiseUnaryKernel::ElementwiseUnaryKernel> available_kernels =
{
    {
        "sve_fp32_elementwise_unary",
        [](const DataTypeISASelectorData & data)
        {
            return (data.dt == DataType::F32 && data.isa.sve);
        },
        REGISTER_FP32_SVE(sve_fp32_elementwise_unary),
        nullptr,
    },
    {
        "sve_fp16_elementwise_unary",
        [](const DataTypeISASelectorData & data)
        {
            return (data.dt == DataType::F16 && data.isa.sve && data.isa.fp16);
        },
        REGISTER_FP16_SVE(sve_fp16_elementwise_unary),
        nullptr,
    },
    {
        "sve_s32_elementwise_unary",
        [](const DataTypeISASelectorData & data)
        {
            return (data.dt == DataType::S32 && data.isa.sve);
        },
        REGISTER_INTEGER_SVE(sve_s32_elementwise_unary),
        nullptr,
    },
    {
        "neon_fp32_elementwise_unary",
        [](const DataTypeISASelectorData & data)
        {
            return data.dt == DataType::F32;
        },
        REGISTER_FP32_NEON(neon_fp32_elementwise_unary),
        nullptr,
    },
    {
        "neon_fp16_elementwise_unary",
        [](const DataTypeISASelectorData & data)
        {
            return data.dt == DataType::F16 && data.isa.fp16;
        },
        REGISTER_FP16_NEON(neon_fp16_elementwise_unary),
        nullptr,
    },
    {
        "neon_s32_elementwise_unary",
        [](const DataTypeISASelectorData & data)
        {
            return data.dt == DataType::S32;
        },
        REGISTER_INTEGER_NEON(neon_s32_elementwise_unary),
        nullptr,
    },
#ifdef __aarch64__
    {
        "sve_q8_elementwise_unary",
        [](const DataTypeISASelectorData & data)
        {
            return (data.dt == DataType::QASYMM8 || data.dt == DataType::QASYMM8_SIGNED) && data.isa.sve;
        },
        REGISTER_QASYMM8_SVE(sve_q8_elementwise_unary),
        &q8_prepare_lut,
    },
    {
        "neon_q8_elementwise_unary",
        [](const DataTypeISASelectorData & data)
        {
            return data.dt == DataType::QASYMM8 || data.dt == DataType::QASYMM8_SIGNED;
        },
        REGISTER_QASYMM8_NEON(neon_q8_elementwise_unary),
        &q8_prepare_lut,
    },
#else  // __aarch64__
    {
        "neon_qasymm8_signed_elementwise_unary",
        [](const DataTypeISASelectorData & data)
        {
            return data.dt == DataType::QASYMM8_SIGNED;
        },
        REGISTER_QASYMM8_SIGNED_NEON(neon_qasymm8_signed_elementwise_unary),
        nullptr,
    },
    {
        "neon_qasymm8_elementwise_unary",
        [](const DataTypeISASelectorData & data)
        {
            return data.dt == DataType::QASYMM8;
        },
        REGISTER_QASYMM8_NEON(neon_qasymm8_elementwise_unary),
        nullptr,
    },
#endif // __aarch64__
};

} // namespace

void CpuElementwiseUnaryKernel::configure(ElementWiseUnary op, const ITensorInfo &src, ITensorInfo &dst)
{
    ARM_COMPUTE_ERROR_THROW_ON(validate(op, src, dst));
    const auto uk = CpuElementwiseUnaryKernel::get_implementation(DataTypeISASelectorData{ src.data_type(), CPUInfo::get().get_isa() });
    ARM_COMPUTE_ERROR_ON(uk == nullptr || uk->ukernel == nullptr);

    _op         = op;
    _run_method = uk->ukernel;
    _name       = std::string("CpuElementwiseUnaryKernel").append("/").append(uk->name);

    // If input shape is dynamic, expect a configured window and dst at run-time.
    if(src.is_dynamic())
    {
        return;
    }

    if(uk->prepare_func != nullptr)
    {
        _lut = uk->prepare_func(op, &src, &dst);
    }

    auto shape_and_window = compute_output_shape_and_window(src.tensor_shape());
    auto_init_if_empty(dst, shape_and_window.first, 1, src.data_type());
    ICpuKernel::configure(shape_and_window.second);
}

Status CpuElementwiseUnaryKernel::validate(ElementWiseUnary op, const ITensorInfo &src, const ITensorInfo &dst)
{
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(&src);

    const auto *uk = CpuElementwiseUnaryKernel::get_implementation(DataTypeISASelectorData{ src.data_type(), CPUInfo::get().get_isa() });

    ARM_COMPUTE_RETURN_ERROR_ON(uk == nullptr || uk->ukernel == nullptr);

    switch(op)
    {
        case ElementWiseUnary::EXP:
        case ElementWiseUnary::RSQRT:
        case ElementWiseUnary::LOG:
        case ElementWiseUnary::ROUND:
        case ElementWiseUnary::SIN:
            ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&src, 1, DataType::F16, DataType::F32, DataType::QASYMM8, DataType::QASYMM8_SIGNED);
            break;
        case ElementWiseUnary::NEG:
        case ElementWiseUnary::ABS:
            ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&src, 1, DataType::F16, DataType::F32, DataType::S32, DataType::QASYMM8, DataType::QASYMM8_SIGNED);
            break;
        default:
            ARM_COMPUTE_ERROR("ElementWiseUnary operation not supported");
    }
    // Validate in case of configured dst
    if(dst.total_size() > 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(&src, &dst);
    }

    return Status{};
}

void CpuElementwiseUnaryKernel::run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);

    auto src = tensors.get_const_tensor(TensorType::ACL_SRC);
    auto dst = tensors.get_tensor(TensorType::ACL_DST);

    _run_method(src, dst, window, _op, _lut.get());
}

const char *CpuElementwiseUnaryKernel::name() const
{
    return _name.c_str();
}

const std::vector<CpuElementwiseUnaryKernel::ElementwiseUnaryKernel> &CpuElementwiseUnaryKernel::get_available_kernels()
{
    return available_kernels;
}

} // namespace kernels
} // namespace cpu
} // namespace arm_compute
