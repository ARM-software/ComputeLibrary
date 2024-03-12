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
#include "src/cpu/operators/CpuGemmLowpOutputStage.h"

#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"

#include "src/common/utils/Log.h"
#include "src/cpu/kernels/CpuGemmLowpQuantizeDownInt32ScaleKernel.h"
#include "src/cpu/kernels/CpuGemmLowpQuantizeDownInt32ToInt16ScaleByFixedPointKernel.h"
#include "src/cpu/kernels/CpuGemmLowpQuantizeDownInt32ToInt8ScaleByFixedPointKernel.h"
#include "src/cpu/kernels/CpuGemmLowpQuantizeDownInt32ToUint8ScaleByFixedPointKernel.h"

namespace arm_compute
{
namespace cpu
{
void CpuGemmLowpOutputStage::configure(ITensorInfo                   *src,
                                       ITensorInfo                   *bias,
                                       ITensorInfo                   *dst,
                                       const GEMMLowpOutputStageInfo &info)
{
    // Perform validate step
    ARM_COMPUTE_ERROR_THROW_ON(CpuGemmLowpOutputStage::validate(src, bias, dst, info));
    ARM_COMPUTE_LOG_PARAMS(src, bias, dst, info);

    switch (info.type)
    {
        case GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT:
        {
            switch (info.output_data_type)
            {
                case DataType::QASYMM8:
                {
                    auto k = std::make_unique<kernels::CpuGemmLowpQuantizeDownInt32ToUint8ScaleByFixedPointKernel>();
                    k->configure(src, bias, dst, info.gemmlowp_multiplier, info.gemmlowp_shift, info.gemmlowp_offset,
                                 info.gemmlowp_min_bound, info.gemmlowp_max_bound);
                    _kernel = std::move(k);
                    break;
                }
                case DataType::QASYMM8_SIGNED:
                {
                    auto k = std::make_unique<kernels::CpuGemmLowpQuantizeDownInt32ToInt8ScaleByFixedPointKernel>();
                    k->configure(src, bias, dst, info.gemmlowp_multiplier, info.gemmlowp_shift, info.gemmlowp_offset,
                                 info.gemmlowp_min_bound, info.gemmlowp_max_bound);
                    _kernel = std::move(k);
                    break;
                }
                case DataType::QSYMM16:
                {
                    auto k = std::make_unique<kernels::CpuGemmLowpQuantizeDownInt32ToInt16ScaleByFixedPointKernel>();
                    k->configure(src, bias, dst, info.gemmlowp_multiplier, info.gemmlowp_shift, info.gemmlowp_min_bound,
                                 info.gemmlowp_max_bound);
                    _kernel = std::move(k);
                    break;
                }
                default:
                {
                    ARM_COMPUTE_ERROR("Unsupported output data type.");
                    break;
                }
            }
            break;
        }
        case GEMMLowpOutputStageType::QUANTIZE_DOWN:
        {
            switch (info.output_data_type)
            {
                case DataType::QASYMM8:
                case DataType::QASYMM8_SIGNED:
                {
                    auto k = std::make_unique<kernels::CpuGemmLowpQuantizeDownInt32ScaleKernel>();
                    k->configure(src, bias, dst, &info);
                    _kernel = std::move(k);
                    break;
                }
                default:
                {
                    ARM_COMPUTE_ERROR("Unsupported output data type.");
                    break;
                }
            }
            break;
        }
        default:
            ARM_COMPUTE_ERROR("Unsupported GEMMLowpOutputStage type.");
    }
}

Status CpuGemmLowpOutputStage::validate(const ITensorInfo             *src,
                                        const ITensorInfo             *bias,
                                        const ITensorInfo             *dst,
                                        const GEMMLowpOutputStageInfo &info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src, dst);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(dst->data_type() == DataType::UNKNOWN,
                                    "CpuGemmLowpOutputStage cannot be used with UNKNOWN output data type.");
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(dst, 1, DataType::QASYMM8, DataType::QASYMM8_SIGNED,
                                                         DataType::QSYMM16);
    ARM_COMPUTE_RETURN_ERROR_ON((info.type != GEMMLowpOutputStageType::QUANTIZE_DOWN) &&
                                (info.type != GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT));

    switch (info.type)
    {
        case GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT:
        {
            switch (dst->data_type())
            {
                case DataType::QASYMM8:
                    return kernels::CpuGemmLowpQuantizeDownInt32ToUint8ScaleByFixedPointKernel::validate(
                        src, bias, dst, info.gemmlowp_min_bound, info.gemmlowp_max_bound);
                case DataType::QASYMM8_SIGNED:
                    return kernels::CpuGemmLowpQuantizeDownInt32ToInt8ScaleByFixedPointKernel::validate(
                        src, bias, dst, info.gemmlowp_min_bound, info.gemmlowp_max_bound);
                case DataType::QSYMM16:
                    return kernels::CpuGemmLowpQuantizeDownInt32ToInt16ScaleByFixedPointKernel::validate(
                        src, bias, dst, info.gemmlowp_min_bound, info.gemmlowp_max_bound);
                default:
                    return ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Unsupported output data type.");
            }
        }
        case GEMMLowpOutputStageType::QUANTIZE_DOWN:
        {
            switch (dst->data_type())
            {
                case DataType::QASYMM8:
                case DataType::QASYMM8_SIGNED:
                    return kernels::CpuGemmLowpQuantizeDownInt32ScaleKernel::validate(src, bias, dst, &info);
                default:
                    return ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Unsupported output data type.");
            }
        }
        default:
            return ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Unsupported GEMMLowpOutputStage type.");
    }
}

void CpuGemmLowpOutputStage::run(ITensorPack &tensors)
{
    NEScheduler::get().schedule_op(_kernel.get(), Window::DimY, _kernel->window(), tensors);
}
} // namespace cpu
} // namespace arm_compute
