/*
 * Copyright (c) 2017-2020 Arm Limited.
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
#include "arm_compute/runtime/NEON/functions/NEGEMMLowpOutputStage.h"

#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Validate.h"
#include "src/core/NEON/kernels/NEGEMMLowpQuantizeDownInt32ScaleKernel.h"
#include "src/core/NEON/kernels/NEGEMMLowpQuantizeDownInt32ToInt16ScaleByFixedPointKernel.h"
#include "src/core/NEON/kernels/NEGEMMLowpQuantizeDownInt32ToInt8ScaleByFixedPointKernel.h"
#include "src/core/NEON/kernels/NEGEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPointKernel.h"
#include "support/MemorySupport.h"

namespace arm_compute
{
NEGEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPoint::~NEGEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPoint() = default;

void NEGEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPoint::configure(const ITensor *input, const ITensor *bias, ITensor *output, int result_fixedpoint_multiplier, int result_shift,
                                                                    int result_offset_after_shift, int min, int max)
{
    auto k = arm_compute::support::cpp14::make_unique<NEGEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPointKernel>();
    k->configure(input, bias, output, result_fixedpoint_multiplier, result_shift, result_offset_after_shift, min, max);
    _kernel = std::move(k);
}

Status NEGEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPoint::validate(const ITensorInfo *input, const ITensorInfo *bias, const ITensorInfo *output, int min, int max)
{
    return NEGEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPointKernel::validate(input, bias, output, min, max);
}

NEGEMMLowpQuantizeDownInt32ToInt8ScaleByFixedPoint::~NEGEMMLowpQuantizeDownInt32ToInt8ScaleByFixedPoint() = default;

void NEGEMMLowpQuantizeDownInt32ToInt8ScaleByFixedPoint::configure(const ITensor *input, const ITensor *bias, ITensor *output, int result_fixedpoint_multiplier, int result_shift,
                                                                   int result_offset_after_shift, int min, int max)
{
    auto k = arm_compute::support::cpp14::make_unique<NEGEMMLowpQuantizeDownInt32ToInt8ScaleByFixedPointKernel>();
    k->configure(input, bias, output, result_fixedpoint_multiplier, result_shift, result_offset_after_shift, min, max);
    _kernel = std::move(k);
}

Status NEGEMMLowpQuantizeDownInt32ToInt8ScaleByFixedPoint::validate(const ITensorInfo *input, const ITensorInfo *bias, const ITensorInfo *output, int min, int max)
{
    return NEGEMMLowpQuantizeDownInt32ToInt8ScaleByFixedPointKernel::validate(input, bias, output, min, max);
}

NEGEMMLowpQuantizeDownInt32ToInt16ScaleByFixedPoint::~NEGEMMLowpQuantizeDownInt32ToInt16ScaleByFixedPoint() = default;

void NEGEMMLowpQuantizeDownInt32ToInt16ScaleByFixedPoint::configure(const ITensor *input, const ITensor *bias, ITensor *output, int result_fixedpoint_multiplier, int result_shift, int min, int max)
{
    auto k = arm_compute::support::cpp14::make_unique<NEGEMMLowpQuantizeDownInt32ToInt16ScaleByFixedPointKernel>();
    k->configure(input, bias, output, result_fixedpoint_multiplier, result_shift, min, max);
    _kernel = std::move(k);
}

Status NEGEMMLowpQuantizeDownInt32ToInt16ScaleByFixedPoint::validate(const ITensorInfo *input, const ITensorInfo *bias, const ITensorInfo *output, int min, int max)
{
    return NEGEMMLowpQuantizeDownInt32ToInt16ScaleByFixedPointKernel::validate(input, bias, output, min, max);
}

NEGEMMLowpOutputStage::~NEGEMMLowpOutputStage() = default;

void NEGEMMLowpOutputStage::configure(const ITensor *input, const ITensor *bias, ITensor *output, const GEMMLowpOutputStageInfo &info)
{
    // Perform validate step
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_ERROR_THROW_ON(NEGEMMLowpOutputStage::validate(input->info(), bias != nullptr ? bias->info() : nullptr, output->info(), info));

    switch(info.type)
    {
        case GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT:
        {
            switch(info.output_data_type)
            {
                case DataType::QASYMM8:
                {
                    auto k = arm_compute::support::cpp14::make_unique<NEGEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPointKernel>();
                    k->configure(input, bias, output, info.gemmlowp_multiplier, info.gemmlowp_shift, info.gemmlowp_offset, info.gemmlowp_min_bound, info.gemmlowp_max_bound);
                    _kernel = std::move(k);
                    break;
                }
                case DataType::QASYMM8_SIGNED:
                {
                    auto k = arm_compute::support::cpp14::make_unique<NEGEMMLowpQuantizeDownInt32ToInt8ScaleByFixedPointKernel>();
                    k->configure(input, bias, output, info.gemmlowp_multiplier, info.gemmlowp_shift, info.gemmlowp_offset, info.gemmlowp_min_bound, info.gemmlowp_max_bound);
                    _kernel = std::move(k);
                    break;
                }
                case DataType::QSYMM16:
                {
                    auto k = arm_compute::support::cpp14::make_unique<NEGEMMLowpQuantizeDownInt32ToInt16ScaleByFixedPointKernel>();
                    k->configure(input, bias, output, info.gemmlowp_multiplier, info.gemmlowp_shift, info.gemmlowp_min_bound, info.gemmlowp_max_bound);
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
            switch(info.output_data_type)
            {
                case DataType::QASYMM8:
                case DataType::QASYMM8_SIGNED:
                {
                    auto k = arm_compute::support::cpp14::make_unique<NEGEMMLowpQuantizeDownInt32ScaleKernel>();
                    k->configure(input, bias, output, &info);
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

Status NEGEMMLowpOutputStage::validate(const ITensorInfo *input, const ITensorInfo *bias, const ITensorInfo *output, const GEMMLowpOutputStageInfo &info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(output);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(output->data_type() == DataType::UNKNOWN, "NEGEMMLowpQuantizeDownScaleByFixedPoint cannot be used with UNKNOWN output data type.");
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::QASYMM8, DataType::QASYMM8_SIGNED, DataType::QSYMM16);

    ARM_COMPUTE_RETURN_ERROR_ON((info.type != GEMMLowpOutputStageType::QUANTIZE_DOWN) && (info.type != GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT));

    switch(info.type)
    {
        case GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT:
        {
            switch(output->data_type())
            {
                case DataType::QASYMM8:
                    return NEGEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPointKernel::validate(input, bias, output, info.gemmlowp_min_bound, info.gemmlowp_max_bound);
                case DataType::QASYMM8_SIGNED:
                    return NEGEMMLowpQuantizeDownInt32ToInt8ScaleByFixedPointKernel::validate(input, bias, output, info.gemmlowp_min_bound, info.gemmlowp_max_bound);
                case DataType::QSYMM16:
                    return NEGEMMLowpQuantizeDownInt32ToInt16ScaleByFixedPointKernel::validate(input, bias, output, info.gemmlowp_min_bound, info.gemmlowp_max_bound);
                default:
                    return ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Unsupported output data type.");
            }
        }
        case GEMMLowpOutputStageType::QUANTIZE_DOWN:
        {
            switch(output->data_type())
            {
                case DataType::QASYMM8:
                case DataType::QASYMM8_SIGNED:
                    return NEGEMMLowpQuantizeDownInt32ScaleKernel::validate(input, bias, output, &info);
                default:
                    return ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Unsupported output data type.");
            }
        }
        default:
            return ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Unsupported GEMMLowpOutputStage type.");
    }
}
} // namespace arm_compute
