/*
 * Copyright (c) 2017-2020 ARM Limited.
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
#include "arm_compute/runtime/CL/functions/CLGEMMLowpOutputStage.h"

#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/kernels/CLGEMMLowpQuantizeDownInt32ScaleByFloatKernel.h"
#include "arm_compute/core/CL/kernels/CLGEMMLowpQuantizeDownInt32ScaleKernel.h"
#include "arm_compute/core/CL/kernels/CLGEMMLowpQuantizeDownInt32ToInt16ScaleByFixedPointKernel.h"
#include "arm_compute/core/CL/kernels/CLGEMMLowpQuantizeDownInt32ToInt8ScaleByFixedPointKernel.h"
#include "arm_compute/core/CL/kernels/CLGEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPointKernel.h"
#include "support/MemorySupport.h"

namespace arm_compute
{
void CLGEMMLowpQuantizeDownInt32ToUint8Scale::configure(const ICLTensor *input, const ICLTensor *bias, ICLTensor *output, int result_offset, int result_mult_int, int result_shift, int min, int max)
{
    GEMMLowpOutputStageInfo info = GEMMLowpOutputStageInfo();
    info.gemmlowp_offset         = result_offset;
    info.gemmlowp_multiplier     = result_mult_int;
    info.gemmlowp_shift          = result_shift;
    info.gemmlowp_min_bound      = min;
    info.gemmlowp_max_bound      = max;

    auto k = arm_compute::support::cpp14::make_unique<CLGEMMLowpQuantizeDownInt32ScaleKernel>();
    k->configure(input, bias, output, &info);
    _kernel = std::move(k);
}

Status CLGEMMLowpQuantizeDownInt32ToUint8Scale::validate(const ITensorInfo *input, const ITensorInfo *bias, const ITensorInfo *output, int min, int max)
{
    GEMMLowpOutputStageInfo info = GEMMLowpOutputStageInfo();
    info.gemmlowp_min_bound      = min;
    info.gemmlowp_max_bound      = max;

    return CLGEMMLowpQuantizeDownInt32ScaleKernel::validate(input, bias, output, &info);
}

void CLGEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPoint::configure(const ICLTensor *input, const ICLTensor *bias, ICLTensor *output,
                                                                    int result_fixedpoint_multiplier, int result_shift, int result_offset_after_shift,
                                                                    int min, int max)
{
    auto k = arm_compute::support::cpp14::make_unique<CLGEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPointKernel>();
    k->configure(input, bias, output, result_fixedpoint_multiplier, result_shift, result_offset_after_shift, min, max);
    _kernel = std::move(k);
}

Status CLGEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPoint::validate(const ITensorInfo *input, const ITensorInfo *bias, const ITensorInfo *output,
                                                                     int min, int max)
{
    return CLGEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPointKernel::validate(input, bias, output, min, max);
}

void CLGEMMLowpQuantizeDownInt32ToInt8ScaleByFixedPoint::configure(const ICLTensor *input, const ICLTensor *bias, ICLTensor *output,
                                                                   int result_fixedpoint_multiplier, int result_shift, int result_offset_after_shift,
                                                                   int min, int max)
{
    auto k = arm_compute::support::cpp14::make_unique<CLGEMMLowpQuantizeDownInt32ToInt8ScaleByFixedPointKernel>();
    k->configure(input, bias, output, result_fixedpoint_multiplier, result_shift, result_offset_after_shift, min, max);
    _kernel = std::move(k);
}

Status CLGEMMLowpQuantizeDownInt32ToInt8ScaleByFixedPoint::validate(const ITensorInfo *input, const ITensorInfo *bias, const ITensorInfo *output,
                                                                    int min, int max)
{
    return CLGEMMLowpQuantizeDownInt32ToInt8ScaleByFixedPointKernel::validate(input, bias, output, min, max);
}

void CLGEMMLowpQuantizeDownInt32ToUint8ScaleByFloat::configure(const ICLTensor *input, const ICLTensor *bias, ICLTensor *output,
                                                               float multiplier, int offset,
                                                               int min, int max)
{
    GEMMLowpOutputStageInfo info  = GEMMLowpOutputStageInfo();
    info.gemmlowp_offset          = offset;
    info.gemmlowp_real_multiplier = multiplier;
    info.gemmlowp_min_bound       = min;
    info.gemmlowp_max_bound       = max;

    auto k = arm_compute::support::cpp14::make_unique<CLGEMMLowpQuantizeDownInt32ScaleByFloatKernel>();
    k->configure(input, bias, output, &info);
    _kernel = std::move(k);
}

Status CLGEMMLowpQuantizeDownInt32ToUint8ScaleByFloat::validate(const ITensorInfo *input, const ITensorInfo *bias, const ITensorInfo *output,
                                                                int min, int max)
{
    GEMMLowpOutputStageInfo info = GEMMLowpOutputStageInfo();
    info.gemmlowp_min_bound      = min;
    info.gemmlowp_max_bound      = max;
    return CLGEMMLowpQuantizeDownInt32ScaleByFloatKernel::validate(input, bias, output, &info);
}

void CLGEMMLowpQuantizeDownInt32ToInt16ScaleByFixedPoint::configure(const ICLTensor *input, const ICLTensor *bias, ICLTensor *output,
                                                                    int result_fixedpoint_multiplier, int result_shift,
                                                                    int min, int max)
{
    auto k = arm_compute::support::cpp14::make_unique<CLGEMMLowpQuantizeDownInt32ToInt16ScaleByFixedPointKernel>();
    k->configure(input, bias, output, result_fixedpoint_multiplier, result_shift, min, max);
    _kernel = std::move(k);
}

Status CLGEMMLowpQuantizeDownInt32ToInt16ScaleByFixedPoint::validate(const ITensorInfo *input, const ITensorInfo *bias, const ITensorInfo *output,
                                                                     int min, int max)
{
    return CLGEMMLowpQuantizeDownInt32ToInt16ScaleByFixedPointKernel::validate(input, bias, output, min, max);
}

void CLGEMMLowpOutputStage::configure(const ICLTensor *input, const ICLTensor *bias, ICLTensor *output, const GEMMLowpOutputStageInfo &info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

    switch(info.type)
    {
        case GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT:
        {
            switch(info.output_data_type)
            {
                case DataType::QASYMM8:
                {
                    auto k = arm_compute::support::cpp14::make_unique<CLGEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPointKernel>();
                    k->configure(input, bias, output, info.gemmlowp_multiplier, info.gemmlowp_shift, info.gemmlowp_offset, info.gemmlowp_min_bound, info.gemmlowp_max_bound);
                    _kernel = std::move(k);
                    break;
                }
                case DataType::QASYMM8_SIGNED:
                {
                    auto k = arm_compute::support::cpp14::make_unique<CLGEMMLowpQuantizeDownInt32ToInt8ScaleByFixedPointKernel>();
                    k->configure(input, bias, output, info.gemmlowp_multiplier, info.gemmlowp_shift, info.gemmlowp_offset, info.gemmlowp_min_bound, info.gemmlowp_max_bound);
                    _kernel = std::move(k);
                    break;
                }
                default:
                    ARM_COMPUTE_ERROR("Unsupported output data type.");
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
                    auto k = arm_compute::support::cpp14::make_unique<CLGEMMLowpQuantizeDownInt32ScaleKernel>();
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
        case GEMMLowpOutputStageType::QUANTIZE_DOWN_FLOAT:
        {
            auto k = arm_compute::support::cpp14::make_unique<CLGEMMLowpQuantizeDownInt32ScaleByFloatKernel>();
            k->configure(input, bias, output, &info);
            _kernel = std::move(k);
            break;
        }
        default:
            ARM_COMPUTE_ERROR("Unsupported GEMMLowpOutputStage type.");
    }
}

Status CLGEMMLowpOutputStage::validate(const ITensorInfo *input, const ITensorInfo *bias, const ITensorInfo *output, const GEMMLowpOutputStageInfo &info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(output);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::QASYMM8, DataType::QASYMM8_SIGNED);

    switch(info.type)
    {
        case GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT:
        {
            switch(output->data_type())
            {
                case DataType::QASYMM8:
                    return CLGEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPointKernel::validate(input, bias, output, info.gemmlowp_min_bound, info.gemmlowp_max_bound);
                case DataType::QASYMM8_SIGNED:
                    return CLGEMMLowpQuantizeDownInt32ToInt8ScaleByFixedPointKernel::validate(input, bias, output, info.gemmlowp_min_bound, info.gemmlowp_max_bound);
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
                {
                    return CLGEMMLowpQuantizeDownInt32ScaleKernel::validate(input, bias, output, &info);
                }
                default:
                    return ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Unsupported output data type.");
            }
        }
        case GEMMLowpOutputStageType::QUANTIZE_DOWN_FLOAT:
        {
            return CLGEMMLowpQuantizeDownInt32ScaleByFloatKernel::validate(input, bias, output, &info);
        }
        default:
            return ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Unsupported GEMMLowpOutputStage type.");
    }
}
} // namespace arm_compute