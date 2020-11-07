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
#include "arm_compute/runtime/CL/functions/CLGEMMLowpOutputStage.h"

#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Types.h"
#include "src/core/CL/kernels/CLGEMMLowpQuantizeDownInt32ScaleByFixedPointKernel.h"
#include "src/core/CL/kernels/CLGEMMLowpQuantizeDownInt32ScaleByFloatKernel.h"
#include "src/core/CL/kernels/CLGEMMLowpQuantizeDownInt32ScaleKernel.h"
#include "support/MemorySupport.h"

#include <algorithm>

namespace arm_compute
{
void CLGEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPoint::configure(const ICLTensor *input, const ICLTensor *bias, ICLTensor *output,
                                                                    int result_fixedpoint_multiplier, int result_shift, int result_offset_after_shift,
                                                                    int min, int max)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, bias, output, result_fixedpoint_multiplier, result_shift, result_offset_after_shift, min, max);
}

void CLGEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPoint::configure(const CLCompileContext &compile_context, const ICLTensor *input, const ICLTensor *bias, ICLTensor *output,
                                                                    int result_fixedpoint_multiplier, int result_shift, int result_offset_after_shift,
                                                                    int min, int max)
{
    GEMMLowpOutputStageInfo info{};
    info.gemmlowp_multiplier = result_fixedpoint_multiplier;
    info.gemmlowp_shift      = result_shift;
    info.gemmlowp_offset     = result_offset_after_shift;
    info.gemmlowp_min_bound  = min;
    info.gemmlowp_max_bound  = max;
    info.output_data_type    = DataType::QASYMM8;
    auto k                   = arm_compute::support::cpp14::make_unique<CLGEMMLowpQuantizeDownInt32ScaleByFixedPointKernel>();
    k->configure(compile_context, input, bias, output, &info);
    _kernel = std::move(k);
}

Status CLGEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPoint::validate(const ITensorInfo *input, const ITensorInfo *bias, const ITensorInfo *output,
                                                                     int min, int max)
{
    GEMMLowpOutputStageInfo info{};
    info.gemmlowp_min_bound = min;
    info.gemmlowp_max_bound = max;
    info.output_data_type   = DataType::QASYMM8;
    return CLGEMMLowpQuantizeDownInt32ScaleByFixedPointKernel::validate(input, bias, output, &info);
}

void CLGEMMLowpQuantizeDownInt32ToInt8ScaleByFixedPoint::configure(const ICLTensor *input, const ICLTensor *bias, ICLTensor *output,
                                                                   int result_fixedpoint_multiplier, int result_shift, int result_offset_after_shift,
                                                                   int min, int max)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, bias, output, result_fixedpoint_multiplier, result_shift, result_offset_after_shift, min, max);
}

void CLGEMMLowpQuantizeDownInt32ToInt8ScaleByFixedPoint::configure(const CLCompileContext &compile_context, const ICLTensor *input, const ICLTensor *bias, ICLTensor *output,
                                                                   int result_fixedpoint_multiplier, int result_shift, int result_offset_after_shift,
                                                                   int min, int max)
{
    GEMMLowpOutputStageInfo info{};
    info.gemmlowp_multiplier = result_fixedpoint_multiplier;
    info.gemmlowp_shift      = result_shift;
    info.gemmlowp_offset     = result_offset_after_shift;
    info.gemmlowp_min_bound  = min;
    info.gemmlowp_max_bound  = max;
    info.output_data_type    = DataType::QASYMM8_SIGNED;
    auto k                   = arm_compute::support::cpp14::make_unique<CLGEMMLowpQuantizeDownInt32ScaleByFixedPointKernel>();
    k->configure(compile_context, input, bias, output, &info);
    _kernel = std::move(k);
}

Status CLGEMMLowpQuantizeDownInt32ToInt8ScaleByFixedPoint::validate(const ITensorInfo *input, const ITensorInfo *bias, const ITensorInfo *output,
                                                                    int min, int max)
{
    GEMMLowpOutputStageInfo info{};
    info.gemmlowp_min_bound = min;
    info.gemmlowp_max_bound = max;
    info.output_data_type   = DataType::QASYMM8_SIGNED;
    return CLGEMMLowpQuantizeDownInt32ScaleByFixedPointKernel::validate(input, bias, output, &info);
}

void CLGEMMLowpQuantizeDownInt32ToInt16ScaleByFixedPoint::configure(const ICLTensor *input, const ICLTensor *bias, ICLTensor *output,
                                                                    int result_fixedpoint_multiplier, int result_shift,
                                                                    int min, int max)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, bias, output, result_fixedpoint_multiplier, result_shift, min, max);
}

void CLGEMMLowpQuantizeDownInt32ToInt16ScaleByFixedPoint::configure(const CLCompileContext &compile_context, const ICLTensor *input, const ICLTensor *bias, ICLTensor *output,
                                                                    int result_fixedpoint_multiplier, int result_shift,
                                                                    int min, int max)
{
    GEMMLowpOutputStageInfo info{};
    info.gemmlowp_multiplier = result_fixedpoint_multiplier;
    info.gemmlowp_shift      = result_shift;
    info.gemmlowp_min_bound  = min;
    info.gemmlowp_max_bound  = max;
    info.output_data_type    = DataType::QSYMM16;
    auto k                   = arm_compute::support::cpp14::make_unique<CLGEMMLowpQuantizeDownInt32ScaleByFixedPointKernel>();
    k->configure(compile_context, input, bias, output, &info);
    _kernel = std::move(k);
}

Status CLGEMMLowpQuantizeDownInt32ToInt16ScaleByFixedPoint::validate(const ITensorInfo *input, const ITensorInfo *bias, const ITensorInfo *output,
                                                                     int min, int max)
{
    GEMMLowpOutputStageInfo info{};
    info.gemmlowp_min_bound = min;
    info.gemmlowp_max_bound = max;
    info.output_data_type   = DataType::QSYMM16;
    return CLGEMMLowpQuantizeDownInt32ScaleByFixedPointKernel::validate(input, bias, output, &info);
}

void CLGEMMLowpOutputStage::configure(const ICLTensor *input, const ICLTensor *bias, ICLTensor *output, const GEMMLowpOutputStageInfo &info)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, bias, output, info);
}

void CLGEMMLowpOutputStage::configure(const CLCompileContext &compile_context, const ICLTensor *input, const ICLTensor *bias, ICLTensor *output, const GEMMLowpOutputStageInfo &info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

    switch(info.type)
    {
        case GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT:
        {
            auto k = arm_compute::support::cpp14::make_unique<CLGEMMLowpQuantizeDownInt32ScaleByFixedPointKernel>();
            k->configure(compile_context, input, bias, output, &info);
            _kernel = std::move(k);
            break;
        }
        case GEMMLowpOutputStageType::QUANTIZE_DOWN:
        {
            auto k = arm_compute::support::cpp14::make_unique<CLGEMMLowpQuantizeDownInt32ScaleKernel>();
            k->configure(compile_context, input, bias, output, &info);
            _kernel = std::move(k);
            break;
        }
        case GEMMLowpOutputStageType::QUANTIZE_DOWN_FLOAT:
        {
            auto k = arm_compute::support::cpp14::make_unique<CLGEMMLowpQuantizeDownInt32ScaleByFloatKernel>();
            k->configure(compile_context, input, bias, output, &info);
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
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::QASYMM8, DataType::QASYMM8_SIGNED, DataType::QSYMM16);

    switch(info.type)
    {
        case GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT:
            return CLGEMMLowpQuantizeDownInt32ScaleByFixedPointKernel::validate(input, bias, output, &info);
        case GEMMLowpOutputStageType::QUANTIZE_DOWN:
            return CLGEMMLowpQuantizeDownInt32ScaleKernel::validate(input, bias, output, &info);
        case GEMMLowpOutputStageType::QUANTIZE_DOWN_FLOAT:
            return CLGEMMLowpQuantizeDownInt32ScaleByFloatKernel::validate(input, bias, output, &info);
        default:
            return ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Unsupported GEMMLowpOutputStage type.");
    }
}
} // namespace arm_compute
