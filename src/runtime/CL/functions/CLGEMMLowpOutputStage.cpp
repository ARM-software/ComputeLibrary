/*
 * Copyright (c) 2017-2021 Arm Limited.
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
#include "arm_compute/runtime/CL/CLScheduler.h"

#include "src/core/gpu/cl/kernels/ClGemmLowpQuantizeDownInt32ScaleByFixedPointKernel.h"
#include "src/core/gpu/cl/kernels/ClGemmLowpQuantizeDownInt32ScaleByFloatKernel.h"
#include "src/core/gpu/cl/kernels/ClGemmLowpQuantizeDownInt32ScaleKernel.h"

#include <algorithm>

namespace arm_compute
{
CLGEMMLowpOutputStage::CLGEMMLowpOutputStage()
    : _kernel(nullptr), _input(nullptr), _bias(nullptr), _output(nullptr)
{
}
CLGEMMLowpOutputStage::CLGEMMLowpOutputStage(CLGEMMLowpOutputStage &&) = default;
CLGEMMLowpOutputStage &CLGEMMLowpOutputStage::operator=(CLGEMMLowpOutputStage &&) = default;
CLGEMMLowpOutputStage::~CLGEMMLowpOutputStage()                                   = default;

void CLGEMMLowpOutputStage::configure(const ICLTensor *input, const ICLTensor *bias, ICLTensor *output, const GEMMLowpOutputStageInfo &info)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, bias, output, info);
}

void CLGEMMLowpOutputStage::configure(const CLCompileContext &compile_context, const ICLTensor *input, const ICLTensor *bias, ICLTensor *output, const GEMMLowpOutputStageInfo &info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

    _input  = input;
    _bias   = bias;
    _output = output;

    switch(info.type)
    {
        case GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT:
        {
            auto k = std::make_unique<opencl::kernels::ClGemmLowpQuantizeDownInt32ScaleByFixedPointKernel>();
            k->configure(compile_context, input->info(), bias != nullptr ? bias->info() : nullptr, output->info(), &info);
            _kernel = std::move(k);
            break;
        }
        case GEMMLowpOutputStageType::QUANTIZE_DOWN:
        {
            auto k = std::make_unique<opencl::kernels::ClGemmLowpQuantizeDownInt32ScaleKernel>();
            k->configure(compile_context, input->info(), bias != nullptr ? bias->info() : nullptr, output->info(), &info);
            _kernel = std::move(k);
            break;
        }
        case GEMMLowpOutputStageType::QUANTIZE_DOWN_FLOAT:
        {
            auto k = std::make_unique<opencl::kernels::ClGemmLowpQuantizeDownInt32ScaleByFloatKernel>();
            k->configure(compile_context, input->info(), bias != nullptr ? bias->info() : nullptr, output->info(), &info);
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
            return opencl::kernels::ClGemmLowpQuantizeDownInt32ScaleByFixedPointKernel::validate(input, bias, output, &info);
        case GEMMLowpOutputStageType::QUANTIZE_DOWN:
            return opencl::kernels::ClGemmLowpQuantizeDownInt32ScaleKernel::validate(input, bias, output, &info);
        case GEMMLowpOutputStageType::QUANTIZE_DOWN_FLOAT:
            return opencl::kernels::ClGemmLowpQuantizeDownInt32ScaleByFloatKernel::validate(input, bias, output, &info);
        default:
            return ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Unsupported GEMMLowpOutputStage type.");
    }
}

void CLGEMMLowpOutputStage::run()
{
    ITensorPack pack{ { ACL_SRC, _input }, { ACL_BIAS, _bias }, { ACL_DST, _output } };
    CLScheduler::get().enqueue_op(*_kernel, pack, true);
}
} // namespace arm_compute
