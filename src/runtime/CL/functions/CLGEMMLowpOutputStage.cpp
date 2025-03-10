/*
 * Copyright (c) 2017-2021, 2024 Arm Limited.
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

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/KernelDescriptors.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"

#include "src/core/CL/ICLKernel.h"
#include "src/gpu/cl/operators/ClGemmLowpOutputStage.h"

#include <algorithm>

namespace arm_compute
{
struct CLGEMMLowpOutputStage::Impl
{
    const ICLTensor                               *src{nullptr};
    const ICLTensor                               *bias{nullptr};
    ICLTensor                                     *dst{nullptr};
    std::unique_ptr<opencl::ClGemmLowpOutputStage> op{nullptr};
    ITensorPack                                    run_pack{};
};

CLGEMMLowpOutputStage::CLGEMMLowpOutputStage() : _impl(std::make_unique<Impl>())
{
}
CLGEMMLowpOutputStage::CLGEMMLowpOutputStage(CLGEMMLowpOutputStage &&)            = default;
CLGEMMLowpOutputStage &CLGEMMLowpOutputStage::operator=(CLGEMMLowpOutputStage &&) = default;
CLGEMMLowpOutputStage::~CLGEMMLowpOutputStage()                                   = default;

void CLGEMMLowpOutputStage::configure(const ICLTensor               *input,
                                      const ICLTensor               *bias,
                                      ICLTensor                     *output,
                                      const GEMMLowpOutputStageInfo &info)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, bias, output, info);
}

void CLGEMMLowpOutputStage::configure(const CLCompileContext        &compile_context,
                                      const ICLTensor               *input,
                                      const ICLTensor               *bias,
                                      ICLTensor                     *output,
                                      const GEMMLowpOutputStageInfo &info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

    _impl->src  = input;
    _impl->bias = bias;
    _impl->dst  = output;

    _impl->op = std::make_unique<opencl::ClGemmLowpOutputStage>();
    _impl->op->configure(compile_context, input->info(), bias != nullptr ? bias->info() : nullptr, output->info(),
                         info);
    _impl->run_pack = {{ACL_SRC, _impl->src}, {ACL_BIAS, _impl->bias}, {ACL_DST, _impl->dst}};
}

Status CLGEMMLowpOutputStage::validate(const ITensorInfo             *input,
                                       const ITensorInfo             *bias,
                                       const ITensorInfo             *output,
                                       const GEMMLowpOutputStageInfo &info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_DYNAMIC_SHAPE(input, bias, output);
    return opencl::ClGemmLowpOutputStage::validate(input, bias, output, info);
}

void CLGEMMLowpOutputStage::run()
{
    _impl->op->run(_impl->run_pack);
}
} // namespace arm_compute
