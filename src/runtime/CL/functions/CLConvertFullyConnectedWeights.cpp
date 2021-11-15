/*
 * Copyright (c) 2018-2021 Arm Limited.
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
#include "arm_compute/runtime/CL/functions/CLConvertFullyConnectedWeights.h"

#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "src/core/CL/ICLKernel.h"
#include "src/gpu/cl/operators/ClConvertFullyConnectedWeights.h"

#include "src/common/utils/Log.h"

namespace arm_compute
{
struct CLConvertFullyConnectedWeights::Impl
{
    const ICLTensor                                        *src{ nullptr };
    ICLTensor                                              *dst{ nullptr };
    std::unique_ptr<opencl::ClConvertFullyConnectedWeights> op{ nullptr };
};
CLConvertFullyConnectedWeights::CLConvertFullyConnectedWeights()
    : _impl(std::make_unique<Impl>())
{
}
CLConvertFullyConnectedWeights::~CLConvertFullyConnectedWeights() = default;

void CLConvertFullyConnectedWeights::configure(const ICLTensor *input, ICLTensor *output, const TensorShape &original_input_shape,
                                               DataLayout data_layout)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, output, original_input_shape, data_layout);
}

void CLConvertFullyConnectedWeights::configure(const CLCompileContext &compile_context, const ICLTensor *input, ICLTensor *output, const TensorShape &original_input_shape,
                                               DataLayout data_layout)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_LOG_PARAMS(input, output, original_input_shape, data_layout);
    _impl->src = input;
    _impl->dst = output;
    _impl->op  = std::make_unique<opencl::ClConvertFullyConnectedWeights>();
    _impl->op->configure(compile_context, _impl->src->info(), _impl->dst->info(), original_input_shape, data_layout);
}

Status CLConvertFullyConnectedWeights::validate(const ITensorInfo *input, const ITensorInfo *output, const TensorShape &original_input_shape,
                                                DataLayout data_layout)
{
    return opencl::ClConvertFullyConnectedWeights::validate(input, output, original_input_shape, data_layout);
}

void CLConvertFullyConnectedWeights::run()
{
    ITensorPack pack;
    pack.add_tensor(TensorType::ACL_SRC, _impl->src);
    pack.add_tensor(TensorType::ACL_DST, _impl->dst);
    _impl->op->run(pack);
}

} // namespace arm_compute
