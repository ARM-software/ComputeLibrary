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
#include "arm_compute/runtime/CL/functions/CLFlattenLayer.h"

#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/core/Validate.h"

#include "src/core/CL/ICLKernel.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/gpu/cl/operators/ClFlatten.h"

namespace arm_compute
{
struct CLFlattenLayer::Impl
{
    const ICLTensor                   *src{nullptr};
    ICLTensor                         *dst{nullptr};
    std::unique_ptr<opencl::ClFlatten> op{nullptr};
};

CLFlattenLayer::CLFlattenLayer() : _impl(std::make_unique<Impl>())
{
}
CLFlattenLayer::CLFlattenLayer(CLFlattenLayer &&)            = default;
CLFlattenLayer &CLFlattenLayer::operator=(CLFlattenLayer &&) = default;
CLFlattenLayer::~CLFlattenLayer()                            = default;

void CLFlattenLayer::configure(const ICLTensor *input, ICLTensor *output)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, output);
}

void CLFlattenLayer::configure(const CLCompileContext &compile_context, const ICLTensor *input, ICLTensor *output)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    _impl->src = input;
    _impl->dst = output;
    auto_init_if_empty(*output->info(), input->info()->clone()->set_tensor_shape(
                                            misc::shape_calculator::compute_flatten_shape(input->info())));

    _impl->op = std::make_unique<opencl::ClFlatten>();
    _impl->op->configure(compile_context, _impl->src->info(), _impl->dst->info());
}

Status CLFlattenLayer::validate(const ITensorInfo *input, const ITensorInfo *output)
{
    // Checks performed when output is configured
    if (output->total_size() != 0)
    {
        const TensorInfo tensor_info_output =
            input->clone()->set_tensor_shape(misc::shape_calculator::compute_flatten_shape(input));
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(output, &tensor_info_output);
    }
    return opencl::ClFlatten::validate(input, output);
}

void CLFlattenLayer::run()
{
    ITensorPack pack;
    pack.add_tensor(TensorType::ACL_SRC, _impl->src);
    pack.add_tensor(TensorType::ACL_DST, _impl->dst);
    _impl->op->run(pack);
}
} // namespace arm_compute
