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
#include "arm_compute/runtime/CL/functions/CLConcatenateLayer.h"

#include "arm_compute/core/CL/ICLTensor.h"
#include "src/core/CL/ICLKernel.h"
#include "src/gpu/cl/operators/ClConcatenate.h"

namespace arm_compute
{
struct CLConcatenateLayer::Impl
{
    std::vector<const ICLTensor *>         srcs{};
    ICLTensor                             *dst{ nullptr };
    unsigned int                           num_inputs{ 0 };
    unsigned int                           axis{ 0 };
    std::unique_ptr<opencl::ClConcatenate> op{ nullptr };
};

CLConcatenateLayer::CLConcatenateLayer()
    : _impl(std::make_unique<Impl>())
{
}

CLConcatenateLayer::CLConcatenateLayer(CLConcatenateLayer &&) = default;

CLConcatenateLayer &CLConcatenateLayer::operator=(CLConcatenateLayer &&) = default;

CLConcatenateLayer::~CLConcatenateLayer() = default;

void CLConcatenateLayer::configure(std::vector<const ICLTensor *> &inputs_vector, ICLTensor *output, size_t axis)
{
    configure(CLKernelLibrary::get().get_compile_context(), inputs_vector, output, axis);
}

void CLConcatenateLayer::configure(const CLCompileContext &compile_context, std::vector<const ICLTensor *> &inputs_vector, ICLTensor *output, size_t axis)
{
    ARM_COMPUTE_ERROR_ON(output == nullptr);

    _impl->srcs       = inputs_vector;
    _impl->dst        = output;
    _impl->axis       = axis;
    _impl->num_inputs = inputs_vector.size();
    _impl->op         = std::make_unique<opencl::ClConcatenate>();

    std::vector<ITensorInfo *> inputs_vector_info;
    for(unsigned int i = 0; i < inputs_vector.size(); ++i)
    {
        ARM_COMPUTE_ERROR_ON_NULLPTR(inputs_vector.at(i));
        inputs_vector_info.emplace_back(inputs_vector.at(i)->info());
    }
    _impl->op->configure(compile_context, inputs_vector_info, _impl->dst->info(), axis);
}

Status CLConcatenateLayer::validate(const std::vector<const ITensorInfo *> &inputs_vector, const ITensorInfo *output, size_t axis)
{
    return opencl::ClConcatenate::validate(inputs_vector, output, axis);
}

void CLConcatenateLayer::run()
{
    ITensorPack pack;
    for(unsigned i = 0; i < _impl->num_inputs; ++i)
    {
        pack.add_tensor(TensorType::ACL_SRC_VEC + i, _impl->srcs.at(i));
    }
    pack.add_tensor(TensorType::ACL_DST, _impl->dst);

    _impl->op->run(pack);
}
} // namespace arm_compute
