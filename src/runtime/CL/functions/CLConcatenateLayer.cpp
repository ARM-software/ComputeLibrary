/*
 * Copyright (c) 2018-2020 Arm Limited.
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

#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "src/core/CL/kernels/CLDepthConcatenateLayerKernel.h"
#include "src/core/CL/kernels/CLHeightConcatenateLayerKernel.h"
#include "src/core/CL/kernels/CLWidthConcatenate2TensorsKernel.h"
#include "src/core/CL/kernels/CLWidthConcatenate4TensorsKernel.h"
#include "src/core/CL/kernels/CLWidthConcatenateLayerKernel.h"

#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "src/core/CL/kernels/CLBatchConcatenateLayerKernel.h"
#include "src/core/helpers/AutoConfiguration.h"

namespace arm_compute
{
namespace experimental
{
CLConcatenation::CLConcatenation()
    : _concat_kernels(),
      _num_inputs(0),
      _axis(Window::DimX)
{
}

void CLConcatenation::configure(const CLCompileContext &compile_context, const std::vector<ITensorInfo *> &inputs_vector, ITensorInfo *output, size_t axis)
{
    ARM_COMPUTE_ERROR_ON(output == nullptr);
    _axis       = axis;
    _num_inputs = inputs_vector.size();

    TensorShape                      output_shape = arm_compute::misc::shape_calculator::calculate_concatenate_shape(inputs_vector, _axis);
    std::vector<const ITensorInfo *> const_inputs_vector(inputs_vector.size());
    std::transform(inputs_vector.begin(), inputs_vector.end(), const_inputs_vector.begin(), [](ITensorInfo * t)
    {
        ARM_COMPUTE_ERROR_ON_NULLPTR(t);
        return t;
    });

    // Output auto inizialitation if not yet initialized
    auto_init_if_empty(*output, output_shape, 1, inputs_vector[0]->data_type());
    ARM_COMPUTE_ERROR_THROW_ON(CLConcatenateLayer::validate(const_inputs_vector, output, axis));

    unsigned int offset = 0;
    switch(_axis)
    {
        case Window::DimX:
        {
            switch(_num_inputs)
            {
                case 2:
                {
                    // Configure WidthConcatenate2Tensors kernel
                    auto kernel = std::make_unique<CLWidthConcatenate2TensorsKernel>();
                    kernel->configure(compile_context, inputs_vector.at(0), inputs_vector.at(1), output);
                    _concat_kernels.emplace_back(std::move(kernel));
                    break;
                }
                case 4:
                {
                    // Configure WidthConcatenate4Tensors kernel
                    auto kernel = std::make_unique<CLWidthConcatenate4TensorsKernel>();
                    kernel->configure(compile_context, inputs_vector.at(0), inputs_vector.at(1), inputs_vector.at(2), inputs_vector.at(3), output);
                    _concat_kernels.emplace_back(std::move(kernel));
                    break;
                }
                default:
                {
                    // Configure generic case WidthConcatenate kernels
                    for(unsigned int i = 0; i < _num_inputs; ++i)
                    {
                        auto kernel = std::make_unique<CLWidthConcatenateLayerKernel>();
                        kernel->configure(compile_context, inputs_vector.at(i), offset, output);
                        offset += inputs_vector.at(i)->dimension(_axis);
                        _concat_kernels.emplace_back(std::move(kernel));
                    }
                    break;
                }
            }
            break;
        }
        case Window::DimY:
        {
            for(unsigned int i = 0; i < _num_inputs; ++i)
            {
                auto kernel = std::make_unique<CLHeightConcatenateLayerKernel>();
                kernel->configure(compile_context, inputs_vector.at(i), offset, output);
                offset += inputs_vector.at(i)->dimension(_axis);
                _concat_kernels.emplace_back(std::move(kernel));
            }
            break;
        }
        case Window::DimZ:
        {
            for(unsigned int i = 0; i < _num_inputs; ++i)
            {
                auto kernel = std::make_unique<CLDepthConcatenateLayerKernel>();
                kernel->configure(compile_context, inputs_vector.at(i), offset, output);
                offset += inputs_vector.at(i)->dimension(_axis);
                _concat_kernels.emplace_back(std::move(kernel));
            }
            break;
        }
        case 3:
        {
            for(unsigned int i = 0; i < _num_inputs; ++i)
            {
                auto kernel = std::make_unique<CLBatchConcatenateLayerKernel>();
                kernel->configure(compile_context, inputs_vector.at(i), offset, output);
                offset += inputs_vector.at(i)->dimension(_axis);
                _concat_kernels.emplace_back(std::move(kernel));
            }
            break;
        }
        default:
            ARM_COMPUTE_ERROR("Axis not supported");
    }
}

Status CLConcatenation::validate(const std::vector<const ITensorInfo *> &inputs_vector, const ITensorInfo *output, size_t axis)
{
    ARM_COMPUTE_RETURN_ERROR_ON(output == nullptr);
    const unsigned int num_inputs = inputs_vector.size();

    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(output);
    ARM_COMPUTE_RETURN_ERROR_ON(num_inputs < 2);

    unsigned int offset = 0;
    switch(axis)
    {
        case Window::DimX:
        {
            switch(num_inputs)
            {
                case 2:
                    // Validate WidthConcatenate2Tensors kernels if there are 2 inputs
                    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(inputs_vector[0], inputs_vector[1]);
                    ARM_COMPUTE_RETURN_ON_ERROR(CLWidthConcatenate2TensorsKernel::validate(inputs_vector[0], inputs_vector[1], output));
                    break;
                case 4:
                    // Validate WidthConcatenate4Tensors kernels if there are 4 inputs
                    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(inputs_vector[0], inputs_vector[1], inputs_vector[2], inputs_vector[3]);
                    ARM_COMPUTE_RETURN_ON_ERROR(CLWidthConcatenate4TensorsKernel::validate(inputs_vector[0], inputs_vector[1], inputs_vector[2], inputs_vector[3], output));
                    break;
                default:
                    // Validate generic case of WidthConcatenate kernel
                    for(const auto &input : inputs_vector)
                    {
                        ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input);
                        ARM_COMPUTE_RETURN_ON_ERROR(CLWidthConcatenateLayerKernel::validate(input, offset, output));
                        offset += input->dimension(axis);
                    }
                    break;
            }
            break;
        }
        case Window::DimY:
        {
            for(const auto &input : inputs_vector)
            {
                ARM_COMPUTE_RETURN_ON_ERROR(CLHeightConcatenateLayerKernel::validate(input, offset, output));
                offset += input->dimension(axis);
            }
            break;
        }
        case Window::DimZ:
        {
            for(const auto &input : inputs_vector)
            {
                ARM_COMPUTE_RETURN_ON_ERROR(CLDepthConcatenateLayerKernel::validate(input, offset, output));
                offset += input->dimension(axis);
            }
            break;
        }
        case 3:
        {
            for(const auto &input : inputs_vector)
            {
                ARM_COMPUTE_RETURN_ON_ERROR(CLBatchConcatenateLayerKernel::validate(input, offset, output));
                offset += input->dimension(axis);
            }
            break;
        }
        default:
            ARM_COMPUTE_ERROR("Axis not supported");
    }

    if(output->total_size() != 0)
    {
        TensorShape output_shape = arm_compute::misc::shape_calculator::calculate_concatenate_shape(inputs_vector, axis);
        ARM_COMPUTE_RETURN_ERROR_ON(output_shape.total_size() != output->tensor_shape().total_size());
    }

    return Status{};
}

void CLConcatenation::run(ITensorPack &tensors)
{
    if(tensors.empty())
    {
        ARM_COMPUTE_ERROR("No inputs provided");
    }

    if(static_cast<int>(tensors.size()) - 1 != static_cast<int>(_num_inputs))
    {
        ARM_COMPUTE_ERROR("Configured with different number of inputs");
    }

    if(_axis == Window::DimX && (_num_inputs == 2 || _num_inputs == 4))
    {
        ARM_COMPUTE_ERROR_ON(_concat_kernels.empty());
        CLScheduler::get().enqueue_op(*_concat_kernels.at(0), tensors, true);
    }
    else
    {
        int i = 0;
        for(auto &k : _concat_kernels)
        {
            ITensorPack pack;
            pack.add_tensor(TensorType::ACL_SRC, tensors.get_const_tensor(ACL_SRC_VEC + i));
            pack.add_tensor(TensorType::ACL_DST, tensors.get_tensor(ACL_DST));
            CLScheduler::get().enqueue_op(*k, pack, true);
            ++i;
        }
    }
}
} // namespace experimental

struct CLConcatenateLayer::Impl
{
    std::vector<const ICLTensor *>                 srcs{};
    ICLTensor                                     *dst{ nullptr };
    unsigned int                                   num_inputs{ 0 };
    unsigned int                                   axis{ 0 };
    std::unique_ptr<experimental::CLConcatenation> op{ nullptr };
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
    _impl->op         = std::make_unique<experimental::CLConcatenation>();

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
    return experimental::CLConcatenation::validate(inputs_vector, output, axis);
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
