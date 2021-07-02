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
#include "src/runtime/gpu/cl/operators/ClConcatenate.h"

#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/runtime/CL/CLScheduler.h"

#include "src/core/gpu/cl/kernels/ClBatchConcatenateKernel.h"
#include "src/core/gpu/cl/kernels/ClDepthConcatenateKernel.h"
#include "src/core/gpu/cl/kernels/ClHeightConcatenateKernel.h"
#include "src/core/gpu/cl/kernels/ClWidthConcatenate2TensorsKernel.h"
#include "src/core/gpu/cl/kernels/ClWidthConcatenate4TensorsKernel.h"
#include "src/core/gpu/cl/kernels/ClWidthConcatenateKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "src/core/helpers/AutoConfiguration.h"

namespace arm_compute
{
namespace opencl
{
void ClConcatenate::configure(const CLCompileContext &compile_context, const std::vector<ITensorInfo *> &src_vector, ITensorInfo *dst, size_t axis)
{
    ARM_COMPUTE_ERROR_ON(dst == nullptr);
    _axis       = axis;
    _num_inputs = src_vector.size();

    TensorShape                      dst_shape = arm_compute::misc::shape_calculator::calculate_concatenate_shape(src_vector, _axis);
    std::vector<const ITensorInfo *> const_src_vector(src_vector.size());
    std::transform(src_vector.begin(), src_vector.end(), const_src_vector.begin(), [](ITensorInfo * t)
    {
        ARM_COMPUTE_ERROR_ON_NULLPTR(t);
        return t;
    });

    // dst auto inizialitation if not yet initialized
    auto_init_if_empty(*dst, dst_shape, 1, src_vector[0]->data_type());
    ARM_COMPUTE_ERROR_THROW_ON(ClConcatenate::validate(const_src_vector, dst, axis));

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
                    auto kernel = std::make_unique<kernels::ClWidthConcatenate2TensorsKernel>();
                    kernel->configure(compile_context, src_vector.at(0), src_vector.at(1), dst);
                    _concat_kernels.emplace_back(std::move(kernel));
                    break;
                }
                case 4:
                {
                    // Configure WidthConcatenate4Tensors kernel
                    auto kernel = std::make_unique<kernels::ClWidthConcatenate4TensorsKernel>();
                    kernel->configure(compile_context, src_vector.at(0), src_vector.at(1), src_vector.at(2), src_vector.at(3), dst);
                    _concat_kernels.emplace_back(std::move(kernel));
                    break;
                }
                default:
                {
                    // Configure generic case WidthConcatenate kernels
                    for(unsigned int i = 0; i < _num_inputs; ++i)
                    {
                        auto kernel = std::make_unique<kernels::ClWidthConcatenateKernel>();
                        kernel->configure(compile_context, src_vector.at(i), offset, dst);
                        offset += src_vector.at(i)->dimension(_axis);
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
                auto kernel = std::make_unique<kernels::ClHeightConcatenateKernel>();
                kernel->configure(compile_context, src_vector.at(i), offset, dst);
                offset += src_vector.at(i)->dimension(_axis);
                _concat_kernels.emplace_back(std::move(kernel));
            }
            break;
        }
        case Window::DimZ:
        {
            for(unsigned int i = 0; i < _num_inputs; ++i)
            {
                auto kernel = std::make_unique<kernels::ClDepthConcatenateKernel>();
                kernel->configure(compile_context, src_vector.at(i), offset, dst);
                offset += src_vector.at(i)->dimension(_axis);
                _concat_kernels.emplace_back(std::move(kernel));
            }
            break;
        }
        case 3:
        {
            for(unsigned int i = 0; i < _num_inputs; ++i)
            {
                auto kernel = std::make_unique<kernels::ClBatchConcatenateKernel>();
                kernel->configure(compile_context, src_vector.at(i), offset, dst);
                offset += src_vector.at(i)->dimension(_axis);
                _concat_kernels.emplace_back(std::move(kernel));
            }
            break;
        }
        default:
            ARM_COMPUTE_ERROR("Axis not supported");
    }
}

Status ClConcatenate::validate(const std::vector<const ITensorInfo *> &src_vector, const ITensorInfo *dst, size_t axis)
{
    ARM_COMPUTE_RETURN_ERROR_ON(dst == nullptr);
    const unsigned int num_inputs = src_vector.size();

    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(dst);
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
                    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src_vector[0], src_vector[1]);
                    ARM_COMPUTE_RETURN_ON_ERROR(kernels::ClWidthConcatenate2TensorsKernel::validate(src_vector[0], src_vector[1], dst));
                    break;
                case 4:
                    // Validate WidthConcatenate4Tensors kernels if there are 4 inputs
                    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src_vector[0], src_vector[1], src_vector[2], src_vector[3]);
                    ARM_COMPUTE_RETURN_ON_ERROR(kernels::ClWidthConcatenate4TensorsKernel::validate(src_vector[0], src_vector[1], src_vector[2], src_vector[3], dst));
                    break;
                default:
                    // Validate generic case of WidthConcatenate kernel
                    for(const auto &src : src_vector)
                    {
                        ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src);
                        ARM_COMPUTE_RETURN_ON_ERROR(kernels::ClWidthConcatenateKernel::validate(src, offset, dst));
                        offset += src->dimension(axis);
                    }
                    break;
            }
            break;
        }
        case Window::DimY:
        {
            for(const auto &src : src_vector)
            {
                ARM_COMPUTE_RETURN_ON_ERROR(kernels::ClHeightConcatenateKernel::validate(src, offset, dst));
                offset += src->dimension(axis);
            }
            break;
        }
        case Window::DimZ:
        {
            for(const auto &src : src_vector)
            {
                ARM_COMPUTE_RETURN_ON_ERROR(kernels::ClDepthConcatenateKernel::validate(src, offset, dst));
                offset += src->dimension(axis);
            }
            break;
        }
        case 3:
        {
            for(const auto &src : src_vector)
            {
                ARM_COMPUTE_RETURN_ON_ERROR(kernels::ClBatchConcatenateKernel::validate(src, offset, dst));
                offset += src->dimension(axis);
            }
            break;
        }
        default:
            ARM_COMPUTE_ERROR("Axis not supported");
    }

    if(dst->total_size() != 0)
    {
        TensorShape dst_shape = arm_compute::misc::shape_calculator::calculate_concatenate_shape(src_vector, axis);
        ARM_COMPUTE_RETURN_ERROR_ON(dst_shape.total_size() != dst->tensor_shape().total_size());
    }

    return Status{};
}

void ClConcatenate::run(ITensorPack &tensors)
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
} // namespace opencl
} // namespace arm_compute
