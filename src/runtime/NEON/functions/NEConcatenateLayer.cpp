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
#include "arm_compute/runtime/NEON/functions/NEConcatenateLayer.h"

#include "arm_compute/core/NEON/kernels/NEBatchConcatenateLayerKernel.h"
#include "arm_compute/core/NEON/kernels/NEDepthConcatenateLayerKernel.h"
#include "arm_compute/core/NEON/kernels/NEHeightConcatenateLayerKernel.h"
#include "arm_compute/core/NEON/kernels/NEWidthConcatenateLayerKernel.h"

#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "support/MemorySupport.h"

namespace arm_compute
{
namespace experimental
{
NEConcatenation::NEConcatenation()
    : _concat_kernels(), _num_inputs(0), _axis(0)
{
}

void NEConcatenation::configure(const std::vector<const ITensorInfo *> &inputs_vector, ITensorInfo *output, size_t axis)
{
    ARM_COMPUTE_ERROR_ON(output == nullptr);

    _axis       = axis;
    _num_inputs = inputs_vector.size();

    TensorShape output_shape = arm_compute::misc::shape_calculator::calculate_concatenate_shape(inputs_vector, axis);

    // Output auto inizialitation if not yet initialized
    auto_init_if_empty(*output, output_shape, 1, inputs_vector[0]->data_type());
    ARM_COMPUTE_ERROR_THROW_ON(NEConcatenateLayer::validate(inputs_vector, output, axis));

    unsigned int offset = 0;

    for(unsigned int i = 0; i < _num_inputs; ++i)
    {
        switch(axis)
        {
            case Window::DimX:
            {
                auto kernel = support::cpp14::make_unique<NEWidthConcatenateLayerKernel>();
                kernel->configure(inputs_vector.at(i), offset, output);
                _concat_kernels.emplace_back(std::move(kernel));
                break;
            }
            case Window::DimY:
            {
                auto kernel = support::cpp14::make_unique<NEHeightConcatenateLayerKernel>();
                kernel->configure(inputs_vector.at(i), offset, output);
                _concat_kernels.emplace_back(std::move(kernel));
                break;
            }
            case Window::DimZ:
            {
                auto kernel = support::cpp14::make_unique<NEDepthConcatenateLayerKernel>();
                kernel->configure(inputs_vector.at(i), offset, output);
                _concat_kernels.emplace_back(std::move(kernel));
                break;
            }
            case 3:
            {
                auto kernel = support::cpp14::make_unique<NEBatchConcatenateLayerKernel>();
                kernel->configure(inputs_vector.at(i), offset, output);
                _concat_kernels.emplace_back(std::move(kernel));
                break;
            }
            default:
                ARM_COMPUTE_ERROR("Axis not supported");
        }
        offset += inputs_vector.at(i)->dimension(axis);
    }
}

Status NEConcatenation::validate(const std::vector<const ITensorInfo *> &inputs_vector, const ITensorInfo *output, size_t axis)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(output);
    ARM_COMPUTE_RETURN_ERROR_ON(inputs_vector.size() < 2);

    unsigned int offset = 0;
    for(const auto &input : inputs_vector)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input);
        switch(axis)
        {
            case Window::DimX:
            {
                ARM_COMPUTE_RETURN_ON_ERROR(NEWidthConcatenateLayerKernel::validate(input, offset, output));
                break;
            }
            case Window::DimY:
            {
                ARM_COMPUTE_RETURN_ON_ERROR(NEHeightConcatenateLayerKernel::validate(input, offset, output));
                break;
            }
            case Window::DimZ:
            {
                ARM_COMPUTE_RETURN_ON_ERROR(NEDepthConcatenateLayerKernel::validate(input, offset, output));
                break;
            }
            case 3:
            {
                ARM_COMPUTE_RETURN_ON_ERROR(NEBatchConcatenateLayerKernel::validate(input, offset, output));
                break;
            }
            default:
                ARM_COMPUTE_ERROR("Axis not supported");
        }
        offset += input->dimension(axis);
    }

    if(output->total_size() != 0)
    {
        TensorShape output_shape = arm_compute::misc::shape_calculator::calculate_concatenate_shape(inputs_vector, axis);
        ARM_COMPUTE_RETURN_ERROR_ON(output_shape.total_size() != output->tensor_shape().total_size());
    }

    return Status{};
}

void NEConcatenation::run(ITensorPack &tensors)
{
    if(tensors.empty())
    {
        ARM_COMPUTE_ERROR("No inputs provided");
    }

    if(static_cast<int>(tensors.size() - 1) != static_cast<int>(_num_inputs))
    {
        ARM_COMPUTE_ERROR("Configured with different number of inputs");
    }

    int i = 0;
    for(auto &k : _concat_kernels)
    {
        ITensorPack pack;
        pack.add_tensor(TensorType::ACL_SRC, tensors.get_const_tensor(ACL_SRC_VEC + i));
        pack.add_tensor(TensorType::ACL_DST, tensors.get_tensor(ACL_DST));
        NEScheduler::get().schedule_op(k.get(), Window::DimY, pack);
        ++i;
    }
}
} // namespace experimental

struct NEConcatenateLayer::Impl
{
    std::vector<const ITensor *>                   srcs{};
    ITensor                                       *dst{ nullptr };
    unsigned int                                   num_inputs{ 0 };
    unsigned int                                   axis{ 0 };
    std::unique_ptr<experimental::NEConcatenation> op{ nullptr };
};

NEConcatenateLayer::NEConcatenateLayer()
    : _impl(support::cpp14::make_unique<Impl>())
{
}

NEConcatenateLayer::NEConcatenateLayer(NEConcatenateLayer &&) = default;

NEConcatenateLayer &NEConcatenateLayer::operator=(NEConcatenateLayer &&) = default;

NEConcatenateLayer::~NEConcatenateLayer() = default;

void NEConcatenateLayer::configure(std::vector<const ITensor *> inputs_vector, ITensor *output, size_t axis)
{
    ARM_COMPUTE_ERROR_ON(output == nullptr);

    _impl->srcs       = inputs_vector;
    _impl->dst        = output;
    _impl->axis       = axis;
    _impl->num_inputs = inputs_vector.size();
    _impl->op         = arm_compute::support::cpp14::make_unique<experimental::NEConcatenation>();

    std::vector<const ITensorInfo *> inputs_vector_info;
    for(unsigned int i = 0; i < inputs_vector.size(); ++i)
    {
        ARM_COMPUTE_ERROR_ON_NULLPTR(inputs_vector.at(i));
        inputs_vector_info.emplace_back(inputs_vector.at(i)->info());
    }
    _impl->op->configure(inputs_vector_info, _impl->dst->info(), axis);
}

Status NEConcatenateLayer::validate(const std::vector<const ITensorInfo *> &inputs_vector, const ITensorInfo *output, size_t axis)
{
    return experimental::NEConcatenation::validate(inputs_vector, output, axis);
}

void NEConcatenateLayer::run()
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
