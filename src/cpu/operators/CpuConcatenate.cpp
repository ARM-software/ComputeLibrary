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
#include "src/cpu/operators/CpuConcatenate.h"

#include "src/cpu/kernels/CpuConcatenateBatchKernel.h"
#include "src/cpu/kernels/CpuConcatenateDepthKernel.h"
#include "src/cpu/kernels/CpuConcatenateHeightKernel.h"
#include "src/cpu/kernels/CpuConcatenateWidthKernel.h"

#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "src/core/helpers/AutoConfiguration.h"

namespace arm_compute
{
namespace cpu
{
void CpuConcatenate::configure(const std::vector<const ITensorInfo *> &srcs_vector, ITensorInfo *dst, size_t axis)
{
    ARM_COMPUTE_ERROR_ON(dst == nullptr);

    _axis     = axis;
    _num_srcs = srcs_vector.size();

    TensorShape dst_shape = arm_compute::misc::shape_calculator::calculate_concatenate_shape(srcs_vector, axis);

    // Output auto inizialitation if not yet initialized
    auto_init_if_empty(*dst, dst_shape, 1, srcs_vector[0]->data_type());
    ARM_COMPUTE_ERROR_THROW_ON(CpuConcatenate::validate(srcs_vector, dst, axis));

    unsigned int offset = 0;

    for(unsigned int i = 0; i < _num_srcs; ++i)
    {
        switch(axis)
        {
            case Window::DimX:
            {
                auto kernel = std::make_unique<kernels::CpuConcatenateWidthKernel>();
                kernel->configure(srcs_vector.at(i), offset, dst);
                _concat_kernels.emplace_back(std::move(kernel));
                break;
            }
            case Window::DimY:
            {
                auto kernel = std::make_unique<kernels::CpuConcatenateHeightKernel>();
                kernel->configure(srcs_vector.at(i), offset, dst);
                _concat_kernels.emplace_back(std::move(kernel));
                break;
            }
            case Window::DimZ:
            {
                auto kernel = std::make_unique<kernels::CpuConcatenateDepthKernel>();
                kernel->configure(srcs_vector.at(i), offset, dst);
                _concat_kernels.emplace_back(std::move(kernel));
                break;
            }
            case 3:
            {
                auto kernel = std::make_unique<kernels::CpuConcatenateBatchKernel>();
                kernel->configure(srcs_vector.at(i), offset, dst);
                _concat_kernels.emplace_back(std::move(kernel));
                break;
            }
            default:
                ARM_COMPUTE_ERROR("Axis not supported");
        }
        offset += srcs_vector.at(i)->dimension(axis);
    }
}

Status CpuConcatenate::validate(const std::vector<const ITensorInfo *> &srcs_vector, const ITensorInfo *dst, size_t axis)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(dst);
    ARM_COMPUTE_RETURN_ERROR_ON(srcs_vector.size() < 2);

    unsigned int offset = 0;
    for(const auto &src : srcs_vector)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src);
        switch(axis)
        {
            case Window::DimX:
            {
                ARM_COMPUTE_RETURN_ON_ERROR(kernels::CpuConcatenateWidthKernel::validate(src, offset, dst));
                break;
            }
            case Window::DimY:
            {
                ARM_COMPUTE_RETURN_ON_ERROR(kernels::CpuConcatenateHeightKernel::validate(src, offset, dst));
                break;
            }
            case Window::DimZ:
            {
                ARM_COMPUTE_RETURN_ON_ERROR(kernels::CpuConcatenateDepthKernel::validate(src, offset, dst));
                break;
            }
            case 3:
            {
                ARM_COMPUTE_RETURN_ON_ERROR(kernels::CpuConcatenateBatchKernel::validate(src, offset, dst));
                break;
            }
            default:
                ARM_COMPUTE_ERROR("Axis not supported");
        }
        offset += src->dimension(axis);
    }

    if(dst->total_size() != 0)
    {
        TensorShape dst_shape = arm_compute::misc::shape_calculator::calculate_concatenate_shape(srcs_vector, axis);
        ARM_COMPUTE_RETURN_ERROR_ON(dst_shape.total_size() != dst->tensor_shape().total_size());
    }

    return Status{};
}

void CpuConcatenate::run(ITensorPack &tensors)
{
    if(tensors.empty())
    {
        ARM_COMPUTE_ERROR("No inputs provided");
    }

    if(static_cast<int>(tensors.size() - 1) != static_cast<int>(_num_srcs))
    {
        ARM_COMPUTE_ERROR("Configured with different number of inputs");
    }

    int i = 0;
    for(auto &k : _concat_kernels)
    {
        ITensorPack pack;
        pack.add_tensor(TensorType::ACL_SRC, tensors.get_const_tensor(ACL_SRC_VEC + i));
        pack.add_tensor(TensorType::ACL_DST, tensors.get_tensor(ACL_DST));
        NEScheduler::get().schedule_op(k.get(), Window::DimY, k->window(), pack);
        ++i;
    }
}
} // namespace cpu
} // namespace arm_compute
