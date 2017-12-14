/*
 * Copyright (c) 2017 ARM Limited.
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
#include "arm_compute/graph/Tensor.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/Tensor.h"
#include "utils/TypePrinter.h"

using namespace arm_compute::graph;

namespace
{
template <typename TensorType>
std::unique_ptr<arm_compute::ITensor> initialise_tensor(TensorInfo &info)
{
    auto tensor = arm_compute::support::cpp14::make_unique<TensorType>();
    tensor->allocator()->init(info);
    return std::move(tensor);
}

template <typename TensorType>
void tensor_allocate(arm_compute::ITensor &tensor)
{
    auto itensor = dynamic_cast<TensorType *>(&tensor);
    ARM_COMPUTE_ERROR_ON_NULLPTR(itensor);
    itensor->allocator()->allocate();
}
} // namespace

Tensor::Tensor(TensorInfo &&info)
    : _target(TargetHint::DONT_CARE), _info(info), _accessor(nullptr), _tensor(nullptr)
{
}

Tensor::Tensor(Tensor &&src) noexcept
    : _target(src._target),
      _info(std::move(src._info)),
      _accessor(std::move(src._accessor)),
      _tensor(std::move(src._tensor))
{
}

void Tensor::set_info(TensorInfo &&info)
{
    _info = info;
}

bool Tensor::call_accessor()
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(_accessor.get());
    auto cl_tensor = dynamic_cast<arm_compute::CLTensor *>(_tensor.get());
    if(cl_tensor != nullptr && cl_tensor->buffer() == nullptr)
    {
        cl_tensor->map();
    }
    bool retval = _accessor->access_tensor(*_tensor);
    if(cl_tensor != nullptr)
    {
        cl_tensor->unmap();
    }
    return retval;
}

bool Tensor::has_accessor() const
{
    return (_accessor != nullptr);
}

arm_compute::ITensor *Tensor::tensor()
{
    return _tensor.get();
}

const arm_compute::ITensor *Tensor::tensor() const
{
    return _tensor.get();
}

const TensorInfo &Tensor::info() const
{
    return _info;
}

arm_compute::ITensor *Tensor::set_target(TargetHint target)
{
    if(_tensor != nullptr)
    {
        ARM_COMPUTE_ERROR_ON(target != _target);
    }
    else
    {
        switch(target)
        {
            case TargetHint::OPENCL:
                _tensor = initialise_tensor<arm_compute::CLTensor>(_info);
                break;
            case TargetHint::NEON:
                _tensor = initialise_tensor<arm_compute::Tensor>(_info);
                break;
            default:
                ARM_COMPUTE_ERROR("Invalid TargetHint");
        }
        _target = target;
    }
    return _tensor.get();
}

void Tensor::allocate()
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(_tensor.get());
    switch(_target)
    {
        case TargetHint::OPENCL:
            tensor_allocate<arm_compute::CLTensor>(*_tensor);
            break;
        case TargetHint::NEON:
            tensor_allocate<arm_compute::Tensor>(*_tensor);
            break;
        default:
            ARM_COMPUTE_ERROR("Invalid TargetHint");
    }
}

void Tensor::allocate_and_fill_if_needed()
{
    allocate();
    if(_accessor != nullptr)
    {
        call_accessor();
    }
}

TargetHint Tensor::target() const
{
    return _target;
}
