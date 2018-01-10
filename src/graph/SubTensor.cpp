/*
 * Copyright (c) 2017-2018 ARM Limited.
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
#include "arm_compute/graph/SubTensor.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/CL/CLSubTensor.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/SubTensor.h"
#include "arm_compute/runtime/Tensor.h"
#include "utils/TypePrinter.h"

using namespace arm_compute::graph;

namespace
{
template <typename SubTensorType, typename ParentTensorType>
std::unique_ptr<arm_compute::ITensor> initialise_subtensor(arm_compute::ITensor *parent, TensorShape shape, Coordinates coords, bool extend_parent)
{
    auto ptensor   = dynamic_cast<ParentTensorType *>(parent);
    auto subtensor = arm_compute::support::cpp14::make_unique<SubTensorType>(ptensor, shape, coords, extend_parent);
    return std::move(subtensor);
}
} // namespace

SubTensor::SubTensor()
    : _target(TargetHint::DONT_CARE), _tensor_shape(), _coords(), _parent(nullptr), _subtensor(nullptr), _extend_parent(false)
{
}

SubTensor::SubTensor(Tensor &parent, TensorShape tensor_shape, Coordinates coords, bool extend_parent)
    : _target(TargetHint::DONT_CARE), _tensor_shape(tensor_shape), _coords(coords), _parent(nullptr), _subtensor(nullptr), _extend_parent(extend_parent)
{
    ARM_COMPUTE_ERROR_ON(parent.tensor() == nullptr);
    _parent = parent.tensor();
    _target = parent.target();

    instantiate_subtensor();
}

SubTensor::SubTensor(arm_compute::ITensor *parent, TensorShape tensor_shape, Coordinates coords, TargetHint target, bool extend_parent)
    : _target(target), _tensor_shape(tensor_shape), _coords(coords), _parent(parent), _subtensor(nullptr), _extend_parent(extend_parent)
{
    ARM_COMPUTE_ERROR_ON(parent == nullptr);
    instantiate_subtensor();
}

bool SubTensor::call_accessor()
{
    return true;
}

bool SubTensor::has_accessor() const
{
    return false;
}

arm_compute::ITensor *SubTensor::set_target(TargetHint target)
{
    ARM_COMPUTE_ERROR_ON(target != _target);
    return (target == _target) ? _subtensor.get() : nullptr;
}

arm_compute::ITensor *SubTensor::tensor()
{
    return _subtensor.get();
}

const arm_compute::ITensor *SubTensor::tensor() const
{
    return _subtensor.get();
}

TargetHint SubTensor::target() const
{
    return _target;
}

void SubTensor::allocate()
{
    // NOP for sub-tensors
}

void SubTensor::instantiate_subtensor()
{
    switch(_target)
    {
        case TargetHint::OPENCL:
            _subtensor = initialise_subtensor<arm_compute::CLSubTensor, arm_compute::ICLTensor>(_parent, _tensor_shape, _coords, _extend_parent);
            break;
        case TargetHint::NEON:
            _subtensor = initialise_subtensor<arm_compute::SubTensor, arm_compute::ITensor>(_parent, _tensor_shape, _coords, _extend_parent);
            break;
        default:
            ARM_COMPUTE_ERROR("Invalid TargetHint");
    }
}
