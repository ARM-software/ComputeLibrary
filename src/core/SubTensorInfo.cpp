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
#include "arm_compute/core/SubTensorInfo.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Validate.h"
#include "support/ToolchainSupport.h"

using namespace arm_compute;

namespace
{
/** Extends parent shape depending on subtensor's coordinates and shape
 *
 * @param parent_shape Parent shape
 * @param shape        Subtensor shape
 * @param coords       Subtensor coordinates inside parent tensor
 *
 * @return Extended parent shape
 */
TensorShape extend_parent_shape(TensorShape parent_shape, TensorShape shape, Coordinates coords)
{
    // Subtensor should not index in x, y dimensions.
    ARM_COMPUTE_ERROR_ON((coords.x() != 0) || (coords.y() != 0));

    // Cannot extend on x, y ?
    ARM_COMPUTE_ERROR_ON((parent_shape.total_size() != 0) && (parent_shape.x() != shape.x()) && (parent_shape.y() != shape.y()));

    // Extend shape
    for(unsigned int i = 0; i < TensorShape::num_max_dimensions; ++i)
    {
        int dimension_extend = coords[i] + static_cast<int>(shape[i]);
        if((dimension_extend > static_cast<int>(parent_shape[i])) && (dimension_extend > 0))
        {
            parent_shape.set(i, static_cast<size_t>(dimension_extend));
        }
    }

    return parent_shape;
}
} // namespace

SubTensorInfo::SubTensorInfo()
    : _parent(nullptr), _tensor_shape(), _coords(), _valid_region{ Coordinates(), _tensor_shape }, _extend_parent(false)
{
}

SubTensorInfo::SubTensorInfo(ITensorInfo *parent, TensorShape tensor_shape, Coordinates coords, bool extend_parent)
    : _parent(parent), _tensor_shape(tensor_shape), _coords(coords), _valid_region{ Coordinates(), _tensor_shape }, _extend_parent(extend_parent)
{
    ARM_COMPUTE_ERROR_ON(parent == nullptr);
    // Check if subtensor is valid if parent is configured
    if(parent->tensor_shape().total_size() != 0 && !_extend_parent)
    {
        ARM_COMPUTE_ERROR_ON_INVALID_SUBTENSOR(parent->tensor_shape(), coords, tensor_shape);
    }

    // Initialize valid region
    _valid_region = ValidRegion{ Coordinates(), _tensor_shape };
}

std::unique_ptr<ITensorInfo> SubTensorInfo::clone() const
{
    // Clone creates a TensorInfo object from SubTensorInfo's parent which will conclude to a TensorInfo
    // For now it does not make sense to copy a SubTensorInfo explicitly
    ARM_COMPUTE_ERROR_ON(_parent == nullptr);
    auto clone_obj = _parent->clone();
    clone_obj->set_tensor_shape(_tensor_shape);
    clone_obj->set_valid_region(_valid_region);
    return clone_obj;
}

ITensorInfo &SubTensorInfo::set_tensor_shape(TensorShape shape)
{
    ARM_COMPUTE_ERROR_ON(_parent == nullptr);

    // Check if subtensor is valid if parent is configured
    if(_parent->tensor_shape().total_size() != 0 && !_extend_parent)
    {
        ARM_COMPUTE_ERROR_ON_INVALID_SUBTENSOR(_parent->tensor_shape(), _coords, shape);
    }
    else if(_extend_parent) // Extend parent shape, configure if specified
    {
        ARM_COMPUTE_ERROR_ON((_parent->data_type() == DataType::UNKNOWN) && (_parent->format() == Format::UNKNOWN));
        TensorShape parent_extended_shape = extend_parent_shape(_parent->tensor_shape(), shape, _coords);
        _parent->set_tensor_shape(parent_extended_shape);
        _parent->set_valid_region(ValidRegion{ Coordinates(), parent_extended_shape });
    }
    _tensor_shape = shape;
    return *this;
}

bool SubTensorInfo::extend_padding(const PaddingSize &padding)
{
    ARM_COMPUTE_ERROR_ON(_parent == nullptr);
    ARM_COMPUTE_ERROR_ON(!_parent->is_resizable());
    ARM_COMPUTE_ERROR_ON(_parent->total_size() == 0);

    // Extend parent padding if required
    return _parent->extend_padding(padding);
}

size_t SubTensorInfo::offset_element_in_bytes(const Coordinates &pos) const
{
    ARM_COMPUTE_ERROR_ON_COORDINATES_DIMENSIONS_GTE(pos, _tensor_shape.num_dimensions());

    size_t         offset  = offset_first_element_in_bytes();
    const Strides &strides = strides_in_bytes();

    for(size_t i = 0; i < _tensor_shape.num_dimensions(); ++i)
    {
        offset += pos[i] * strides[i];
    }

    return offset;
}
