/*
 * Copyright (c) 2016, 2017 ARM Limited.
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
#include "arm_compute/runtime/TensorAllocator.h"

#include "arm_compute/core/Coordinates.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/TensorInfo.h"

#include <cstddef>

using namespace arm_compute;

namespace
{
bool validate_subtensor_shape(const TensorInfo &parent_info, const TensorInfo &child_info, const Coordinates &coords)
{
    bool               is_valid     = true;
    const TensorShape &parent_shape = parent_info.tensor_shape();
    const TensorShape &child_shape  = child_info.tensor_shape();
    const size_t       parent_dims  = parent_info.num_dimensions();
    const size_t       child_dims   = child_info.num_dimensions();

    if(child_dims <= parent_dims)
    {
        for(size_t num_dimensions = child_dims; num_dimensions > 0; --num_dimensions)
        {
            const size_t child_dim_size = coords[num_dimensions - 1] + child_shape[num_dimensions - 1];

            if((coords[num_dimensions - 1] < 0) || (child_dim_size > parent_shape[num_dimensions - 1]))
            {
                is_valid = false;
                break;
            }
        }
    }
    else
    {
        is_valid = false;
    }

    return is_valid;
}
} // namespace

TensorAllocator::TensorAllocator()
    : _buffer(nullptr)
{
}

void TensorAllocator::init(const TensorAllocator &allocator, const Coordinates &coords, TensorInfo sub_info)
{
    // Get parent info
    const TensorInfo parent_info = allocator.info();

    // Check if coordinates and new shape are within the parent tensor
    ARM_COMPUTE_ERROR_ON(!validate_subtensor_shape(parent_info, sub_info, coords));
    ARM_COMPUTE_UNUSED(validate_subtensor_shape);

    // Copy pointer to buffer
    _buffer = allocator._buffer;

    // Init tensor info with new dimensions
    size_t total_size = parent_info.offset_element_in_bytes(coords) + sub_info.total_size() - sub_info.offset_first_element_in_bytes();
    sub_info.init(sub_info.tensor_shape(), sub_info.format(), parent_info.strides_in_bytes(), parent_info.offset_element_in_bytes(coords), total_size);

    // Set TensorInfo
    init(sub_info);
}

uint8_t *TensorAllocator::data() const
{
    return (_buffer != nullptr) ? _buffer.get()->data() : nullptr;
}

void TensorAllocator::allocate()
{
    ARM_COMPUTE_ERROR_ON(_buffer != nullptr);

    _buffer = std::make_shared<std::vector<uint8_t>>(info().total_size());
    info().set_is_resizable(false);
}

void TensorAllocator::free()
{
    ARM_COMPUTE_ERROR_ON(_buffer == nullptr);

    _buffer.reset();
    info().set_is_resizable(true);
}

uint8_t *TensorAllocator::lock()
{
    return (_buffer != nullptr) ? _buffer.get()->data() : nullptr;
}

void TensorAllocator::unlock()
{
}
