/*
 * Copyright (c) 2025 Arm Limited.
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
#include "arm_compute/runtime/SparseTensorAllocator.h"

#include "arm_compute/core/Coordinates.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/runtime/MemoryGroup.h"
#include "arm_compute/runtime/MemoryRegion.h"

#include <cstddef>

namespace arm_compute
{
SparseTensorAllocator::SparseTensorAllocator(IMemoryManageable *owner) : _owner(owner), _associated_memory_group(nullptr), _memory(), _values_bytes(0), _indices_bytes(0)
{
}

SparseTensorAllocator::~SparseTensorAllocator()
{
    info().set_is_resizable(true);
}

SparseTensorAllocator::SparseTensorAllocator(SparseTensorAllocator &&o) noexcept
    : ITensorAllocator(std::move(o)),
      _owner(o._owner),
      _associated_memory_group(o._associated_memory_group),
      _memory(std::move(o._memory)),
      _values_bytes(o._values_bytes),
      _indices_bytes(o._indices_bytes)
{
    o._owner                   = nullptr;
    o._associated_memory_group = nullptr;
    o._memory                  = Memory();
    o._values_bytes       = 0;
    o._indices_bytes      = 0;
}

SparseTensorAllocator &SparseTensorAllocator::operator=(SparseTensorAllocator &&o) noexcept
{
    if (&o != this)
    {
        _owner   = o._owner;
        o._owner = nullptr;

        _associated_memory_group   = o._associated_memory_group;
        o._associated_memory_group = nullptr;

        _memory   = std::move(o._memory);
        o._memory = Memory();

        _values_bytes = o._values_bytes;
        _values_bytes = 0;

        _indices_bytes = o._indices_bytes;
        _indices_bytes = 0;

        ITensorAllocator::operator=(std::move(o));
    }
    return *this;
}

void SparseTensorAllocator::init(const TensorInfo &input, size_t values_bytes, size_t indices_bytes, size_t alignment)
{
    ITensorAllocator::init(input, alignment);
    _values_bytes = values_bytes;
    _indices_bytes = indices_bytes;
}

void SparseTensorAllocator::init(const SparseTensorAllocator &allocator, const Coordinates &coords, TensorInfo &sub_info)
{
    // Get parent info
    const TensorInfo parent_info = allocator.info();

    // Copy pointer to buffer
    _memory = Memory(allocator._memory.region());

    // Init tensor info with new dimensions
    size_t total_size =
        parent_info.offset_element_in_bytes(coords) + sub_info.total_size() - sub_info.offset_first_element_in_bytes();
    sub_info.init(sub_info.tensor_shape(), sub_info.format(), parent_info.strides_in_bytes(),
                  parent_info.offset_element_in_bytes(coords), total_size);

    // Set TensorInfo
    ITensorAllocator::init(sub_info);
}

uint8_t *SparseTensorAllocator::data() const
{
    return (_memory.region() == nullptr) ? nullptr : reinterpret_cast<uint8_t *>(_memory.region()->buffer());
}

void SparseTensorAllocator::allocate()
{
    // Align to 64-byte boundaries by default if alignment is not specified
    const size_t alignment_to_use = (alignment() != 0) ? alignment() : 64;
    const size_t size = size_bytes();

    if (_associated_memory_group == nullptr)
    {
        _memory.set_owned_region(std::make_unique<MemoryRegion>(size, alignment_to_use));
    }
    else
    {
        _associated_memory_group->finalize_memory(_owner, _memory, size, alignment_to_use);
    }
    info().set_is_resizable(false);
}

void SparseTensorAllocator::free()
{
    _memory.set_region(nullptr);
    info().set_is_resizable(true);
}

bool SparseTensorAllocator::is_allocated() const
{
    return _memory.region() != nullptr;
}

Status SparseTensorAllocator::import_memory(void *memory)
{
    ARM_COMPUTE_RETURN_ERROR_ON(memory == nullptr);
    ARM_COMPUTE_RETURN_ERROR_ON(_associated_memory_group != nullptr);
    ARM_COMPUTE_RETURN_ERROR_ON(alignment() != 0 && !arm_compute::utility::check_aligned(memory, alignment()));

    _memory.set_owned_region(std::make_unique<MemoryRegion>(memory, info().total_size()));
    info().set_is_resizable(false);

    return Status{};
}

void SparseTensorAllocator::set_associated_memory_group(IMemoryGroup *associated_memory_group)
{
    ARM_COMPUTE_ERROR_ON(associated_memory_group == nullptr);
    ARM_COMPUTE_ERROR_ON(_associated_memory_group != nullptr && _associated_memory_group != associated_memory_group);
    ARM_COMPUTE_ERROR_ON(_memory.region() != nullptr && _memory.region()->buffer() != nullptr);

    _associated_memory_group = associated_memory_group;
}

size_t SparseTensorAllocator::size_bytes() const
{
    return _values_bytes + _indices_bytes;
}

uint8_t *SparseTensorAllocator::lock()
{
    ARM_COMPUTE_ERROR_ON(_memory.region() == nullptr);
    return reinterpret_cast<uint8_t *>(_memory.region()->buffer());
}

void SparseTensorAllocator::unlock()
{
}
} // namespace arm_compute
