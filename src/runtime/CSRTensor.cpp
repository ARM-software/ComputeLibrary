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
#include "arm_compute/runtime/COOTensor.h"

#include "arm_compute/core/CoreTypes.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/TensorFormat.h"
#include "arm_compute/runtime/Tensor.h"

#include "src/core/helpers/Utils.h"

#include <functional>

namespace arm_compute
{

namespace
{
TensorInfo csr_tensor_info(const ITensorInfo *src_info)
{
    return src_info->clone()->set_tensor_format(TensorFormat::CSR);
}
} // namespace

CSRTensor::CSRTensor(const ITensor *tensor, size_t sparse_dim) : SparseTensor(tensor->info()->num_dimensions(), 2), _crow_bytes(), _col_bytes(), _allocator(this)
{
    ARM_COMPUTE_UNUSED(sparse_dim);
    ARM_COMPUTE_ERROR_ON_NULLPTR(tensor);
    const auto *info = tensor->info();

    // As of now, CSRTensor only supports 2D tensors with NCHW layout.
    ARM_COMPUTE_ERROR_ON_MSG(info->data_layout() != DataLayout::NCHW, "CSRTensor only supports NCHW layout at the moment");
    ARM_COMPUTE_ERROR_ON_MSG(info->is_sparse(), "cannot create a CSRTensor from a sparse tensor");
    ARM_COMPUTE_ERROR_ON_MSG(dim() != 2, "CSRTensor only supports 2D tensors at the moment");

    const int32_t           rows = info->dimension(0);
    const int32_t           cols = info->dimension(1);
    const int32_t   element_size = info->element_size();
    const int32_t row_size_bytes = cols * element_size;
    const auto        is_nonzero = make_is_nonzero_predicate(info->data_type());
    const uint8_t          *data = tensor->buffer() + info->offset_first_element_in_bytes();
    size_t       value_byte_size = 0;

    _crow_bytes = index_size;  // The first row index is always a 0
    _col_bytes = 0;

    for(int32_t row = 0; row < rows; ++row)
    {
        const int32_t row_offset = row * row_size_bytes;
        _crow_bytes += index_size;
        for(int32_t col = 0; col < cols; ++col)
        {
            const int32_t element_offset =  row_offset + col * element_size;
            if(is_nonzero(data + element_offset))
            {
                _col_bytes += index_size;
                value_byte_size += element_size * dense_volume(sparse_dim);
            }
        }
    }

    _allocator.init(csr_tensor_info(info), value_byte_size, _crow_bytes + _col_bytes);
    _allocator.allocate();

    uint8_t *_row_offsets = _allocator.data();
    uint8_t *_col_indices = _allocator.data() + _crow_bytes;
    uint8_t      *_values = _allocator.data() + _crow_bytes + _col_bytes;
    size_t   num_non_zero = 0;
    int32_t      last_row = 0;
    int32_t     col_index = 0;

    std::memcpy(_row_offsets, &last_row, index_size);

    for(int32_t row = 0; row < rows; ++row)
    {
        const int32_t row_offset = row * row_size_bytes;
        int32_t non_zero_row_elements = 0;
        for(int32_t col = 0; col < cols; ++col)
        {
            const size_t element_offset =  row_offset + col * element_size;
            if(is_nonzero(data + element_offset))
            {
                std::memcpy(_col_indices + col_index * index_size, &col, index_size);
                std::memcpy(_values + num_non_zero * element_size, data + element_offset, element_size);
                non_zero_row_elements++;
                num_non_zero++;
                col_index++;
            }
        }

        last_row += non_zero_row_elements;
        std::memcpy(_row_offsets + (row + 1) * index_size, &last_row, index_size);
    }
}

CSRTensor::CSRTensor(const ITensor *tensor) : CSRTensor(tensor, 2)
{
}

size_t CSRTensor::nnz() const
{
    return static_cast<size_t>(_col_bytes / index_size);
}

ITensorInfo *CSRTensor::info() const
{
    return &_allocator.info();
}

ITensorInfo *CSRTensor::info()
{
    return &_allocator.info();
}

uint8_t *CSRTensor::buffer() const
{
    return _allocator.data();
}

Coordinates CSRTensor::get_coordinates(size_t nth) const
{
    ARM_COMPUTE_ERROR_ON_MSG(nth >= nnz(), "Invalid index");

    const uint8_t *row_indices = _allocator.data();
    const uint8_t *col_indices = _allocator.data() + _crow_bytes;
    size_t                 low = 0;
    size_t                high = (_crow_bytes / index_size) - 1;

    while(low < high)
    {
        const size_t mid = (low + high) / 2;
        if(*reinterpret_cast<const int32_t *>(row_indices + (mid + 1) * index_size) <= static_cast<int32_t>(nth))
        {
            low = mid + 1;
        }
        else
        {
            high = mid;
        }
    }

    return Coordinates{ low , *reinterpret_cast<const int32_t *>(col_indices + (nth * index_size)) };
}

const uint8_t *CSRTensor::get_value(Coordinates coords) const
{
    ARM_COMPUTE_ERROR_ON_MSG(coords.num_dimensions() != info()->num_dimensions(), "Invalid coordinate dimension");
    for(size_t i = 0; i < coords.num_dimensions(); ++i)
    {
        ARM_COMPUTE_ERROR_ON_MSG(static_cast<size_t>(coords[i]) >= info()->tensor_shape()[i], "Invalid coordinates shape");
    }

    const uint8_t *row_indices = _allocator.data();
    const uint8_t *col_indices = _allocator.data() + _crow_bytes;
    const uint8_t      *values = _allocator.data() + _crow_bytes + _col_bytes;
    const int32_t          row = coords[0];
    const int32_t          col = coords[1];

    const int32_t start = *reinterpret_cast<const int32_t *>(row_indices + row * index_size);
    const int32_t end   = *reinterpret_cast<const int32_t *>(row_indices + (row + 1) * index_size);

    for(int32_t i = start; i < end; ++i)
    {
        if(*reinterpret_cast<const int32_t *>(col_indices + (i * index_size)) == col)
        {
            return values + i * info()->element_size();
        }
    }

    // This coordinates point to a zero value
    return nullptr;
}

std::unique_ptr<ITensor> CSRTensor::to_dense()
{
    ARM_COMPUTE_ERROR_ON_MSG(info()->data_layout() != DataLayout::NCHW, "CSRTensor only supports NCHW layout at the moment");

    auto tensor = std::make_unique<Tensor>();
    tensor->allocator()->init(info()->clone()->set_tensor_format(TensorFormat::Dense));
    tensor->allocator()->allocate();

    const size_t first_elem_offset = info()->offset_first_element_in_bytes();
    uint8_t                  *data = tensor->buffer() + first_elem_offset;
    const size_t      element_size = info()->element_size();
    const size_t        total_size = info()->total_size();
    const uint8_t     *row_offsets = _allocator.data();
    const uint8_t     *col_indices = _allocator.data() + _crow_bytes;
    const uint8_t          *values = _allocator.data() + _crow_bytes + _col_bytes;
    size_t                 element = 0;

    std::memset(data, 0, total_size);

    for(size_t i = 0, j = 1, row = 0; j < _crow_bytes / index_size; ++i, ++j, ++row)
    {
        const int32_t current_col = *reinterpret_cast<const int32_t *>(row_offsets + (i * index_size));
        const int32_t next_col = *reinterpret_cast<const int32_t *>(row_offsets + (j * index_size));

        if(current_col == next_col)
        {
            continue;
        }

        for(size_t current = static_cast<size_t>(current_col); current < static_cast<size_t>(next_col); ++current)
        {
            const size_t col = *reinterpret_cast<const int32_t *>(col_indices + (current * index_size));
            const uint8_t *value_ptr = values + (element * element_size);

            std::memcpy(data + (row * info()->dimension(1) + col) * element_size, value_ptr, element_size);
            element++;
        }
    }

    return tensor;
}

void CSRTensor::associate_memory_group(IMemoryGroup *memory_group)
{
    _allocator.set_associated_memory_group(memory_group);
}

#ifdef ARM_COMPUTE_ASSERTS_ENABLED
void CSRTensor::print(std::ostream &os) const
{
    const uint8_t *_row_offsets = _allocator.data();
    const uint8_t *_col_indices = _allocator.data() + _crow_bytes;
    const uint8_t       *values = _allocator.data() + _crow_bytes + _col_bytes;

    os << "r_offsets: [";
    for(size_t i = 0; i < _crow_bytes / index_size; ++i)
    {
        os << *reinterpret_cast<const int32_t *>(_row_offsets + (i * index_size));
        if (i < _crow_bytes / index_size - 1)
        {
            os << ", ";
        }
    }
    os << "] cols: [";
    for(size_t i = 0; i < _col_bytes / index_size; ++i)
    {
        os << *reinterpret_cast<const int32_t *>(_col_indices + (i * index_size));
        if (i < _col_bytes / index_size - 1)
        {
            os << ", ";
        }
    }
    os << "] values: ";
    print_values(os, values, 0, nnz());
}
#endif // ARM_COMPUTE_ASSERTS_ENABLED

} // namespace arm_compute
