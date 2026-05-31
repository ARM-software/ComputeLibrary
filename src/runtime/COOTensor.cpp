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

#include <cstring>
#include <numeric>
#include <vector>


namespace arm_compute
{
namespace
{
TensorInfo coo_tensor_info(const ITensorInfo *src_info)
{
    return src_info->clone()->set_tensor_format(TensorFormat::COO);
}
} // namespace

COOTensor::COOTensor(const ITensor *tensor, size_t sparse_dim)
    : SparseTensor(tensor->info()->num_dimensions(), sparse_dim), _indices_bytes(0), _allocator(this)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(tensor);
    const ITensorInfo *info = tensor->info();

    ARM_COMPUTE_ERROR_ON_MSG(info->data_layout() != DataLayout::NCHW, "COOTensor only supports NCHW layout at the moment");
    ARM_COMPUTE_ERROR_ON_MSG(info->is_sparse(), "cannot create a COOTensor from a sparse tensor");
    ARM_COMPUTE_ERROR_ON_MSG_VAR(sparse_dim < 1 || sparse_dim > dim(),
                                 "argument must be in [1,%zu] range. %zu is given",
                                 dim(), sparse_dim);

    const uint8_t     *data = tensor->buffer() + info->offset_first_element_in_bytes();
    const size_t dense_dims = dense_dim();
    const auto   is_nonzero = make_is_nonzero_predicate(info->data_type());

    std::vector<size_t> sparse_shape(sparse_dim);
    std::vector<size_t> dense_shape(dense_dims);
    for(size_t i = 0; i < sparse_dim; i++)
    {
        sparse_shape[i] = info->dimension(i);
    }
    for(size_t i = 0; i < dense_dims; i++)
    {
        dense_shape[i] = info->dimension(sparse_dim + i);
    }

    const size_t         step = std::accumulate(dense_shape.begin(), dense_shape.end(), size_t(1), std::multiplies<size_t>());
    const size_t     max_iter = std::accumulate(sparse_shape.begin(), sparse_shape.end(), size_t(1), std::multiplies<size_t>());
    const size_t element_size = info->element_size();
    const size_t   slice_size = step * element_size;

    // count non-zero blocks to determine allocation sizes.
    size_t nnz_count = 0;
    for(size_t i = 0; i < max_iter; i++)
    {
        if(has_non_zero_elements(const_cast<uint8_t *>(data + i * slice_size), slice_size, element_size, is_nonzero))
        {
            nnz_count++;
        }
    }

    // Buffer layout: [ indices section | values section ]
    //   indices: nnz * sparse_dim * index_size bytes
    //   values:  nnz * slice_size bytes
    _indices_bytes = nnz_count * sparse_dim * index_size;
    const size_t value_byte_size = nnz_count * slice_size;

    _allocator.init(coo_tensor_info(info), value_byte_size, _indices_bytes);
    _allocator.allocate();

    uint8_t *indices_ptr = _allocator.data();
    uint8_t  *values_ptr = _allocator.data() + _indices_bytes;
    size_t         entry = 0;

    // Second pass: write indices and values directly into the buffer.
    for(size_t i = 0; i < max_iter; i++)
    {
        const size_t offset = i * slice_size;
        if(!has_non_zero_elements(const_cast<uint8_t *>(data + offset), slice_size, element_size, is_nonzero))
        {
            continue;
        }

        // Compute per-dimension sparse coordinates (equivalent to numpy unravel_index).
        std::vector<int32_t> multi_index(sparse_dim, int32_t{ 0 });
        size_t remainder = i;
        for(size_t sd = sparse_dim; sd-- > 0;)
        {
            multi_index[sd] = remainder % sparse_shape[sd];
            remainder /= sparse_shape[sd];
        }

        std::memcpy(indices_ptr + entry * sparse_dim * index_size, multi_index.data(), sparse_dim * index_size);
        std::memcpy(values_ptr  + entry * slice_size,               data + offset,      slice_size);
        entry++;
    }
}

COOTensor::COOTensor(const ITensor *tensor) : COOTensor(tensor, tensor->info()->num_dimensions())
{
}

size_t COOTensor::nnz() const
{
    return (sparse_dim() > 0) ? (_indices_bytes / (sparse_dim() * index_size)) : 0;
}

ITensorInfo *COOTensor::info() const
{
    return &_allocator.info();
}

ITensorInfo *COOTensor::info()
{
    return &_allocator.info();
}

uint8_t *COOTensor::buffer() const
{
    return _allocator.data();
}

std::unique_ptr<ITensor> COOTensor::to_dense()
{
    ARM_COMPUTE_ERROR_ON_MSG(info()->data_layout() != DataLayout::NCHW, "COOTensor only supports NCHW layout at the moment");

    std::unique_ptr<Tensor> tensor = std::make_unique<Tensor>();
    tensor->allocator()->init(info()->clone()->set_tensor_format(TensorFormat::Dense));
    tensor->allocator()->allocate();

    const size_t      element_size = info()->element_size();
    const size_t        total_size = info()->total_size();
    const size_t         dense_vol = dense_volume(sparse_dim());
    const size_t first_elem_offset = info()->offset_first_element_in_bytes();

    std::memset(tensor->buffer() + first_elem_offset, 0, total_size);

    if(nnz() == 0)
    {
        return tensor;
    }

    const size_t       sdim = sparse_dim();
    const uint8_t *idx_base = _allocator.data();
    const uint8_t *val_base = _allocator.data() + _indices_bytes;

    for(size_t i = 0; i < nnz(); i++)
    {
        const int32_t    *stored = reinterpret_cast<const int32_t *>(idx_base + i * sdim * index_size);
        const uint8_t *block_ptr = val_base + i * dense_vol * element_size;

        size_t final_offset = 0;
        for(size_t d = 0; d < sdim; d++)
        {
            final_offset += stored[d] * dense_volume(d + 1);
        }
        final_offset *= element_size;

        for(size_t j = 0; j < dense_vol; j++)
        {
            std::memcpy(
                tensor->buffer() + first_elem_offset + final_offset + j * element_size,
                block_ptr + j * element_size,
                element_size);
        }
    }

    return tensor;
}

Coordinates COOTensor::get_coordinates(size_t nth) const
{
    ARM_COMPUTE_ERROR_ON_MSG(nth >= nnz(), "Invalid index");

    const size_t        sdim = sparse_dim();
    const int32_t *index_ptr = reinterpret_cast<const int32_t *>(_allocator.data() + nth * sdim * index_size);
    std::vector<int32_t>  coords_data(index_ptr, index_ptr + sdim);

    return Coordinates(coords_data);
}

const uint8_t *COOTensor::get_value(Coordinates coords) const
{
    ARM_COMPUTE_ERROR_ON_MSG(coords.num_dimensions() != info()->num_dimensions(), "Invalid coordinate dimension");
    for(size_t i = 0; i < coords.num_dimensions(); ++i)
    {
        ARM_COMPUTE_ERROR_ON_MSG(static_cast<size_t>(coords[i]) >= info()->tensor_shape()[i], "Invalid coordinates shape");
    }

    const size_t          n = nnz();
    const size_t       sdim = sparse_dim();
    const size_t  dense_vol = dense_volume(sdim);
    const size_t  elem_size = info()->element_size();
    const uint8_t *idx_base = _allocator.data();
    const uint8_t *val_base = _allocator.data() + _indices_bytes;

    for(size_t i = 0; i < n; i++)
    {
        const int32_t *stored = reinterpret_cast<const int32_t *>(idx_base + i * sdim * index_size);
        bool            match = true;

        for(size_t d = 0; d < sdim; d++)
        {
            if(stored[d] != coords[d])
            {
                match = false;
                break;
            }
        }
        if(match)
        {
            return val_base + i * dense_vol * elem_size;
        }
    }

    // This coordinates point to a zero value
    return nullptr;
}

void COOTensor::associate_memory_group(IMemoryGroup *memory_group)
{
    _allocator.set_associated_memory_group(memory_group);
}

#ifdef ARM_COMPUTE_ASSERTS_ENABLED
void COOTensor::print(std::ostream &os) const
{
    if(nnz() == 0)
    {
        os << "index: [] values: []" << std::endl;
        return;
    }

    const size_t       sdim = sparse_dim();
    const uint8_t *idx_base = _allocator.data();
    const uint8_t *val_base = _allocator.data() + _indices_bytes;

    for(size_t i = 0; i < nnz(); i++)
    {
        const int32_t *stored = reinterpret_cast<const int32_t *>(idx_base + i * sdim * index_size);
        os << "index: [";
        for(size_t d = 0; d < sdim; d++)
        {
            os << stored[d];
            if(d < sdim - 1) os << ", ";
        }
        os << "]  values: ";
        print_values(os, val_base, i, dense_volume(sparse_dim()));
    }
}
#endif // ARM_COMPUTE_ASSERTS_ENABLED

} // namespace arm_compute
