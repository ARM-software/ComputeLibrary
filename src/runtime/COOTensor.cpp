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


namespace arm_compute
{
namespace
{
TensorInfo coo_tensor_info(const ITensorInfo *src_info)
{
    return src_info->clone()->set_tensor_format(TensorFormat::COO);
}
} // namespace

COOTensor::COOTensor(const ITensor *tensor, size_t sparse_dim) : SparseTensor(tensor->info()->num_dimensions(), sparse_dim), _indices(), _allocator(this)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(tensor);
    const ITensorInfo *info = tensor->info();

    ARM_COMPUTE_ERROR_ON_MSG(info->data_layout() != DataLayout::NCHW, "COOTensor only supports NCHW layout at the moment");
    ARM_COMPUTE_ERROR_ON_MSG(info->is_sparse(), "cannot create a COOTensor from a sparse tensor");
    ARM_COMPUTE_ERROR_ON_MSG_VAR(sparse_dim < 1 || sparse_dim > dim(),
                                 "argument must be in [1,%zu] range. %zu is given",
                                 dim(), sparse_dim);

    const uint8_t     *data = tensor->buffer();
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

    std::vector<uint8_t> temp_values;

    const size_t         step = std::accumulate(dense_shape.begin(), dense_shape.end(), size_t(1), std::multiplies<size_t>());
    const size_t     max_iter = std::accumulate(sparse_shape.begin(), sparse_shape.end(), size_t(1), std::multiplies<size_t>());
    const size_t element_size = info->element_size();
    const size_t   slice_size = step * element_size;

    size_t value_byte_size = 0;
    size_t indices_bytes = 0;
    for(size_t i = 0; i < max_iter; i++)
    {
        const size_t offset = i * slice_size;
        if(has_non_zero_elements(const_cast<uint8_t *>(data + offset), slice_size, element_size, is_nonzero))
        {
            value_byte_size += slice_size;
            indices_bytes += dim() * sizeof(int32_t);
        }
    }

    _allocator.init(coo_tensor_info(info), value_byte_size, indices_bytes);
    _allocator.allocate();

    for(size_t i = 0; i < max_iter; i++)
    {
        const size_t offset = i * slice_size;
        if(!has_non_zero_elements(const_cast<uint8_t *>(data + offset), slice_size, element_size, is_nonzero))
        {
            continue;
        }

        // -----------------
        // TODO: I built this function in a similar way to numpy's unravel_index.
        // Do you think it's worth creating a util function somewhere else?
        // Also, it stores the coordinates in a reverse way, with the fastest
        // changing dimension last. Probably it would be better to store them
        // in a more natural order.
        std::vector<int32_t> multi_index(dim(), std::int32_t{0});
        size_t remainder = i;
        for(size_t sd = sparse_dim; sd-- > 0;)
        {
            multi_index[sd] = remainder % sparse_shape[sd];
            remainder /= sparse_shape[sd];
        }
        // -----------------

        _indices.push_back(Coordinates(multi_index));
        temp_values.insert(temp_values.end(), data + offset, data + offset + slice_size);
    }

    if(!temp_values.empty())
    {
        std::memcpy(_allocator.data(), temp_values.data(), temp_values.size());
    }
}

COOTensor::COOTensor(const ITensor *tensor) : COOTensor(tensor, tensor->info()->num_dimensions())
{
}

size_t COOTensor::nnz() const
{
    return _indices.size();
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

    for(size_t i = 0; i < _indices.size(); ++i)
    {
        const Coordinates     &c = _indices[i];
        const uint8_t *block_ptr = buffer() + i * dense_vol * element_size;

        size_t final_offset = 0;
        for(size_t d = 0; d < sparse_dim(); ++d)
        {
            final_offset += c[d] * dense_volume(d+1);
        }
        final_offset *= element_size;

        for(size_t j = 0; j < dense_vol; ++j)
        {
            const void *value_ptr = block_ptr + j * element_size;
            uint8_t     *base_ptr = tensor->buffer() + final_offset + j * element_size;

            std::memcpy(base_ptr, value_ptr, element_size);
        }
    }

    return tensor;
}

Coordinates COOTensor::get_coordinates(size_t nth) const
{
    ARM_COMPUTE_ERROR_ON_MSG(nth >= nnz(), "Invalid index");

    return _indices[nth];
}

const uint8_t *COOTensor::get_value(Coordinates coords) const
{
    ARM_COMPUTE_ERROR_ON_MSG(coords.num_dimensions() != info()->num_dimensions(), "Invalid coordinate dimension");
    for(size_t i = 0; i < coords.num_dimensions(); ++i)
    {
        ARM_COMPUTE_ERROR_ON_MSG(static_cast<size_t>(coords[i]) >= info()->tensor_shape()[i], "Invalid coordinates shape");
    }

    const uint8_t    *data = static_cast<const uint8_t *>(buffer());
    const size_t dense_vol = dense_volume(sparse_dim());

    for(size_t i = 0; i < _indices.size(); ++i)
    {
        const Coordinates &c = _indices[i];
        bool           match = false;

        for(size_t d = 0; d < coords.num_dimensions(); ++d)
        {
            if(c[d] == coords[d])
            {
                match = true;
            }
        }
        if(match)
        {
            return data + i * dense_vol * info()->element_size();
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
    const uint8_t *data = static_cast<const uint8_t *>(buffer());

    if(_indices.empty())
    {
        os << "index: [] values: []" << std::endl;
        return;
    }

    for(size_t i = 0; i < _indices.size(); ++i)
    {
        const Coordinates &coord = _indices[i];
        os << "index: [";
        for (size_t j = 0; j < coord.num_dimensions(); ++j)
        {
            os << coord[j];
            if (j < coord.num_dimensions() - 1) os << ", ";
        }
        os << "]  values: ";
        print_values(os, data, i, dense_volume(sparse_dim()));
    }
}
#endif // ARM_COMPUTE_ASSERTS_ENABLED

} // namespace arm_compute
