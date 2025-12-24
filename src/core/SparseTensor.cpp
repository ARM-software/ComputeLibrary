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
#include "arm_compute/core/SparseTensor.h"

#include "arm_compute/core/Error.h"

namespace arm_compute
{
SparseTensor::SparseTensor(size_t dim, size_t sparse_dim) : _total_dim(dim), _sparse_dim(sparse_dim)
{
}

size_t SparseTensor::sparse_dim() const
{
    return _sparse_dim;
}

size_t SparseTensor::dense_dim() const
{
    return _total_dim - _sparse_dim;
}

float SparseTensor::sparsity() const
{
    return 1.0f - density();
}

float SparseTensor::density() const
{
    return static_cast<float>(nnz()) / static_cast<float>(info()->total_size());
}

size_t SparseTensor::dim() const
{
    return _total_dim;
}

bool SparseTensor::is_hybrid() const
{
    return dense_dim() > 0;
}

uint32_t SparseTensor::dense_volume(size_t sparse_dim) const
{
    const auto &ts = info()->tensor_shape();
    return std::accumulate(ts.begin() + sparse_dim, ts.end(), 1, std::multiplies<int>());
}

bool SparseTensor::has_non_zero_elements(uint8_t *arr, size_t len, size_t element_size, predicate_t is_non_zero) const
{
    for(size_t i = 0; i < len; i += element_size)
    {
        if(is_non_zero(arr + i))
        {
            return true;
        }
    }
    return false;
}

std::function<bool(const void *)> SparseTensor::make_is_nonzero_predicate(DataType dt) const
{
    switch (dt)
    {
    case DataType::F32:
        return [](const void *ptr) {
            return *(static_cast<const float *>(ptr)) != 0.0f;
        };
    case DataType::F16: // raw bitwise comparison
    case DataType::U16:
    case DataType::QSYMM16:
    case DataType::QASYMM16:
        return [](const void *ptr) {
            return *(static_cast<const uint16_t *>(ptr)) != 0;
        };
    case DataType::S32:
        return [](const void *ptr) {
            return *(static_cast<const int32_t *>(ptr)) != 0;
        };
    case DataType::S16:
        return [](const void *ptr) {
            return *(static_cast<const int16_t *>(ptr)) != 0;
        };
    case DataType::U32:
        return [](const void *ptr) {
            return *(static_cast<const uint32_t *>(ptr)) != 0;
        };
    case DataType::U8:
    case DataType::QSYMM8:
    case DataType::QASYMM8:
    case DataType::QSYMM8_PER_CHANNEL:
        return [](const void *ptr) {
            return *(static_cast<const uint8_t *>(ptr)) != 0;
        };
    case DataType::S8:
    case DataType::QASYMM8_SIGNED:
        return [](const void *ptr) {
            return *(static_cast<const int8_t *>(ptr)) != 0;
        };
    default:
        throw std::runtime_error("Unsupported DataType in make_is_nonzero_predicate()");
    }
}

void SparseTensor::print_values(std::ostream &os, const uint8_t *data, size_t offset, size_t count) const
{
    const size_t   element_size = info()->element_size();
    const uint8_t *block_ptr = data + offset * count * element_size;

    os << "[";
    for(size_t j = 0; j < count; ++j)
    {
        const void *value_ptr = block_ptr + j * element_size;

        switch(info()->data_type())
        {
            case DataType::U8:
                os << static_cast<int>(*reinterpret_cast<const uint8_t *>(value_ptr));
                break;
            case DataType::S8:
                os << static_cast<int>(*reinterpret_cast<const int8_t *>(value_ptr));
                break;
            case DataType::U32:
                os << *reinterpret_cast<const uint32_t *>(value_ptr);
                break;
            case DataType::S32:
                os << *reinterpret_cast<const int32_t *>(value_ptr);
                break;
            case DataType::F16:
                os << static_cast<float>(*reinterpret_cast<const half *>(value_ptr));
                break;
            case DataType::F32:
                os << *reinterpret_cast<const float *>(value_ptr);
                break;
            default:
                os << "<not supported by print>";
        }

        if(j < count - 1) os << ", ";
    }
    os << "]" << std::endl;
}
} // namespace arm_compute
