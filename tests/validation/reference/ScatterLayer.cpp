/*
 * Copyright (c) 2024 Arm Limited.
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
#include "ScatterLayer.h"
#include "tests/validation/Helpers.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace reference
{
namespace
{

template <typename T>
T reduce_op(const T &current,const T &update,const ScatterFunction func)
{
    switch(func)
    {
        case ScatterFunction::Update:
            return update;
            break;
        case ScatterFunction::Add:
            return current + update;
            break;
        case ScatterFunction::Sub:
            return current - update;
            break;
        case ScatterFunction::Max:
            return std::max(current, update);
            break;
        case ScatterFunction::Min:
            return std::min(current, update);
            break;
        default:
            ARM_COMPUTE_ERROR("Unsupported Scatter function");
            break;
    }
}

template float reduce_op(const float &current,const float &update,const ScatterFunction func);
}

// Note : This function currently only supports 1D src, 1D updates, 2D indices, 1D output tensors.
template <typename T>
SimpleTensor<T> scatter_layer_internal(const SimpleTensor<T> &src, const SimpleTensor<T> &updates, const SimpleTensor<uint32_t> &indices, const TensorShape &out_shape, const ScatterInfo &info)
{
    SimpleTensor<T> dst{ out_shape, src.data_type(), 1 };

    // 1. If zero initialization variable is true, fill dst with 0 values. Else copy src data to dst.
    if(info.zero_initialization)
    {
        for (int i = 0; i < src.num_elements(); ++i)
        {
            dst[i] = static_cast<T>(0);
        }
    }
    else
    {
        std::copy_n(src.data(), src.num_elements(), dst.data());
    }

    // 2. Get max index of output tensor, then iterate over index tensor.
    const auto x_bound = dst.shape().x();


    for(int i = 0; i < indices.num_elements(); ++i)
    {
        // 3. Check whether index is out of bounds for dst, if not then apply reduce op.
        const auto index = indices[i];
        if (index < x_bound) // Note : index is always >= 0 as datatype is unsigned.
        {
            dst[index] = reduce_op(dst[index], updates[i], info.func);
        }
    }
    return dst;
}

template <typename T>
SimpleTensor<T> scatter_layer(const SimpleTensor<T> &src, const SimpleTensor<T> &updates, const SimpleTensor<uint32_t> &indices, const TensorShape &out_shape, const ScatterInfo &info)
{
    return scatter_layer_internal<T>(src, updates, indices, out_shape, info);
}

template SimpleTensor<float> scatter_layer(const SimpleTensor<float> &src, const SimpleTensor<float> &updates, const SimpleTensor<uint32_t> &indices, const TensorShape &out_shape, const ScatterInfo &info);

} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
