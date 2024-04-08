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
#include "arm_compute/core/TensorShape.h"

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

// NOTE: This function expects collapsed tensors as input.
// Batch dims for update/indices tensors should be collapsed into a single dim.
// Data dims should be collapsed into a single dim for both update and src tensors prior to calling this function.
template <typename T>
SimpleTensor<T> scatter_layer_internal(const SimpleTensor<T> &src, const SimpleTensor<T> &updates, const SimpleTensor<int32_t> &indices, const TensorShape &out_shape, const ScatterInfo &info)
{
    // 1. If zero initialization variable is false, copy src data to dst.
    SimpleTensor<T> dst{ out_shape, src.data_type(), 1 };
    if(!info.zero_initialization)
    {
        std::copy_n(src.data(), src.num_elements(), dst.data());
    }

    // Number of elements between each value of the dim being iterated through
    const unsigned int data_stride = updates.shape().total_size_lower(updates.shape().num_dimensions() - 1);
    const unsigned int no_output_dims = out_shape.num_dimensions();

    // Calculate output stride at given index for all output dims.
    std::vector<unsigned int> out_stride_at_idx(no_output_dims);
    for (unsigned int i = 0 ; i < no_output_dims; i++)
    {
        out_stride_at_idx[i] = out_shape.total_size_lower(i);
    }

    const unsigned int indices_x_dim = static_cast<unsigned int>(indices.shape()[0]);
    const unsigned int indices_y_dim = static_cast<unsigned int>(indices.shape()[1]);

    // 2. Iterate over indices tensor y-dim and replace sections of dst tensor with relevant areas of update tensor.
    for(unsigned int i = 0; i < indices_y_dim; i++)
    {
        // NOTE : Currently, indices.shape() == [X, Y, 1, 1], where  X is the indices dim and Y is the batch dim
        // Starting index for both the update and indices tensors.
        const unsigned int update_dim_start = i * data_stride;
        const unsigned int indices_dim_start = i * indices_x_dim;
        bool out_of_bounds = false;
        unsigned int out_offset_acc = 0;

        // Iterate over each indices value for the relevant batch and accumulate the offset.
        for(unsigned int j = 0; j < indices_x_dim; j++)
        {
            // Get first index value with i * indices_x_dim (iterating through y-dim/batch idx), then iterate through x dim by adding k
            const int index_value = indices[indices_dim_start + j];
            const unsigned int out_dim = no_output_dims - (j+1);   // Calculate corresponding output dim to current index value.
            if(index_value < static_cast<int>(out_shape[out_dim]) && index_value >= 0)
            {
                out_offset_acc += (index_value * out_stride_at_idx[out_dim]); // offset accumulation
            }
            else
            {
                out_of_bounds = true;
                break;
            }
        }

        // If not out of bounds, copy update tensor elements to output
        if(!out_of_bounds)
        {
            for (unsigned int j = 0 ; j < data_stride; j++)
            {
                dst[out_offset_acc + j] = reduce_op(dst[out_offset_acc + j], updates[update_dim_start + j], info.func);
            }
        }
    }
    return dst;
}

template <typename T>
SimpleTensor<T> scatter_layer(const SimpleTensor<T> &src, const SimpleTensor<T> &updates, const SimpleTensor<int32_t> &indices, const TensorShape &out_shape, const ScatterInfo &info)
{
    return scatter_layer_internal<T>(src, updates, indices, out_shape, info);
}

template SimpleTensor<float> scatter_layer(const SimpleTensor<float> &src, const SimpleTensor<float> &updates, const SimpleTensor<int32_t> &indices, const TensorShape &out_shape, const ScatterInfo &info);

} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
