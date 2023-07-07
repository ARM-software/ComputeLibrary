/*
 * Copyright (c) 2023 Arm Limited.
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

#include "src/cl/CLTensorArgument.h"
#include "ckw/Error.h"
#include "src/cl/CLHelpers.h"
#include "src/types/TensorComponentType.h"

#include <algorithm>
#include <vector>

namespace ckw
{
CLTensorArgument::CLTensorArgument(const std::string &name, const TensorInfo &info, bool return_dims_by_value)
{
    _return_dims_by_value = return_dims_by_value;
    _basename             = name;
    _info                 = info;
}

TileVariable CLTensorArgument::component(TensorComponentType x)
{
    if(_return_dims_by_value)
    {
        uint32_t component_type = static_cast<uint32_t>(x);

        const bool is_dimension         = (component_type & static_cast<uint32_t>(TensorComponentBitmask::Dimension)) != 0;
        const bool is_folded_dimensions = (component_type & static_cast<uint32_t>(TensorComponentBitmask::FoldedDimensions)) != 0;

        constexpr auto bitmask_all     = static_cast<uint32_t>(TensorComponentIndexBitmask::All);
        constexpr auto bitmask_index_0 = static_cast<uint32_t>(TensorComponentIndexBitmask::Index0);
#ifdef COMPUTE_KERNEL_WRITER_ASSERTS_ENABLED
        constexpr auto bitmask_index_1 = static_cast<uint32_t>(TensorComponentIndexBitmask::Index1);
        constexpr auto bitmask_index_2 = static_cast<uint32_t>(TensorComponentIndexBitmask::Index2);
        constexpr auto bitmask_index_3 = static_cast<uint32_t>(TensorComponentIndexBitmask::Index3);
#endif // COMPUTE_KERNEL_WRITER_ASSERTS_ENABLED

        // Make sure that the encoding of component type hasn't changed and each nibble is 4 bits apart.
        CKW_ASSERT(bitmask_all == (bitmask_index_0 | bitmask_index_1 | bitmask_index_2 | bitmask_index_3));
        CKW_ASSERT(bitmask_index_0 == bitmask_index_1 >> 4);
        CKW_ASSERT(bitmask_index_1 == bitmask_index_2 >> 4);
        CKW_ASSERT(bitmask_index_2 == bitmask_index_3 >> 4);

        // If we have a dimension or folded dimensions, we can return the corresponding value if it is not dynamic (not equal to -1)
        if(is_dimension == true || is_folded_dimensions == true)
        {
            component_type = component_type & bitmask_all;

            int32_t idx = 1;
            for(int32_t i = 0; i < tensor_component_index_max_count; ++i)
            {
                uint32_t dim_idx = component_type & bitmask_index_0;

                if(dim_idx == 0)
                {
                    // Stop at the first nibble containing 0
                    break;
                }

                // Subtract - 1. Please refer to the TensorComponentIndexBitmask documentation
                dim_idx -= 1;

                // Get the dimension value
                const int32_t dim_val = _info.shape()[dim_idx];

                if(dim_val == kDynamicTensorDimensionValue)
                {
                    // We cannot return the dimension by value if it is dynamic.
                    // Therefore, force the idx variable to kDynamicTensorDimensionValue and break the loop.
                    idx = kDynamicTensorDimensionValue;
                    break;
                }

                idx *= dim_val;

                // Go to the next nibble
                component_type >>= 4;
            }

            if(idx != kDynamicTensorDimensionValue)
            {
                TileVariable t;
                t.str      = std::to_string(idx);
                t.desc.dt  = DataType::Uint32;
                t.desc.len = 1;
                return t;
            }
        }
    }

    auto it = std::find(_components_used.begin(), _components_used.end(), x);

    // Add to the list of used components if not present yet
    if(it == _components_used.end())
    {
        _components_used.push_back(x);
    }

    TileVariable t;
    t.str      = create_component_name(x);
    t.desc.dt  = DataType::Int32;
    t.desc.len = 1;
    return t;
}

TensorStorageVariable CLTensorArgument::storage(TensorStorageType x)
{
    if(std::find(_storages_used.begin(), _storages_used.end(), x) == _storages_used.end())
    {
        _storages_used.push_back(x);
    }

    TensorStorageVariable t;
    t.val  = create_storage_name(x);
    t.type = cl_get_variable_storagetype_as_string(x);

    return t;
}

std::string CLTensorArgument::create_storage_name(TensorStorageType x) const
{
    std::string var_name = _basename;

    switch(x)
    {
        case TensorStorageType::BufferUint8Ptr:
            var_name += "_ptr";
            break;
        case TensorStorageType::Texture2dReadOnly:
        case TensorStorageType::Texture2dWriteOnly:
            var_name += "_img2d";
            break;
        default:
            CKW_ASSERT_FAILED_MSG("Unsupported tensor storage");
            return "";
    }

    return var_name;
}

std::string CLTensorArgument::create_component_name(TensorComponentType x) const
{
    std::string var_name = _basename;

    switch(x)
    {
        case TensorComponentType::OffsetFirstElement:
            var_name += "_offset_first_element";
            break;
        case TensorComponentType::Stride0:
            var_name += "_stride0";
            break;
        case TensorComponentType::Stride1:
            var_name += "_stride1";
            break;
        case TensorComponentType::Stride2:
            var_name += "_stride2";
            break;
        case TensorComponentType::Stride3:
            var_name += "_stride3";
            break;
        case TensorComponentType::Stride4:
            var_name += "_stride4";
            break;
        case TensorComponentType::Dim0:
            var_name += "_dim0";
            break;
        case TensorComponentType::Dim1:
            var_name += "_dim1";
            break;
        case TensorComponentType::Dim2:
            var_name += "_dim2";
            break;
        case TensorComponentType::Dim3:
            var_name += "_dim3";
            break;
        case TensorComponentType::Dim4:
            var_name += "_dim4";
            break;
        case TensorComponentType::Dim1xDim2:
            var_name += "_dim1xdim2";
            break;
        case TensorComponentType::Dim2xDim3:
            var_name += "_dim2xdim3";
            break;
        case TensorComponentType::Dim1xDim2xDim3:
            var_name += "_dim1xdim2xdim3";
            break;
        default:
            COMPUTE_KERNEL_WRITER_ERROR_ON_MSG("Unsupported tensor component");
            return "";
    }

    return var_name;
}

std::vector<TensorStorageVariable> CLTensorArgument::storages() const
{
    std::vector<TensorStorageVariable> storages;
    for(auto &val : _storages_used)
    {
        TensorStorageVariable t;
        t.val  = create_storage_name(val);
        t.type = cl_get_variable_storagetype_as_string(val);
        storages.push_back(t);
    }

    return storages;
}

std::vector<TileVariable> CLTensorArgument::components() const
{
    std::vector<TileVariable> components;

    for(auto &val : _components_used)
    {
        TileVariable t;
        t.str      = create_component_name(val);
        t.desc.dt  = DataType::Int32;
        t.desc.len = 1;
        components.push_back(t);
    }

    return components;
}
} // namespace ckw
