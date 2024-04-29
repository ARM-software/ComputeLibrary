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
#ifndef ACL_TESTS_VALIDATION_FIXTURES_SCATTERLAYERFIXTURE_H
#define ACL_TESTS_VALIDATION_FIXTURES_SCATTERLAYERFIXTURE_H

#include "arm_compute/core/Utils.h"
#include "arm_compute/runtime/CL/CLTensorAllocator.h"
#include "tests/Globals.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/Helpers.h"
#include "tests/validation/Validation.h"
#include "tests/validation/reference/ScatterLayer.h"
#include "tests/SimpleTensor.h"

#include <random>
#include <cstdint>

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class ScatterGenericValidationFixture : public framework::Fixture
{
public:
    void setup(TensorShape src_shape, TensorShape updates_shape, TensorShape indices_shape,
        TensorShape out_shape, DataType data_type, ScatterInfo scatter_info, bool inplace,
        QuantizationInfo src_qinfo = QuantizationInfo(), QuantizationInfo o_qinfo = QuantizationInfo())
    {
        // this is for improving randomness across tests
        _hash = src_shape[0] + src_shape[1] + src_shape[2] + src_shape[3] + src_shape[4] + src_shape[5]
              + updates_shape[0] + updates_shape[1] + updates_shape[2] + updates_shape[3]
              + updates_shape[4] + updates_shape[5]
              + indices_shape[0] + indices_shape[1] + indices_shape[2] + indices_shape[3];

        _target    = compute_target(src_shape, updates_shape, indices_shape,  out_shape, data_type, scatter_info, inplace, src_qinfo, o_qinfo);
        _reference = compute_reference(src_shape, updates_shape, indices_shape,  out_shape, data_type,scatter_info, src_qinfo , o_qinfo);
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i)
    {
        switch(tensor.data_type())
        {
            case DataType::F32:
            case DataType::F16:
            {
                std::uniform_real_distribution<float> distribution(-10.f, 10.f);
                library->fill(tensor, distribution, i);
                break;
            }
            case DataType::S32:
            case DataType::S16:
            case DataType::S8:
            {
                std::uniform_int_distribution<int32_t> distribution(-100, 100);
                library->fill(tensor, distribution, i);
                break;
            }
            case DataType::U32:
            case DataType::U16:
            case DataType::U8:
            {
                std::uniform_int_distribution<uint32_t> distribution(0, 200);
                library->fill(tensor, distribution, i);
                break;
            }
            default:
            {
                ARM_COMPUTE_ERROR("Unsupported data type.");
            }
        }
    }

    // This is used to fill indices tensor with S32 datatype.
    // Used to prevent ONLY having values that are out of bounds.
    template <typename U>
    void fill_indices(U &&tensor, int i, const TensorShape &shape)
    {
        // Calculate max indices the shape should contain. Add an arbitrary value to allow testing for some out of bounds values (In this case min dimension)
        const int32_t max = std::max({shape[0] , shape[1], shape[2]});
        library->fill_tensor_uniform(tensor, i, static_cast<int32_t>(-2), static_cast<int32_t>(max));
    }

    TensorType compute_target(const TensorShape &shape_a, const TensorShape &shape_b, const TensorShape &shape_c,
        const TensorShape &out_shape, DataType data_type, const ScatterInfo info, bool inplace,
        QuantizationInfo a_qinfo, QuantizationInfo o_qinfo)
    {
        // 1. Create relevant tensors using ScatterInfo data structure.
        // ----------------------------------------------------
        // In order - src, updates, indices, output.
        TensorType src   = create_tensor<TensorType>(shape_a, data_type, 1, a_qinfo);
        TensorType updates   = create_tensor<TensorType>(shape_b, data_type, 1, a_qinfo);
        TensorType indices   = create_tensor<TensorType>(shape_c, DataType::S32, 1, QuantizationInfo());
        TensorType dst = create_tensor<TensorType>(out_shape, data_type, 1, o_qinfo);

        FunctionType scatter;

        // Configure operator
        // When scatter_info.zero_initialization is true, pass nullptr for src
        // because dst does not need to be initialized with src values.
        if(info.zero_initialization)
        {
            scatter.configure(nullptr, &updates, &indices, &dst, info);
        }
        else
        {
            if(inplace)
            {
                scatter.configure(&src, &updates, &indices, &src, info);
            }
            else
            {
                scatter.configure(&src, &updates, &indices, &dst, info);
            }
        }

        // Assertions
        ARM_COMPUTE_ASSERT(src.info()->is_resizable());
        ARM_COMPUTE_ASSERT(updates.info()->is_resizable());
        ARM_COMPUTE_ASSERT(indices.info()->is_resizable());
        ARM_COMPUTE_ASSERT(dst.info()->is_resizable());

        add_padding_x({ &src, &updates, &indices});

        if(!inplace)
        {
            add_padding_x({ &dst });
        }

        // Allocate tensors
        src.allocator()->allocate();
        updates.allocator()->allocate();
        indices.allocator()->allocate();

        if(!inplace)
        {
            dst.allocator()->allocate();
        }

        ARM_COMPUTE_ASSERT(!src.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!updates.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!indices.info()->is_resizable());

        if(!inplace)
        {
            ARM_COMPUTE_ASSERT(!dst.info()->is_resizable());
        }

        // Fill update (a) and indices (b) tensors.
        fill(AccessorType(src), 0 + _hash);
        fill(AccessorType(updates), 1+ _hash);
        fill_indices(AccessorType(indices), 2 + _hash, out_shape);

        scatter.run();

        if(inplace)
        {
            return src;
        }
        else
        {
            return dst;
        }
    }

    SimpleTensor<T> compute_reference(const TensorShape &a_shape, const TensorShape &b_shape, const TensorShape &c_shape,
        const TensorShape &out_shape, DataType data_type, ScatterInfo info, QuantizationInfo a_qinfo, QuantizationInfo o_qinfo)
    {
        // Output Quantization not currently in use - fixture should be extended to support this.
        ARM_COMPUTE_UNUSED(o_qinfo);
        TensorShape src_shape = a_shape;
        TensorShape updates_shape = b_shape;
        TensorShape indices_shape = c_shape;

        // 1. Collapse batch index into a single dim if necessary for update tensor and indices tensor.
        if(c_shape.num_dimensions() >= 3)
        {
            indices_shape = indices_shape.collapsed_from(1);
            updates_shape = updates_shape.collapsed_from(updates_shape.num_dimensions() - 2); // Collapses from last 2 dims
        }

        // 2. Collapse data dims into a single dim.
        //    Collapse all src dims into 2 dims. First one holding data, the other being the index we iterate over.
        src_shape.collapse(updates_shape.num_dimensions() - 1);     // Collapse all data dims into single dim.
        src_shape = src_shape.collapsed_from(1);                    // Collapse all index dims into a single dim
        updates_shape.collapse(updates_shape.num_dimensions() - 1); // Collapse data dims (all except last dim which is batch dim)

        // Create reference tensors
        SimpleTensor<T> src{ a_shape, data_type, 1, a_qinfo };
        SimpleTensor<T> updates{b_shape, data_type, 1, QuantizationInfo() };
        SimpleTensor<int32_t> indices{ c_shape, DataType::S32, 1, QuantizationInfo() };

        // Fill reference
        fill(src, 0 + _hash);
        fill(updates, 1 + _hash);
        fill_indices(indices, 2 + _hash, out_shape);

        // Calculate individual reference.
        return reference::scatter_layer<T>(src, updates, indices, out_shape, info);
    }

    TensorType      _target{};
    SimpleTensor<T> _reference{};
    int32_t _hash{};
};

// This fixture will use the same shape for updates as indices.
template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class ScatterValidationFixture : public ScatterGenericValidationFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    void setup(TensorShape src_shape, TensorShape update_shape, TensorShape indices_shape,
        TensorShape out_shape, DataType data_type, ScatterFunction func, bool zero_init, bool inplace)
    {
        ScatterGenericValidationFixture<TensorType, AccessorType, FunctionType, T>::setup(src_shape, update_shape,
            indices_shape, out_shape, data_type, ScatterInfo(func, zero_init), inplace,
            QuantizationInfo(), QuantizationInfo());
    }
};

} // namespace validation
} // namespace test
} // namespace arm_compute
#endif // ACL_TESTS_VALIDATION_FIXTURES_SCATTERLAYERFIXTURE_H
