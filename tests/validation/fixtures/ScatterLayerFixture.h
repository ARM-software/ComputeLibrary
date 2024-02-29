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
#include "tests/Globals.h"
#include "tests/framework/Asserts.h" // Required for ARM_COMPUTE_ASSERT
#include "tests/framework/Fixture.h"
#include "tests/validation/Validation.h"
#include "tests/validation/reference/ScatterLayer.h"
#include "tests/SimpleTensor.h"
#include <random>

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
    void setup(TensorShape src_shape, TensorShape updates_shape, TensorShape indices_shape, TensorShape out_shape, DataType data_type, ScatterInfo scatter_info, QuantizationInfo src_qinfo = QuantizationInfo(), QuantizationInfo o_qinfo = QuantizationInfo())
    {
        _target    = compute_target(src_shape, updates_shape, indices_shape,  out_shape, data_type, scatter_info, src_qinfo, o_qinfo);
        _reference = compute_reference(src_shape, updates_shape, indices_shape,  out_shape, data_type,scatter_info, src_qinfo , o_qinfo);
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i, float lo = -1.f, float hi = 1.f)
    {
        switch(tensor.data_type())
        {
            case DataType::F32:
            {
                std::uniform_real_distribution<float> distribution(lo, hi);
                library->fill(tensor, distribution, i);
                break;
            }
            default:
            {
                ARM_COMPUTE_ERROR("Unsupported data type.");
            }
        }
    }

    TensorType compute_target(const TensorShape &shape_a, const TensorShape &shape_b, const TensorShape &shape_c, const TensorShape &out_shape, DataType data_type, const ScatterInfo info, QuantizationInfo a_qinfo, QuantizationInfo o_qinfo)
    {
        // 1. Create relevant tensors using ScatterInfo data structure.
        // ----------------------------------------------------
        // In order - src, updates, indices, output.
        TensorType src   = create_tensor<TensorType>(shape_a, data_type, 1, a_qinfo);
        TensorType updates   = create_tensor<TensorType>(shape_b, data_type, 1, a_qinfo);
        TensorType indices   = create_tensor<TensorType>(shape_c, DataType::U32, 1, QuantizationInfo());
        TensorType dst = create_tensor<TensorType>(out_shape, data_type, 1, o_qinfo);

        FunctionType scatter;

        // Configure operator
        scatter.configure(&src, &updates, &indices, &dst, info);

        // Assertions
        ARM_COMPUTE_ASSERT(src.info()->is_resizable());
        ARM_COMPUTE_ASSERT(updates.info()->is_resizable());
        ARM_COMPUTE_ASSERT(indices.info()->is_resizable());
        ARM_COMPUTE_ASSERT(dst.info()->is_resizable());

        // Allocate tensors
        src.allocator()->allocate();
        updates.allocator()->allocate();
        indices.allocator()->allocate();
        dst.allocator()->allocate();

        ARM_COMPUTE_ASSERT(!src.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!updates.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!indices.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!dst.info()->is_resizable());

        // Fill update (a) and indices (b) tensors.
        fill(AccessorType(src), 0);
        fill(AccessorType(updates), 1);
        fill(AccessorType(indices), 2);

        scatter.run();

        return dst;
    }

    SimpleTensor<T> compute_reference(const TensorShape &a_shape, const TensorShape &b_shape, const TensorShape &c_shape, const TensorShape &out_shape, DataType data_type,
                                      ScatterInfo info, QuantizationInfo a_qinfo, QuantizationInfo o_qinfo)
    {
        // Output Quantization not currently in use - fixture should be extended to support this.
        ARM_COMPUTE_UNUSED(o_qinfo);

        // Create reference tensors
        SimpleTensor<T> src{ a_shape, data_type, 1, a_qinfo };
        SimpleTensor<T> updates{b_shape, data_type, 1, QuantizationInfo() };
        SimpleTensor<uint32_t> indices{ c_shape, DataType::U32, 1, QuantizationInfo() };

        // Fill reference
        fill(src, 0);
        fill(updates, 1);
        fill(indices, 2);

        // Calculate individual reference.
        auto result = reference::scatter_layer<T>(src, updates, indices, out_shape, info);

        return result;
    }

    TensorType      _target{};
    SimpleTensor<T> _reference{};
};

// This fixture will use the same shape for updates as indices.
template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class ScatterValidationFixture : public ScatterGenericValidationFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    void setup(TensorShape src_shape, TensorShape indices_shape,  TensorShape out_shape, DataType data_type, ScatterFunction func, bool zero_init)
    {
        ScatterGenericValidationFixture<TensorType, AccessorType, FunctionType, T>::setup(src_shape, indices_shape, indices_shape, out_shape, data_type, ScatterInfo(func, zero_init), QuantizationInfo(), QuantizationInfo());
    }
};

} // namespace validation
} // namespace test
} // namespace arm_compute
#endif // ACL_TESTS_VALIDATION_FIXTURES_SCATTERLAYERFIXTURE_H
