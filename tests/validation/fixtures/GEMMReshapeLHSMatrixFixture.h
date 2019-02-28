/*
 * Copyright (c) 2018 ARM Limited.
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
#ifndef ARM_COMPUTE_TEST_GEMMRESHAPELHSMATRIX_FIXTURE
#define ARM_COMPUTE_TEST_GEMMRESHAPELHSMATRIX_FIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/Helpers.h"
#include "tests/validation/reference/GEMMReshapeLHSMatrix.h"
#include "tests/validation/reference/Utils.h"

#include <random>

namespace arm_compute
{
namespace test
{
namespace validation
{
using namespace arm_compute::misc::shape_calculator;

template <typename TensorType, typename AccessorType, typename FunctionType, typename T, bool reinterpret_input_as_3d = false>
class GEMMReshapeLHSMatrixValidationFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape shape_in, unsigned int batch_size, DataType data_type, unsigned int m0, unsigned int k0, unsigned int v0, bool interleave, bool transpose)
    {
        GEMMLHSMatrixInfo lhs_info;
        lhs_info.m0         = m0;
        lhs_info.k0         = k0;
        lhs_info.v0         = v0;
        lhs_info.interleave = interleave;
        lhs_info.transpose  = transpose;

        // Set the tensor shape
        const TensorShape shape_src(shape_in[0],
                                    shape_in[1],
                                    reinterpret_input_as_3d ? shape_in[2] : batch_size,
                                    reinterpret_input_as_3d ? batch_size : 1);

        _target    = compute_target(shape_src, data_type, lhs_info);
        _reference = compute_reference(shape_src, data_type, lhs_info);
    }

protected:
    template <typename U>
    void fill(U &&tensor)
    {
        library->fill_tensor_uniform(tensor, 0);
    }

    TensorType compute_target(TensorShape input_shape, DataType data_type, const GEMMLHSMatrixInfo &lhs_info)
    {
        // Create tensors
        TensorType src = create_tensor<TensorType>(input_shape, data_type, 1);
        TensorType dst;

        // The output tensor will be auto-initialized within the function

        // Create and configure function
        FunctionType gemm_lhs_reshape;
        gemm_lhs_reshape.configure(&src, &dst, lhs_info, reinterpret_input_as_3d);

        ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Allocate tensors
        src.allocator()->allocate();
        dst.allocator()->allocate();

        ARM_COMPUTE_EXPECT(!src.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!dst.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Fill tensors
        fill(AccessorType(src));

        // Compute GEMM LHS matrix reshape function
        gemm_lhs_reshape.run();

        return dst;
    }

    SimpleTensor<T> compute_reference(const TensorShape &input_shape, DataType data_type, const GEMMLHSMatrixInfo &lhs_info)
    {
        TensorShape src_shape = input_shape;

        // If the input has to be reinterpreted as 3D, collapse the second dimension with the 3rd
        if(reinterpret_input_as_3d)
        {
            src_shape.collapse(2U, 1U);
        }

        // Create reference
        SimpleTensor<T> src{ src_shape, data_type, 1 };

        // Fill reference
        fill(src);

        TensorShape output_shape = compute_lhs_reshaped_shape(TensorInfo(input_shape, 1, data_type), lhs_info, reinterpret_input_as_3d);

        return reference::gemm_reshape_lhs_matrix<T>(src, output_shape, lhs_info);
    }

    TensorType      _target{};
    SimpleTensor<T> _reference{};
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_GEMMRESHAPELHSMATRIX_FIXTURE */