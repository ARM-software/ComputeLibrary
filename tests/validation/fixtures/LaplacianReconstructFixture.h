/*
 * Copyright (c) 2018 Arm Limited.
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
#ifndef ARM_COMPUTE_TEST_LAPLACIAN_RECONSTRUCT_FIXTURE
#define ARM_COMPUTE_TEST_LAPLACIAN_RECONSTRUCT_FIXTURE

#include "arm_compute/core/IPyramid.h"
#include "arm_compute/core/PyramidInfo.h"
#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/fixtures/LaplacianPyramidFixture.h"
#include "tests/validation/reference/LaplacianPyramid.h"
#include "tests/validation/reference/LaplacianReconstruct.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename FunctionType, typename LaplacianPyramidType, typename T, typename U, typename PyramidType>
class LaplacianReconstructValidationFixture : public LaplacianPyramidValidationFixture<TensorType, AccessorType, LaplacianPyramidType, U, T, PyramidType>
{
public:
    template <typename...>
    void setup(TensorShape input_shape, BorderMode border_mode, size_t num_levels, Format format_in, Format format_out)
    {
        std::mt19937                     generator(library->seed());
        std::uniform_int_distribution<U> distribution_u8(0, 255);
        const U                          constant_border_value = distribution_u8(generator);

        using LPF = LaplacianPyramidValidationFixture<TensorType, AccessorType, LaplacianPyramidType, U, T, PyramidType>;
        LPF::setup(input_shape, border_mode, num_levels, format_out, format_in);

        // Compute target and reference values using the pyramid and lowest
        // resolution tensor output from Laplacian Pyramid kernel
        _target    = compute_target(input_shape, LPF::_target, LPF::_dst_target, border_mode, constant_border_value);
        _reference = compute_reference(LPF::_reference, LPF::_dst_reference, border_mode, constant_border_value);
    }

protected:
    template <typename V>
    void fill(V &&tensor)
    {
        library->fill_tensor_uniform(tensor, 0);
    }

    TensorType compute_target(const TensorShape &input_shape, PyramidType &pyramid, TensorType &low_res, BorderMode border_mode, U constant_border_value)
    {
        // Create tensors
        TensorType dst = create_tensor<TensorType>(input_shape, DataType::U8);

        // Create and configure function
        FunctionType laplacian_reconstruct;
        laplacian_reconstruct.configure(&pyramid, &low_res, &dst, border_mode, constant_border_value);

        // Allocate tensors
        dst.allocator()->allocate();
        ARM_COMPUTE_EXPECT(!dst.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Compute function
        laplacian_reconstruct.run();

        return dst;
    }

    SimpleTensor<U> compute_reference(const std::vector<SimpleTensor<T>> &pyramid,
                                      const SimpleTensor<T> &low_res, BorderMode border_mode, U constant_border_value)
    {
        return reference::laplacian_reconstruct<T, U>(pyramid, low_res, border_mode, constant_border_value);
    }

    TensorType      _target{};
    SimpleTensor<U> _reference{};
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_LAPLACIAN_RECONSTRUCT_FIXTURE */
