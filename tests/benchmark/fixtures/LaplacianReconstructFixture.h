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
#ifndef ARM_COMPUTE_TEST_LAPLACIAN_RECONSTRUCT_FIXTURE
#define ARM_COMPUTE_TEST_LAPLACIAN_RECONSTRUCT_FIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "tests/Globals.h"
#include "tests/Utils.h"
#include "tests/benchmark/fixtures/LaplacianPyramidFixture.h"
#include "tests/framework/Fixture.h"

namespace arm_compute
{
namespace test
{
namespace benchmark
{
template <typename TensorType, typename Function, typename Accessor, typename LaplacianPyramidFunc, typename PyramidType>
class LaplacianReconstructFixture : public LaplacianPyramidFixture<TensorType, LaplacianPyramidFunc, Accessor, PyramidType>
{
public:
    template <typename...>
    void setup(TensorShape input_shape, BorderMode border_mode, size_t num_levels, Format format_in, Format format_out)
    {
        const uint8_t constant_border_value = 0;

        LPF::setup(input_shape, border_mode, num_levels, format_out, format_in);
        LPF::run();

        // Create tensor
        dst = create_tensor<TensorType>(input_shape, DataType::U8);

        laplacian_reconstruct_func.configure(&(LPF::pyramid), &(LPF::dst), &dst, border_mode, constant_border_value);

        dst.allocator()->allocate();
    }

    void run()
    {
        laplacian_reconstruct_func.run();
    }

    void sync()
    {
        sync_if_necessary<TensorType>();
        sync_tensor_if_necessary<TensorType>(dst);
    }

private:
    TensorType dst{};
    Function   laplacian_reconstruct_func{};

    using LPF = LaplacianPyramidFixture<TensorType, LaplacianPyramidFunc, Accessor, PyramidType>;
};
} // namespace benchmark
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_LAPLACIAN_RECONSTRUCT_FIXTURE */
