/*
 * Copyright (c) 2017-2018 ARM Limited.
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
#ifndef ARM_COMPUTE_TEST_GEMMLOWPFIXTURE
#define ARM_COMPUTE_TEST_GEMMLOWPFIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "tests/Globals.h"
#include "tests/Utils.h"
#include "tests/framework/Fixture.h"

namespace arm_compute
{
namespace test
{
/** Fixture that can be used for NEON and CL */
template <typename TensorType, typename Function, typename Accessor>
class GEMMLowpMatrixMultiplyCoreFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape shape_a, TensorShape shape_b, TensorShape shape_c, TensorShape shape_dst, float alpha, float beta)
    {
        ARM_COMPUTE_UNUSED(shape_c);
        ARM_COMPUTE_UNUSED(alpha);
        ARM_COMPUTE_UNUSED(beta);

        // Note: The offsets for matrix A and matrix B are set to 0 in order to skip the computation for the offset contribution

        // Create tensors
        a = create_tensor<TensorType>(shape_a, DataType::QASYMM8, 1, 0, QuantizationInfo(1.0f / 255.0f, 0));
        b = create_tensor<TensorType>(shape_b, DataType::QASYMM8, 1, 0, QuantizationInfo(1.0f / 255.0f, 0));
        c = create_tensor<TensorType>(shape_dst, DataType::S32, 1, 0, QuantizationInfo(1.0f / 255.0f, 0));

        // Create and configure function
        gemmlowp.configure(&a, &b, &c);

        // Allocate tensors
        a.allocator()->allocate();
        b.allocator()->allocate();
        c.allocator()->allocate();
    }
    void run()
    {
        gemmlowp.run();
    }

    void sync()
    {
        sync_if_necessary<TensorType>();
        sync_tensor_if_necessary<TensorType>(c);
    }

    void teardown()
    {
        a.allocator()->free();
        b.allocator()->free();
        c.allocator()->free();
    }

private:
    TensorType a{};
    TensorType b{};
    TensorType c{};
    Function   gemmlowp{};
};

} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_GEMMLOWPFIXTURE */
