/*
 * Copyright (c) 2017 ARM Limited.
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
#ifndef ARM_COMPUTE_TEST_MATRIXMULTIPLY_GEMM_DATASET
#define ARM_COMPUTE_TEST_MATRIXMULTIPLY_GEMM_DATASET

#include "tests/datasets/GEMMDataset.h"

#include "utils/TypePrinter.h"

#include "arm_compute/core/TensorShape.h"

namespace arm_compute
{
namespace test
{
namespace datasets
{
class MatrixMultiplyGEMMDataset final : public GEMMDataset
{
public:
    MatrixMultiplyGEMMDataset()
    {
        add_config(TensorShape(1024U, 1U), TensorShape(1000U, 1024U), TensorShape(1000U, 1U), TensorShape(1000U, 1U), 1.0f, 0.0f);
        add_config(TensorShape(256U, 784U), TensorShape(64U, 256U), TensorShape(64U, 784U), TensorShape(64U, 784U), 1.0f, 0.0f);
        add_config(TensorShape(1152U, 2704U), TensorShape(256U, 1152U), TensorShape(256U, 2704U), TensorShape(256U, 2704U), 1.0f, 0.0f);
        add_config(TensorShape{ 128U, 256U }, TensorShape{ 128U, 128U }, TensorShape{ 128U, 256U }, TensorShape{ 128U, 256U }, 1.1f, 0.0f);
        add_config(TensorShape{ 512U, 128U }, TensorShape{ 512U, 512U }, TensorShape{ 512U, 128U }, TensorShape{ 512U, 128U }, 0.5f, 0.0f);
        add_config(TensorShape{ 256U, 256U }, TensorShape{ 256U, 256U }, TensorShape{ 256U, 256U }, TensorShape{ 256U, 256U }, 0.5f, 0.0f);
        add_config(TensorShape{ 128U, 256U }, TensorShape{ 128U, 128U }, TensorShape{ 128U, 256U }, TensorShape{ 128U, 256U }, 1.1f, 1.5f);
        add_config(TensorShape{ 512U, 128U }, TensorShape{ 512U, 512U }, TensorShape{ 512U, 128U }, TensorShape{ 512U, 128U }, 1.2f, 1.1f);
        add_config(TensorShape{ 256U, 256U }, TensorShape{ 256U, 256U }, TensorShape{ 256U, 256U }, TensorShape{ 256U, 256U }, 0.5f, 1.3f);
    }
};
} // namespace datasets
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_MATRIXMULTIPLY_GEMM_DATASET */
