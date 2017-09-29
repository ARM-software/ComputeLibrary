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
#ifndef ARM_COMPUTE_TEST_LARGE_GEMM_DATASET
#define ARM_COMPUTE_TEST_LARGE_GEMM_DATASET

#include "tests/datasets/GEMMDataset.h"

#include "utils/TypePrinter.h"

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"

namespace arm_compute
{
namespace test
{
namespace datasets
{
class LargeGEMMDataset final : public GEMMDataset
{
public:
    LargeGEMMDataset()
    {
        add_config(TensorShape(923U, 429U), TensorShape(871U, 923U), TensorShape(871U, 429U), TensorShape(871U, 429U), 1.0f, 0.0f);
        add_config(TensorShape(1021U, 1U), TensorShape(783U, 1021U), TensorShape(783U, 1U), TensorShape(783U, 1U), 1.0f, 0.0f);
        add_config(TensorShape(681U, 1023U), TensorShape(213U, 681U), TensorShape(213U, 1023U), TensorShape(213U, 1023U), 0.2f, 1.2f);
        add_config(TensorShape(941U, 1U), TensorShape(623U, 941U), TensorShape(623U, 1U), TensorShape(623U, 1U), 0.4f, 0.7f);
    }
};
} // namespace datasets
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_LARGE_GEMM_DATASET */
