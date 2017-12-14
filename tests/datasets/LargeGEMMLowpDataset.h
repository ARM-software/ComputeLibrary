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
#ifndef ARM_COMPUTE_TEST_LARGE_GEMMLOWP_DATASET
#define ARM_COMPUTE_TEST_LARGE_GEMMLOWP_DATASET

#include "tests/datasets/GEMMLowpDataset.h"

#include "utils/TypePrinter.h"

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"

namespace arm_compute
{
namespace test
{
namespace datasets
{
class LargeGEMMLowpDataset final : public GEMMLowpDataset
{
public:
    LargeGEMMLowpDataset()
    {
        add_config(TensorShape(923U, 1U), TensorShape(871U, 923U), TensorShape(871U, 1U), 0, 0);
        add_config(TensorShape(923U, 429U), TensorShape(871U, 923U), TensorShape(871U, 429U), 0, 0);
        add_config(TensorShape(873U, 7U), TensorShape(784U, 873U), TensorShape(784U, 7U), -1, 3);
        add_config(TensorShape(873U, 513U), TensorShape(784U, 873U), TensorShape(784U, 513U), 0, 4);
        add_config(TensorShape(697U, 872U), TensorShape(563U, 697U), TensorShape(563U, 872U), -2, 0);
        add_config(TensorShape(1021U, 973U), TensorShape(783U, 1021U), TensorShape(783U, 973U), 5, 13);
        add_config(TensorShape(681U, 1023U), TensorShape(213U, 681U), TensorShape(213U, 1023U), -3, -2);
        add_config(TensorShape(941U, 1011U), TensorShape(623U, 941U), TensorShape(623U, 1011U), -9, 1);
    }
};
} // namespace datasets
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_LARGE_GEMMLOWP_DATASET */
