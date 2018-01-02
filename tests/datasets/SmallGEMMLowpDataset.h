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
#ifndef ARM_COMPUTE_TEST_SMALL_GEMMLOWP_DATASET
#define ARM_COMPUTE_TEST_SMALL_GEMMLOWP_DATASET

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
class SmallGEMMLowpDataset final : public GEMMLowpDataset
{
public:
    SmallGEMMLowpDataset()
    {
        add_config(TensorShape(21U, 1U), TensorShape(43U, 21U), TensorShape(43U, 1U), 0, 0);
        add_config(TensorShape(21U, 13U), TensorShape(33U, 21U), TensorShape(33U, 13U), 0, 0);
        add_config(TensorShape(31U, 3U), TensorShape(72U, 31U), TensorShape(72U, 3U), -2, 13);
        add_config(TensorShape(52U, 13U), TensorShape(33U, 52U), TensorShape(33U, 13U), 0, 4);
        add_config(TensorShape(52U, 26U), TensorShape(33U, 52U), TensorShape(33U, 26U), -2, 0);
        add_config(TensorShape(31U, 27U), TensorShape(23U, 31U), TensorShape(23U, 27U), 5, 13);
        add_config(TensorShape(38U, 43U), TensorShape(21U, 38U), TensorShape(21U, 43U), -3, -2);
        add_config(TensorShape(32U, 72U), TensorShape(17U, 32U), TensorShape(17U, 72U), -9, 1);
    }
};
} // namespace datasets
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_SMALL_GEMMLOWP_DATASET */
