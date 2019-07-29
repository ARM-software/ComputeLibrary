/*
 * Copyright (c) 2017-2019 ARM Limited.
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
#ifndef ARM_COMPUTE_TEST_SMALL_GEMM_DATASET
#define ARM_COMPUTE_TEST_SMALL_GEMM_DATASET

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
class SmallGEMMDataset final : public GEMMDataset
{
public:
    SmallGEMMDataset()
    {
        add_config(TensorShape(21U, 13U), TensorShape(33U, 21U), TensorShape(33U, 13U), TensorShape(33U, 13U), 1.0f, 0.0f);
        add_config(TensorShape(31U, 1U), TensorShape(23U, 31U), TensorShape(23U, 1U), TensorShape(23U, 1U), 1.0f, 0.0f);
        add_config(TensorShape(31U, 1U), TensorShape(23U, 31U), TensorShape(23U, 1U), TensorShape(23U, 1U), 1.0f, 1.0f);
        add_config(TensorShape(8U, 2U), TensorShape(16U, 8U), TensorShape(16U, 2U), TensorShape(16U, 2U), 1.0f, 0.0f);
        add_config(TensorShape(38U, 12U), TensorShape(21U, 38U), TensorShape(21U, 12U), TensorShape(21U, 12U), 0.2f, 1.2f);
        add_config(TensorShape(32U, 1U), TensorShape(17U, 32U), TensorShape(17U, 1U), TensorShape(17U, 1U), 0.4f, 0.7f);
    }
};
class SmallGEMMOutput3DDataset final : public GEMMDataset
{
public:
    SmallGEMMOutput3DDataset()
    {
        add_config(TensorShape(21U, 14U), TensorShape(34U, 21U), TensorShape(34U), TensorShape(34U, 7U, 2U), 1.0f, 0.0f);
        add_config(TensorShape(31U, 1U), TensorShape(23U, 31U), TensorShape(23U), TensorShape(23U, 1U, 1U), 1.0f, 0.0f);
        add_config(TensorShape(38U, 12U), TensorShape(21U, 38U), TensorShape(21U), TensorShape(21U, 4U, 3U), 0.2f, 1.2f);
        add_config(TensorShape(32U, 1U), TensorShape(17U, 32U), TensorShape(17U), TensorShape(17U, 1U, 1U), 0.4f, 0.7f);
        add_config(TensorShape(16U, 16U), TensorShape(8U, 16U), TensorShape(8U), TensorShape(8U, 8U, 2U), 1.0f, 0.0f);
        add_config(TensorShape(16U, 16U, 5U), TensorShape(8U, 16U, 5U), TensorShape(8U), TensorShape(8U, 8U, 2U, 5U), 1.0f, 0.0f);
    }
};

class SmallGEMMInputOutput3DDataset final : public GEMMDataset
{
public:
    SmallGEMMInputOutput3DDataset()
    {
        add_config(TensorShape(21U, 14U, 13U), TensorShape(34U, 21U), TensorShape(34U), TensorShape(34U, 14U, 13U), 1.0f, 0.0f);
        add_config(TensorShape(31U, 1U, 3U), TensorShape(23U, 31U), TensorShape(23U), TensorShape(23U, 1U, 3U), 1.0f, 0.0f);
        add_config(TensorShape(38U, 12U, 2U), TensorShape(21U, 38U), TensorShape(21U), TensorShape(21U, 12U, 2U), 0.2f, 1.2f);
        add_config(TensorShape(32U, 1U, 4U, 3U), TensorShape(17U, 32U), TensorShape(17U), TensorShape(17U, 1U, 4U, 3U), 0.4f, 0.7f);
        add_config(TensorShape(16U, 16U, 3U, 2U), TensorShape(8U, 16U), TensorShape(8U), TensorShape(8U, 16U, 3U, 2U), 1.0f, 0.0f);
        add_config(TensorShape(16U, 16U, 5U, 3U), TensorShape(8U, 16U), TensorShape(8U), TensorShape(8U, 16U, 5U, 3U), 1.0f, 0.3f);
    }
};
} // namespace datasets
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_SMALL_GEMM_DATASET */
