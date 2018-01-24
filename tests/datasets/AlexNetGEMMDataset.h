/*
 * Copyright (c) 2017, 2018 ARM Limited.
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
#ifndef ARM_COMPUTE_TEST_ALEXNET_GEMM_DATASET
#define ARM_COMPUTE_TEST_ALEXNET_GEMM_DATASET

#include "tests/datasets/GEMMDataset.h"

#include "arm_compute/core/TensorShape.h"

namespace arm_compute
{
namespace test
{
namespace datasets
{
class AlexNetGEMMDataset final : public GEMMDataset
{
public:
    AlexNetGEMMDataset()
    {
        add_config(TensorShape{ 364U, 3025U }, TensorShape{ 96U, 364U }, TensorShape{ 96U, 3025U }, TensorShape{ 96U, 3025U }, 1.f, 0.f);
        add_config(TensorShape{ 1201U, 729U }, TensorShape{ 128U, 1201U }, TensorShape{ 128U, 729U }, TensorShape{ 128U, 729U }, 1.f, 0.f);
        add_config(TensorShape{ 2305U, 169U }, TensorShape{ 384U, 2305U }, TensorShape{ 384U, 169U }, TensorShape{ 384U, 169U }, 1.f, 0.f);
        add_config(TensorShape{ 1729U, 169U }, TensorShape{ 192U, 1729U }, TensorShape{ 192U, 169U }, TensorShape{ 192U, 169U }, 1.f, 0.f);
        add_config(TensorShape{ 1729U, 169U }, TensorShape{ 128U, 1729U }, TensorShape{ 128U, 169U }, TensorShape{ 128U, 169U }, 1.f, 0.f);
        add_config(TensorShape{ 9216U, 1U }, TensorShape{ 4096U, 9216U }, TensorShape{ 4096U, 1U }, TensorShape{ 4096U, 1U }, 1.f, 0.f);
        add_config(TensorShape{ 4096U, 1U }, TensorShape{ 4096U, 4096U }, TensorShape{ 4096U, 1U }, TensorShape{ 4096U, 1U }, 1.f, 0.f);
        add_config(TensorShape{ 4096U, 1U }, TensorShape{ 1000U, 4096U }, TensorShape{ 1000U, 1U }, TensorShape{ 1000U, 1U }, 1.f, 0.f);
    }
};
} // namespace datasets
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_ALEXNET_GEMM_DATASET */
