/*
 * Copyright (c) 2023 Arm Limited.
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

#ifndef ACL_TESTS_DATASETS_LARGEMATMULMMULDATASET
#define ACL_TESTS_DATASETS_LARGEMATMULMMULDATASET

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "tests/datasets/MatMulDataset.h"

namespace arm_compute
{
namespace test
{
namespace datasets
{
/** MatMul MMUL shapes are similar to MatMul shapes except that K has to be a multiple of MMUL_K0 which is 4 (e.g. see src/gpu/cl/kernels/ClMatMulNativeMMULKernel.cpp for the definition)
 */
class LargeMatMulMMULDataset final : public MatMulDataset
{
public:
    LargeMatMulMMULDataset()
    {
        add_config(TensorShape(24U, 13U, 3U, 2U), TensorShape(33U, 24U, 3U, 2U), TensorShape(33U, 13U, 3U, 2U));
        add_config(TensorShape(36U, 12U, 1U, 5U), TensorShape(21U, 36U, 1U, 5U), TensorShape(21U, 12U, 1U, 5U));
        add_config(TensorShape(44U, 38U, 3U, 2U), TensorShape(21U, 44U, 3U, 2U), TensorShape(21U, 38U, 3U, 2U));
    }
};

class HighDimensionalMatMulMMULDataset final : public MatMulDataset
{
public:
    HighDimensionalMatMulMMULDataset()
    {
        add_config(TensorShape(4U, 5U, 2U, 2U, 2U, 2U), TensorShape(5U, 4U, 2U, 2U, 2U, 2U), TensorShape(5U, 5U, 2U, 2U, 2U, 2U)); // 6D tensor
    }
};

} // namespace datasets
} // namespace test
} // namespace arm_compute

#endif /* ACL_TESTS_DATASETS_LARGEMATMULMMULDATASET */
