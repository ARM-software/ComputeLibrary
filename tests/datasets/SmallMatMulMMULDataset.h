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

#ifndef ACL_TESTS_DATASETS_SMALLMATMULMMULDATASET
#define ACL_TESTS_DATASETS_SMALLMATMULMMULDATASET

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
class SmallMatMulMMULDataset final : public MatMulDataset
{
public:
    SmallMatMulMMULDataset()
    {
        add_config(TensorShape(8U, 4U, 2U, 2U), TensorShape(2U, 8U, 2U, 2U), TensorShape(2U, 4U, 2U, 2U));
        add_config(TensorShape(28U, 1U), TensorShape(23U, 28U), TensorShape(23U, 1U));
        add_config(TensorShape(8U, 4U, 2U), TensorShape(16U, 8U, 2U), TensorShape(16U, 4U, 2U));
        add_config(TensorShape(32U, 2U), TensorShape(17U, 32U), TensorShape(17U, 2U));
        add_config(TensorShape(8U, 6U), TensorShape(7U, 8U), TensorShape(7U, 6U));
    }
};

class TinyMatMulMMULDataset final : public MatMulDataset
{
public:
    TinyMatMulMMULDataset()
    {
        add_config(TensorShape(4U, 4U), TensorShape(4U, 4U), TensorShape(4U, 4U));
    }
};

} // namespace datasets
} // namespace test
} // namespace arm_compute

#endif /* ACL_TESTS_DATASETS_SMALLMATMULMMULDATASET */
