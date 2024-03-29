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

#ifndef ACL_TESTS_DATASETS_MATMULLOWPMMULDATASET_H
#define ACL_TESTS_DATASETS_MATMULLOWPMMULDATASET_H

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "tests/datasets/MatMulDataset.h"

namespace arm_compute
{
namespace test
{
namespace datasets
{
/** MatMulLowp MMUL shapes are similar to MatMul MMUL shapes except that K has to be a
 * multiple of MMUL_K0 which is 16 (e.g. see src/gpu/cl/kernels/ClMatMulLowpNativeMMULKernel.cpp for the definition)
 */
class SmallMatMulLowpMMULDataset final : public MatMulDataset
{
public:
    SmallMatMulLowpMMULDataset()
    {
        add_config(TensorShape(16U, 4U), TensorShape(4U, 16U), TensorShape(4U, 4U)); // same as mmul block
        add_config(TensorShape(96U, 1U), TensorShape(1U, 96U), TensorShape(1U, 1U)); // vector x vector
        add_config(TensorShape(32U, 4U, 2U), TensorShape(16U, 32U, 2U), TensorShape(16U, 4U, 2U));
        add_config(TensorShape(48U, 2U), TensorShape(17U, 48U), TensorShape(17U, 2U));
        add_config(TensorShape(32U, 6U), TensorShape(7U, 32U), TensorShape(7U, 6U));
    }
};

// This dataset is for smaller number of tests that will still use small shapes
// e.g. not repeating everything for QASYMM8 while we're already testing for QASYMM8_SIGNED
class SmallMatMulLowpMMULDatasetSubset final : public MatMulDataset
{
public:
    SmallMatMulLowpMMULDatasetSubset()
    {
        add_config(TensorShape(32U, 4U, 2U), TensorShape(16U, 32U, 2U), TensorShape(16U, 4U, 2U));
        add_config(TensorShape(32U, 6U), TensorShape(7U, 32U), TensorShape(7U, 6U));
    }
};

class SmallMatMulLowpMMULWithBiasDataset final : public MatMulDataset
{
public:
    SmallMatMulLowpMMULWithBiasDataset()
    {
        add_config(TensorShape(32U, 4U, 2U, 2U), TensorShape(16U, 32U, 2U, 2U), TensorShape(16U, 4U, 2U, 2U));
    }
};

class LargeMatMulLowpMMULDataset final : public MatMulDataset
{
public:
    LargeMatMulLowpMMULDataset()
    {
        add_config(TensorShape(192U, 38U, 3U, 2U), TensorShape(21U, 192U, 3U, 2U), TensorShape(21U, 38U, 3U, 2U));
    }
};

class HighDimensionalMatMulLowpMMULDataset final : public MatMulDataset
{
public:
    HighDimensionalMatMulLowpMMULDataset()
    {
        add_config(TensorShape(16U, 5U, 2U, 2U, 2U, 2U), TensorShape(5U, 16U, 2U, 2U, 2U, 2U), TensorShape(5U, 5U, 2U, 2U, 2U, 2U)); // 6D tensor
    }
};

} // namespace datasets
} // namespace test
} // namespace arm_compute

#endif // ACL_TESTS_DATASETS_MATMULLOWPMMULDATASET_H
