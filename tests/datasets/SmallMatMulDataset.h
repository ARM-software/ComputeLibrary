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
#ifndef ACL_TESTS_DATASETS_SMALLMATMULDATASET
#define ACL_TESTS_DATASETS_SMALLMATMULDATASET

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "tests/datasets/MatMulDataset.h"

namespace arm_compute
{
namespace test
{
namespace datasets
{
class SmallMatMulDataset final : public MatMulDataset
{
public:
    SmallMatMulDataset()
    {
        add_config(TensorShape(3U, 4U, 2U, 2U), TensorShape(2U, 3U, 2U, 2U), TensorShape(2U, 4U, 2U, 2U));
        add_config(TensorShape(9U, 6U), TensorShape(5U, 9U), TensorShape(5U, 6U));
        add_config(TensorShape(31U, 1U), TensorShape(23U, 31U), TensorShape(23U, 1U));
        add_config(TensorShape(8U, 4U, 2U), TensorShape(16U, 8U, 2U), TensorShape(16U, 4U, 2U));
        add_config(TensorShape(32U, 2U), TensorShape(17U, 32U), TensorShape(17U, 2U));
    }
};

class TinyMatMulDataset final : public MatMulDataset
{
public:
    TinyMatMulDataset()
    {
        add_config(TensorShape(1U), TensorShape(1U), TensorShape(1U));
        add_config(TensorShape(2U, 2U), TensorShape(2U, 2U), TensorShape(2U, 2U));
    }
};

class SmallMatMulDatasetRhsExportToCLImageRhsT final : public MatMulDataset
{
public:
    // Some considerations:
    //  1) K dimension should be a multiple of 4
    //  See (2), (3), and (4) in SmallMatMulDatasetRhsExportToCLImageRhsNT
    SmallMatMulDatasetRhsExportToCLImageRhsT()
    {
        add_config(TensorShape(8U /*K*/, 3U /*M*/, 2U, 1U, 2U), TensorShape(20U /*N*/, 8U /*K*/, 2U, 1U, 2U), TensorShape(20U /*N*/, 3U /*M*/, 2U, 1U, 2U));
    }
};

class SmallMatMulDatasetRhsExportToCLImageRhsNT final : public MatMulDataset
{
public:
    // Some considerations:
    //  (1) N (Dimension 0 of Rhs matrix) dimension should be a multiple of 4
    //  (2) Having N=20 enables us to test all possible N0 values, i.e. 4, 8, 16
    //  (3) It's important to have more than one loop iterations in the K dimension
    //      K has been chosen in accordance with K0
    //  (4) The 5-th dimension has been chosen as non-unit because export_to_cl_iamge checks
    //      were using dim1 * dim2 * dim3 to calculate the CLImage height; however, in our case
    //      the tensor can be > 4D. To stress that case, the fifth dimension is chosen to be non-unit as well
    SmallMatMulDatasetRhsExportToCLImageRhsNT()
    {
        add_config(TensorShape(7U, 3U, 2U, 1U, 2U), TensorShape(20U, 7U, 2U, 1U, 2U), TensorShape(20U, 3U, 2U, 1U, 2U));
    }
};
} // namespace datasets
} // namespace test
} // namespace arm_compute
#endif /* ACL_TESTS_DATASETS_SMALLMATMULDATASET */
