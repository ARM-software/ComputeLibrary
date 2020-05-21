/*
 * Copyright (c) 2020 ARM Limited.
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
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/NEON/functions/NEReshapeLayer.h"
#include "arm_compute/runtime/OperatorTensor.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/NEON/Accessor.h"
#include "tests/Utils.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
TEST_SUITE(NEON)
TEST_SUITE(ReshapeOperator)

TEST_CASE(Run, framework::DatasetMode::ALL)
{
    // Create tensors and info
    TensorInfo src_info(TensorShape(27U, 11U, 3U), 1, DataType::F32);
    TensorInfo dst_info(TensorShape(27U, 11U, 3U), 1, DataType::F32);
    Tensor     src = create_tensor<Tensor>(TensorShape(27U, 11U, 3U), DataType::F32, 1);
    Tensor     dst = create_tensor<Tensor>(TensorShape(27U, 11U, 3U), DataType::F32, 1);

    // Create and configure function
    experimental::NEReshapeLayer reshape_operator;
    reshape_operator.configure(&src_info, &dst_info);

    // Allocate tensors
    src.allocator()->allocate();
    dst.allocator()->allocate();

    InputOperatorTensors  src_0 = std::make_pair(TensorType::ACL_SRC, &src);
    OutputOperatorTensors dst_0 = std::make_pair(TensorType::ACL_DST, &dst);

    std::vector<InputOperatorTensors *>  src_vec  = { &src_0 };
    std::vector<OutputOperatorTensors *> dst_vec  = { &dst_0 };
    std::vector<OperatorTensors *>       work_vec = {};

    // Compute functions
    reshape_operator.run(src_vec, dst_vec, work_vec);
}

TEST_SUITE_END() // ReshapeOperator
TEST_SUITE_END() // NEON
} // namespace validation
} // namespace test
} // namespace arm_compute
