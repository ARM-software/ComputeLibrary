/*
 * Copyright (c) 2026 Arm Limited.
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
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/Tensor.h"

#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/validation/Validation.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
TEST_SUITE(UNIT)
TEST_SUITE(TensorInfo)

TEST_CASE(ImportMemoryDoesNotMutateExternalInfo, framework::DatasetMode::ALL)
{
    // Use F16 if available; otherwise fall back to F32.
    DataType test_dt = DataType::F16;
#if !defined(ARM_COMPUTE_ENABLE_FP16)
    test_dt = DataType::F32;
#endif

    TensorInfo out_info(TensorShape(16U, 4U, 4U), 1, test_dt, DataLayout::NHWC);
    out_info.set_is_resizable(true);

    Tensor out_tensor;
    out_tensor.allocator()->init(out_info);
    out_tensor.allocator()->allocate();

    // Simulate a shared TensorInfo used as a view wrapper.
    TensorInfo shared_info(out_info);
    shared_info.set_is_resizable(true);

    Tensor view_tensor;
    view_tensor.allocator()->soft_init(shared_info);

    // Ensure it's still resizable before import.
    ARM_COMPUTE_EXPECT(shared_info.is_resizable(), framework::LogLevel::ERRORS);

    // Import memory into the view tensor.
    ARM_COMPUTE_ASSERT(bool(view_tensor.allocator()->import_memory(out_tensor.buffer())));

    // Regression assert: import_memory must NOT mutate the caller-owned shared_info.
    ARM_COMPUTE_EXPECT(shared_info.is_resizable(), framework::LogLevel::ERRORS);

    // extend_padding should succeed (not throw).
    ARM_COMPUTE_EXPECT_NO_THROW(shared_info.extend_padding(PaddingSize(1, 1, 1, 1)), framework::LogLevel::ERRORS);
}

TEST_SUITE_END() // TensorInfo
TEST_SUITE_END() // UNIT
} // namespace validation
} // namespace test
} // namespace arm_compute
