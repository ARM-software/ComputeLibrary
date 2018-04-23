/*
 * Copyright (c) 2018 ARM Limited.
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
#include "arm_compute/runtime/CL/CLTensorAllocator.h"

#include "arm_compute/runtime/CL/CLMemoryGroup.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"

#include <memory>

namespace arm_compute
{
namespace test
{
namespace validation
{
TEST_SUITE(CL)
TEST_SUITE(UNIT)
TEST_SUITE(TensorAllocator)

TEST_CASE(ImportMemory, framework::DatasetMode::ALL)
{
    // Init tensor info
    TensorInfo info(TensorShape(24U, 16U, 3U), 1, DataType::F32);

    // Allocate memory
    auto buf = std::make_shared<CLBufferMemoryRegion>(CLScheduler::get().context(), CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE, info.total_size());

    // Negative case : Import empty memory
    CLTensor t1;
    t1.allocator()->init(info);
    ARM_COMPUTE_EXPECT(!bool(t1.allocator()->import_memory(CLMemory())), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(t1.info()->is_resizable(), framework::LogLevel::ERRORS);

    // Negative case : Import memory to a tensor that is memory managed
    CLTensor      t2;
    CLMemoryGroup mg;
    t2.allocator()->set_associated_memory_group(&mg);
    ARM_COMPUTE_EXPECT(!bool(t2.allocator()->import_memory(CLMemory(buf))), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(t2.info()->is_resizable(), framework::LogLevel::ERRORS);

    // Positive case : Set managed pointer
    CLTensor t3;
    t3.allocator()->init(info);
    ARM_COMPUTE_EXPECT(bool(t3.allocator()->import_memory(CLMemory(buf))), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(!t3.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(t3.cl_buffer().get() == buf->cl_data().get(), framework::LogLevel::ERRORS);
    t3.allocator()->free();
    ARM_COMPUTE_EXPECT(t3.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(t3.buffer() == nullptr, framework::LogLevel::ERRORS);
}

TEST_SUITE_END()
TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
