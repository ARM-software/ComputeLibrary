/*
 * Copyright (c) 2018-2019 ARM Limited.
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
#include "arm_compute/runtime/CL/functions/CLActivationLayer.h"
#include "tests/CL/CLAccessor.h"
#include "tests/Globals.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/validation/Validation.h"
#include "tests/validation/reference/ActivationLayer.h"

#include <memory>
#include <random>

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
cl_mem import_malloc_memory_helper(void *ptr, size_t size)
{
    const cl_import_properties_arm import_properties[] =
    {
        CL_IMPORT_TYPE_ARM,
        CL_IMPORT_TYPE_HOST_ARM,
        0
    };

    cl_int err = CL_SUCCESS;
    cl_mem buf = clImportMemoryARM(CLKernelLibrary::get().context().get(), CL_MEM_READ_WRITE, import_properties, ptr, size, &err);
    ARM_COMPUTE_ASSERT(err == CL_SUCCESS);

    return buf;
}
} // namespace

TEST_SUITE(CL)
TEST_SUITE(UNIT)
TEST_SUITE(TensorAllocator)

TEST_CASE(ImportMemoryBuffer, framework::DatasetMode::ALL)
{
    // Init tensor info
    const TensorInfo info(TensorShape(24U, 16U, 3U), 1, DataType::F32);

    // Allocate memory buffer
    const size_t total_size = info.total_size();
    auto         buf        = cl::Buffer(CLScheduler::get().context(), CL_MEM_READ_WRITE, total_size);

    // Negative case : Import nullptr
    CLTensor t1;
    t1.allocator()->init(info);
    ARM_COMPUTE_EXPECT(!bool(t1.allocator()->import_memory(cl::Buffer())), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(t1.info()->is_resizable(), framework::LogLevel::ERRORS);

    // Negative case : Import memory to a tensor that is memory managed
    CLTensor      t2;
    CLMemoryGroup mg;
    t2.allocator()->set_associated_memory_group(&mg);
    ARM_COMPUTE_EXPECT(!bool(t2.allocator()->import_memory(buf)), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(t2.info()->is_resizable(), framework::LogLevel::ERRORS);

    // Negative case : Invalid buffer size
    CLTensor         t3;
    const TensorInfo info_neg(TensorShape(32U, 16U, 3U), 1, DataType::F32);
    t3.allocator()->init(info_neg);
    ARM_COMPUTE_EXPECT(!bool(t3.allocator()->import_memory(buf)), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(t3.info()->is_resizable(), framework::LogLevel::ERRORS);

    // Positive case : Set raw pointer
    CLTensor t4;
    t4.allocator()->init(info);
    ARM_COMPUTE_EXPECT(bool(t4.allocator()->import_memory(buf)), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(!t4.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(t4.cl_buffer().get() == buf.get(), framework::LogLevel::ERRORS);
    t4.allocator()->free();
    ARM_COMPUTE_EXPECT(t4.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(t4.cl_buffer().get() != buf.get(), framework::LogLevel::ERRORS);
}

TEST_CASE(ImportMemoryMalloc, framework::DatasetMode::ALL)
{
    // Check if import extension is supported
    if(!device_supports_extension(CLKernelLibrary::get().get_device(), "cl_arm_import_memory_host"))
    {
        return;
    }
    else
    {
        const ActivationLayerInfo act_info(ActivationLayerInfo::ActivationFunction::RELU);
        const TensorShape         shape     = TensorShape(24U, 16U, 3U);
        const DataType            data_type = DataType::F32;

        // Create tensor
        const TensorInfo info(shape, 1, data_type);
        CLTensor         tensor;
        tensor.allocator()->init(info);

        // Create and configure activation function
        CLActivationLayer act_func;
        act_func.configure(&tensor, nullptr, act_info);

        // Allocate and import tensor
        const size_t total_size_in_elems = tensor.info()->tensor_shape().total_size();
        const size_t total_size_in_bytes = tensor.info()->total_size();
        const size_t alignment           = CLKernelLibrary::get().get_device().getInfo<CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE>();
        size_t       space               = total_size_in_bytes + alignment;
        auto         raw_data            = support::cpp14::make_unique<uint8_t[]>(space);

        void *aligned_ptr = raw_data.get();
        support::cpp11::align(alignment, total_size_in_bytes, aligned_ptr, space);

        cl::Buffer wrapped_buffer(import_malloc_memory_helper(aligned_ptr, total_size_in_bytes));
        ARM_COMPUTE_EXPECT(bool(tensor.allocator()->import_memory(wrapped_buffer)), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!tensor.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Fill tensor
        std::uniform_real_distribution<float> distribution(-5.f, 5.f);
        std::mt19937                          gen(library->seed());
        auto                                 *typed_ptr = reinterpret_cast<float *>(aligned_ptr);
        for(unsigned int i = 0; i < total_size_in_elems; ++i)
        {
            typed_ptr[i] = distribution(gen);
        }

        // Execute function and sync
        act_func.run();
        CLScheduler::get().sync();

        // Validate result by checking that the input has no negative values
        for(unsigned int i = 0; i < total_size_in_elems; ++i)
        {
            ARM_COMPUTE_EXPECT(typed_ptr[i] >= 0, framework::LogLevel::ERRORS);
        }

        // Release resources
        tensor.allocator()->free();
        ARM_COMPUTE_EXPECT(tensor.info()->is_resizable(), framework::LogLevel::ERRORS);
    }
}

TEST_SUITE_END() // TensorAllocator
TEST_SUITE_END() // UNIT
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute
