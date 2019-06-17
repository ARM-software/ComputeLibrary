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
#include "arm_compute/runtime/TensorAllocator.h"

#include "arm_compute/core/utils/misc/MMappedFile.h"
#include "arm_compute/core/utils/misc/Utility.h"
#include "arm_compute/runtime/MemoryGroup.h"
#include "arm_compute/runtime/MemoryRegion.h"
#include "arm_compute/runtime/NEON/functions/NEActivationLayer.h"

#include "support/ToolchainSupport.h"

#include "tests/Globals.h"
#include "tests/Utils.h"
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
TEST_SUITE(NEON)
TEST_SUITE(UNIT)
TEST_SUITE(TensorAllocator)

TEST_CASE(ImportMemory, framework::DatasetMode::ALL)
{
    // Init tensor info
    TensorInfo info(TensorShape(24U, 16U, 3U), 1, DataType::F32);

    // Allocate memory buffer
    const size_t total_size = info.total_size();
    auto         data       = support::cpp14::make_unique<uint8_t[]>(total_size);

    // Negative case : Import nullptr
    Tensor t1;
    t1.allocator()->init(info);
    ARM_COMPUTE_EXPECT(!bool(t1.allocator()->import_memory(nullptr)), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(t1.info()->is_resizable(), framework::LogLevel::ERRORS);

    // Negative case : Import misaligned pointer
    Tensor       t2;
    const size_t required_alignment = 339;
    t2.allocator()->init(info, required_alignment);
    ARM_COMPUTE_EXPECT(!bool(t2.allocator()->import_memory(data.get())), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(t2.info()->is_resizable(), framework::LogLevel::ERRORS);

    // Negative case : Import memory to a tensor that is memory managed
    Tensor      t3;
    MemoryGroup mg;
    t3.allocator()->set_associated_memory_group(&mg);
    ARM_COMPUTE_EXPECT(!bool(t3.allocator()->import_memory(data.get())), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(t3.info()->is_resizable(), framework::LogLevel::ERRORS);

    // Positive case : Set raw pointer
    Tensor t4;
    t4.allocator()->init(info);
    ARM_COMPUTE_EXPECT(bool(t4.allocator()->import_memory(data.get())), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(!t4.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(t4.buffer() == reinterpret_cast<uint8_t *>(data.get()), framework::LogLevel::ERRORS);
    t4.allocator()->free();
    ARM_COMPUTE_EXPECT(t4.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(t4.buffer() == nullptr, framework::LogLevel::ERRORS);
}

TEST_CASE(ImportMemoryMalloc, framework::DatasetMode::ALL)
{
    const ActivationLayerInfo act_info(ActivationLayerInfo::ActivationFunction::RELU);
    const TensorShape         shape     = TensorShape(24U, 16U, 3U);
    const DataType            data_type = DataType::F32;

    // Create tensor
    const TensorInfo info(shape, 1, data_type);
    const size_t     required_alignment = 64;
    Tensor           tensor;
    tensor.allocator()->init(info, required_alignment);

    // Create and configure activation function
    NEActivationLayer act_func;
    act_func.configure(&tensor, nullptr, act_info);

    // Allocate and import tensor
    const size_t total_size_in_elems = tensor.info()->tensor_shape().total_size();
    const size_t total_size_in_bytes = tensor.info()->total_size();
    size_t       space               = total_size_in_bytes + required_alignment;
    auto         raw_data            = support::cpp14::make_unique<uint8_t[]>(space);

    void *aligned_ptr = raw_data.get();
    support::cpp11::align(required_alignment, total_size_in_bytes, aligned_ptr, space);

    ARM_COMPUTE_EXPECT(bool(tensor.allocator()->import_memory(aligned_ptr)), framework::LogLevel::ERRORS);
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

    // Validate result by checking that the input has no negative values
    for(unsigned int i = 0; i < total_size_in_elems; ++i)
    {
        ARM_COMPUTE_EXPECT(typed_ptr[i] >= 0, framework::LogLevel::ERRORS);
    }

    // Release resources
    tensor.allocator()->free();
    ARM_COMPUTE_EXPECT(tensor.info()->is_resizable(), framework::LogLevel::ERRORS);
}

TEST_CASE(ImportMemoryMallocPadded, framework::DatasetMode::ALL)
{
    // Create tensor
    Tensor tensor;
    tensor.allocator()->init(TensorInfo(TensorShape(24U, 16U, 3U), 1, DataType::F32));

    // Enforce tensor padding and validate that meta-data were updated
    // Note: Padding might be updated after the function configuration in case of increased padding requirements
    const PaddingSize enforced_padding(3U, 5U, 2U, 4U);
    tensor.info()->extend_padding(enforced_padding);
    validate(tensor.info()->padding(), enforced_padding);

    // Create and configure activation function
    NEActivationLayer act_func;
    act_func.configure(&tensor, nullptr, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

    // Allocate and import tensor
    const size_t total_size_in_bytes = tensor.info()->total_size();
    auto         raw_data            = support::cpp14::make_unique<uint8_t[]>(total_size_in_bytes);

    ARM_COMPUTE_EXPECT(bool(tensor.allocator()->import_memory(raw_data.get())), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(!tensor.info()->is_resizable(), framework::LogLevel::ERRORS);

    // Fill tensor while accounting padding
    std::uniform_real_distribution<float> distribution(-5.f, 5.f);
    std::mt19937                          gen(library->seed());

    Window tensor_window;
    tensor_window.use_tensor_dimensions(tensor.info()->tensor_shape());
    Iterator tensor_it(&tensor, tensor_window);

    execute_window_loop(tensor_window, [&](const Coordinates &)
    {
        *reinterpret_cast<float *>(tensor_it.ptr()) = distribution(gen);
    },
    tensor_it);

    // Execute function and sync
    act_func.run();

    // Validate result by checking that the input has no negative values
    execute_window_loop(tensor_window, [&](const Coordinates &)
    {
        const float val = *reinterpret_cast<float *>(tensor_it.ptr());
        ARM_COMPUTE_EXPECT(val >= 0, framework::LogLevel::ERRORS);
    },
    tensor_it);

    // Release resources
    tensor.allocator()->free();
    ARM_COMPUTE_EXPECT(tensor.info()->is_resizable(), framework::LogLevel::ERRORS);
}

#if !defined(BARE_METAL)
TEST_CASE(ImportMemoryMappedFile, framework::DatasetMode::ALL)
{
    const ActivationLayerInfo act_info(ActivationLayerInfo::ActivationFunction::RELU);
    const TensorShape         shape     = TensorShape(24U, 16U, 3U);
    const DataType            data_type = DataType::F32;

    // Create tensor
    const TensorInfo info(shape, 1, data_type);
    Tensor           tensor;
    tensor.allocator()->init(info);

    // Create and configure activation function
    NEActivationLayer act_func;
    act_func.configure(&tensor, nullptr, act_info);

    // Get number of elements
    const size_t total_size_in_elems = tensor.info()->tensor_shape().total_size();
    const size_t total_size_in_bytes = tensor.info()->total_size();

    // Create file
    std::ofstream output_file("test_mmap_import.bin", std::ios::binary | std::ios::out);
    output_file.seekp(total_size_in_bytes - 1);
    output_file.write("", 1);
    output_file.close();

    // Map file
    utils::mmap_io::MMappedFile mmapped_file("test_mmap_import.bin", 0 /** Whole file */, 0);
    ARM_COMPUTE_EXPECT(mmapped_file.is_mapped(), framework::LogLevel::ERRORS);
    unsigned char *data = mmapped_file.data();

    // Import memory mapped memory
    ARM_COMPUTE_EXPECT(bool(tensor.allocator()->import_memory(data)), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(!tensor.info()->is_resizable(), framework::LogLevel::ERRORS);

    // Fill tensor
    std::uniform_real_distribution<float> distribution(-5.f, 5.f);
    std::mt19937                          gen(library->seed());
    auto                                 *typed_ptr = reinterpret_cast<float *>(data);
    for(unsigned int i = 0; i < total_size_in_elems; ++i)
    {
        typed_ptr[i] = distribution(gen);
    }

    // Execute function and sync
    act_func.run();

    // Validate result by checking that the input has no negative values
    for(unsigned int i = 0; i < total_size_in_elems; ++i)
    {
        ARM_COMPUTE_EXPECT(typed_ptr[i] >= 0, framework::LogLevel::ERRORS);
    }

    // Release resources
    tensor.allocator()->free();
    ARM_COMPUTE_EXPECT(tensor.info()->is_resizable(), framework::LogLevel::ERRORS);
}
#endif // !defined(BARE_METAL)

TEST_CASE(AlignedAlloc, framework::DatasetMode::ALL)
{
    // Init tensor info
    TensorInfo   info(TensorShape(24U, 16U, 3U), 1, DataType::F32);
    const size_t requested_alignment = 1024;

    Tensor t;
    t.allocator()->init(info, requested_alignment);
    t.allocator()->allocate();

    ARM_COMPUTE_EXPECT(t.buffer() != nullptr, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(t.allocator()->alignment() == requested_alignment, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(arm_compute::utility::check_aligned(reinterpret_cast<void *>(t.buffer()), requested_alignment),
                       framework::LogLevel::ERRORS);
}

TEST_SUITE_END()
TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
