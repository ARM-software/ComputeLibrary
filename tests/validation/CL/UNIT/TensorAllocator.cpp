/*
 * Copyright (c) 2018-2021 Arm Limited.
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

#include "arm_compute/core/utils/misc/MMappedFile.h"
#include "arm_compute/runtime/BlobLifetimeManager.h"
#include "arm_compute/runtime/CL/CLBufferAllocator.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "arm_compute/runtime/CL/functions/CLActivationLayer.h"
#include "arm_compute/runtime/CL/functions/CLGEMMConvolutionLayer.h"
#include "arm_compute/runtime/MemoryGroup.h"
#include "arm_compute/runtime/MemoryManagerOnDemand.h"
#include "arm_compute/runtime/PoolManager.h"
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

class DummyAllocator final : public IAllocator
{
public:
    DummyAllocator() = default;

    void *allocate(size_t size, size_t alignment) override
    {
        ++_n_calls;
        return _backend_allocator.allocate(size, alignment);
    }
    void free(void *ptr) override
    {
        return _backend_allocator.free(ptr);
    }
    std::unique_ptr<IMemoryRegion> make_region(size_t size, size_t alignment) override
    {
        // Needs to be implemented as is the one that is used internally by the CLTensorAllocator
        ++_n_calls;
        return std::move(_backend_allocator.make_region(size, alignment));
    }
    int get_n_calls() const
    {
        return _n_calls;
    }

private:
    int               _n_calls{};
    CLBufferAllocator _backend_allocator{};
};

void run_conv2d(std::shared_ptr<IMemoryManager> mm, IAllocator &mm_allocator)
{
    // Create tensors
    CLTensor src, weights, bias, dst;
    src.allocator()->init(TensorInfo(TensorShape(16U, 32U, 32U, 2U), 1, DataType::F32, DataLayout::NHWC));
    weights.allocator()->init(TensorInfo(TensorShape(16U, 3U, 3U, 32U), 1, DataType::F32, DataLayout::NHWC));
    bias.allocator()->init(TensorInfo(TensorShape(32U), 1, DataType::F32, DataLayout::NHWC));
    dst.allocator()->init(TensorInfo(TensorShape(32U, 32U, 32U, 2U), 1, DataType::F32, DataLayout::NHWC));

    // Create and configure function
    CLGEMMConvolutionLayer conv(mm);
    conv.configure(&src, &weights, &bias, &dst, PadStrideInfo(1U, 1U, 1U, 1U));

    // Allocate tensors
    src.allocator()->allocate();
    weights.allocator()->allocate();
    bias.allocator()->allocate();
    dst.allocator()->allocate();

    // Finalize memory manager
    if(mm != nullptr)
    {
        mm->populate(mm_allocator, 1 /* num_pools */);
        ARM_COMPUTE_EXPECT(mm->lifetime_manager()->are_all_finalized(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(mm->pool_manager()->num_pools() == 1, framework::LogLevel::ERRORS);
    }

    conv.run();
}
} // namespace

TEST_SUITE(CL)
TEST_SUITE(UNIT)
TEST_SUITE(TensorAllocator)

/* Validate that an external global allocator can be used for all internal allocations */
TEST_CASE(ExternalGlobalAllocator, framework::DatasetMode::ALL)
{
    DummyAllocator global_tensor_alloc;
    CLTensorAllocator::set_global_allocator(&global_tensor_alloc);

    // Run a convolution
    run_conv2d(nullptr /* mm */, global_tensor_alloc);

    // Check that allocator has been called multiple times > 4
    ARM_COMPUTE_EXPECT(global_tensor_alloc.get_n_calls() > 4, framework::LogLevel::ERRORS);

    // Nullify global allocator
    CLTensorAllocator::set_global_allocator(nullptr);
}

/* Validate that an external global allocator can be used for the pool manager */
TEST_CASE(ExternalGlobalAllocatorMemoryPool, framework::DatasetMode::ALL)
{
    auto lifetime_mgr = std::make_shared<BlobLifetimeManager>();
    auto pool_mgr     = std::make_shared<PoolManager>();
    auto mm           = std::make_shared<MemoryManagerOnDemand>(lifetime_mgr, pool_mgr);

    DummyAllocator global_tensor_alloc;
    CLTensorAllocator::set_global_allocator(&global_tensor_alloc);

    // Run a convolution
    run_conv2d(mm, global_tensor_alloc);

    // Check that allocator has been called multiple times > 4
    ARM_COMPUTE_EXPECT(global_tensor_alloc.get_n_calls() > 4, framework::LogLevel::ERRORS);

    // Nullify global allocator
    CLTensorAllocator::set_global_allocator(nullptr);
}

/** Validates import memory interface when importing cl buffer objects */
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
    CLTensor    t2;
    MemoryGroup mg;
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

/** Validates import memory interface when importing malloced memory */
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
        auto         raw_data            = std::make_unique<uint8_t[]>(space);

        void *aligned_ptr = raw_data.get();
        std::align(alignment, total_size_in_bytes, aligned_ptr, space);

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

#if !defined(BARE_METAL)
/** Validates import memory interface when importing memory mapped objects */
TEST_CASE(ImportMemoryMappedFile, framework::DatasetMode::ALL)
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

        cl::Buffer wrapped_buffer(import_malloc_memory_helper(data, total_size_in_bytes));
        ARM_COMPUTE_EXPECT(bool(tensor.allocator()->import_memory(wrapped_buffer)), framework::LogLevel::ERRORS);
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
#endif // !defined(BARE_METAL)

/** Validates symmetric per channel quantization */
TEST_CASE(Symm8PerChannelQuantizationInfo, framework::DatasetMode::ALL)
{
    // Create tensor
    CLTensor                 tensor;
    const std::vector<float> scale = { 0.25f, 1.4f, 3.2f, 2.3f, 4.7f };
    const TensorInfo         info(TensorShape(32U, 16U), 1, DataType::QSYMM8_PER_CHANNEL, QuantizationInfo(scale));
    tensor.allocator()->init(info);

    // Check quantization information
    ARM_COMPUTE_EXPECT(!tensor.info()->quantization_info().empty(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(!tensor.info()->quantization_info().scale().empty(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(tensor.info()->quantization_info().scale().size() == scale.size(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(tensor.info()->quantization_info().offset().empty(), framework::LogLevel::ERRORS);

    CLQuantization quantization = tensor.quantization();
    ARM_COMPUTE_ASSERT(quantization.scale != nullptr);
    ARM_COMPUTE_ASSERT(quantization.offset != nullptr);

    // Check OpenCL quantization arrays before allocating
    ARM_COMPUTE_EXPECT(quantization.scale->max_num_values() == 0, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(quantization.offset->max_num_values() == 0, framework::LogLevel::ERRORS);

    // Check OpenCL quantization arrays after allocating
    tensor.allocator()->allocate();
    ARM_COMPUTE_EXPECT(quantization.scale->max_num_values() == scale.size(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(quantization.offset->max_num_values() == 0, framework::LogLevel::ERRORS);

    // Validate that the scale values are the same
    auto  cl_scale_buffer = quantization.scale->cl_buffer();
    void *mapped_ptr      = CLScheduler::get().queue().enqueueMapBuffer(cl_scale_buffer, CL_TRUE, CL_MAP_READ, 0, scale.size());
    auto  cl_scale_ptr    = static_cast<float *>(mapped_ptr);
    for(unsigned int i = 0; i < scale.size(); ++i)
    {
        ARM_COMPUTE_EXPECT(cl_scale_ptr[i] == scale[i], framework::LogLevel::ERRORS);
    }
    CLScheduler::get().queue().enqueueUnmapMemObject(cl_scale_buffer, mapped_ptr);
}

TEST_SUITE_END() // TensorAllocator
TEST_SUITE_END() // UNIT
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute
