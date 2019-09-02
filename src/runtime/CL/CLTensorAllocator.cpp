/*
 * Copyright (c) 2016-2019 ARM Limited.
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

#include "arm_compute/core/Error.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/runtime/CL/CLMemoryGroup.h"
#include "arm_compute/runtime/CL/CLScheduler.h"

namespace arm_compute
{
const cl::Buffer CLTensorAllocator::_empty_buffer = cl::Buffer();

namespace
{
/** Helper function used to allocate the backing memory of a tensor
 *
 * @param[in] context   OpenCL context to use
 * @param[in] size      Size of the allocation
 * @param[in] alignment Alignment of the allocation
 *
 * @return A wrapped memory region
 */
std::unique_ptr<ICLMemoryRegion> allocate_region(const cl::Context &context, size_t size, cl_uint alignment)
{
    // Try fine-grain SVM
    std::unique_ptr<ICLMemoryRegion> region = support::cpp14::make_unique<CLFineSVMMemoryRegion>(context,
                                                                                                 CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER,
                                                                                                 size,
                                                                                                 alignment);

    // Try coarse-grain SVM in case of failure
    if(region != nullptr && region->ptr() == nullptr)
    {
        region = support::cpp14::make_unique<CLCoarseSVMMemoryRegion>(context, CL_MEM_READ_WRITE, size, alignment);
    }
    // Try legacy buffer memory in case of failure
    if(region != nullptr && region->ptr() == nullptr)
    {
        region = support::cpp14::make_unique<CLBufferMemoryRegion>(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE, size);
    }
    return region;
}
/** Clears quantization arrays
 *
 * @param[in, out] scale  Quantization scale array
 * @param[in, out] offset Quantization offset array
 */
void clear_quantization_arrays(CLFloatArray &scale, CLInt32Array &offset)
{
    // Clear arrays
    scale  = CLFloatArray();
    offset = CLInt32Array();
}
/** Helper function used to create quantization data arrays
 *
 * @param[in, out] scale    Quantization scale array
 * @param[in, out] offset   Quantization offset array
 * @param[in]      qinfo    Quantization info
 * @param[in]      pad_size Pad size to use in case array needs to be padded for computation purposes
 *
 * @return A pair (scale, offset) containing the respective allocated and filled arrays
 */
void populate_quantization_info(CLFloatArray &scale, CLInt32Array &offset, const QuantizationInfo &qinfo, size_t pad_size)
{
    clear_quantization_arrays(scale, offset);

    // Create scale array
    const std::vector<float> &qscale       = qinfo.scale();
    const size_t              num_elements = qscale.size();
    const size_t              element_size = sizeof(std::remove_reference<decltype(qscale)>::type::value_type);
    scale                                  = CLFloatArray(num_elements + pad_size);
    scale.resize(num_elements);
    CLScheduler::get().queue().enqueueWriteBuffer(scale.cl_buffer(), CL_TRUE, 0, num_elements * element_size, qinfo.scale().data());
}
} // namespace

CLTensorAllocator::CLTensorAllocator(CLTensor *owner)
    : _associated_memory_group(nullptr), _memory(), _mapping(nullptr), _owner(owner), _scale(), _offset()
{
}

CLQuantization CLTensorAllocator::quantization() const
{
    return { &_scale, &_offset };
}

uint8_t *CLTensorAllocator::data()
{
    return _mapping;
}

const cl::Buffer &CLTensorAllocator::cl_data() const
{
    return _memory.region() == nullptr ? _empty_buffer : _memory.cl_region()->cl_data();
}

void CLTensorAllocator::allocate()
{
    // Allocate tensor backing memory
    if(_associated_memory_group == nullptr)
    {
        if(_memory.region() != nullptr && _memory.cl_region()->cl_data().get() != nullptr)
        {
            // Memory is already allocated. Reuse it if big enough, otherwise fire an assertion
            ARM_COMPUTE_ERROR_ON_MSG(info().total_size() > _memory.region()->size(),
                                     "Reallocation of a bigger memory region is not allowed!");
        }
        else
        {
            // Perform memory allocation
            _memory.set_owned_region(allocate_region(CLScheduler::get().context(), info().total_size(), 0));
        }
    }
    else
    {
        _associated_memory_group->finalize_memory(_owner, _memory, info().total_size());
    }

    // Allocate and fill the quantization parameter arrays
    if(info().data_type() == DataType::QSYMM8_PER_CHANNEL)
    {
        const size_t pad_size = 0;
        populate_quantization_info(_scale, _offset, info().quantization_info(), pad_size);
    }

    // Lock allocator
    info().set_is_resizable(false);
}

void CLTensorAllocator::free()
{
    _mapping = nullptr;
    _memory.set_region(nullptr);
    clear_quantization_arrays(_scale, _offset);
    info().set_is_resizable(true);
}

Status CLTensorAllocator::import_memory(cl::Buffer buffer)
{
    ARM_COMPUTE_RETURN_ERROR_ON(buffer.get() == nullptr);
    ARM_COMPUTE_RETURN_ERROR_ON(buffer.getInfo<CL_MEM_SIZE>() < info().total_size());
    ARM_COMPUTE_RETURN_ERROR_ON(buffer.getInfo<CL_MEM_CONTEXT>().get() != CLScheduler::get().context().get());
    ARM_COMPUTE_RETURN_ERROR_ON(_associated_memory_group != nullptr);

    _memory.set_owned_region(support::cpp14::make_unique<CLBufferMemoryRegion>(buffer));
    info().set_is_resizable(false);

    return Status{};
}

void CLTensorAllocator::set_associated_memory_group(CLMemoryGroup *associated_memory_group)
{
    ARM_COMPUTE_ERROR_ON(associated_memory_group == nullptr);
    ARM_COMPUTE_ERROR_ON(_associated_memory_group != nullptr);
    ARM_COMPUTE_ERROR_ON(_memory.region() != nullptr && _memory.cl_region()->cl_data().get() != nullptr);

    _associated_memory_group = associated_memory_group;
}

uint8_t *CLTensorAllocator::lock()
{
    return map(CLScheduler::get().queue(), true);
}

void CLTensorAllocator::unlock()
{
    ARM_COMPUTE_ERROR_ON(_memory.region() == nullptr);
    unmap(CLScheduler::get().queue(), reinterpret_cast<uint8_t *>(_memory.region()->buffer()));
}

uint8_t *CLTensorAllocator::map(cl::CommandQueue &q, bool blocking)
{
    ARM_COMPUTE_ERROR_ON(_mapping != nullptr);
    ARM_COMPUTE_ERROR_ON(_memory.region() == nullptr);
    ARM_COMPUTE_ERROR_ON(_memory.region()->buffer() != nullptr);

    _mapping = reinterpret_cast<uint8_t *>(_memory.cl_region()->map(q, blocking));
    return _mapping;
}

void CLTensorAllocator::unmap(cl::CommandQueue &q, uint8_t *mapping)
{
    ARM_COMPUTE_ERROR_ON(_mapping == nullptr);
    ARM_COMPUTE_ERROR_ON(_mapping != mapping);
    ARM_COMPUTE_ERROR_ON(_memory.region() == nullptr);
    ARM_COMPUTE_ERROR_ON(_memory.region()->buffer() == nullptr);
    ARM_COMPUTE_UNUSED(mapping);

    _memory.cl_region()->unmap(q);
    _mapping = nullptr;
}
} // namespace arm_compute
