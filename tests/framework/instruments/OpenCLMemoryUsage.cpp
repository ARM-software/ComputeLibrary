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
#include "OpenCLMemoryUsage.h"

#include "../Framework.h"
#include "../Utils.h"

#ifndef ARM_COMPUTE_CL
#error "You can't use OpenCLMemoryUsage without OpenCL"
#endif /* ARM_COMPUTE_CL */

#include "arm_compute/core/CL/CLKernelLibrary.h"

namespace arm_compute
{
namespace test
{
namespace framework
{
std::string OpenCLMemoryUsage::id() const
{
    return "OpenCLMemoryUsage";
}

OpenCLMemoryUsage::OpenCLMemoryUsage(ScaleFactor scale_factor)
    : real_clCreateBuffer(CLSymbols::get().clCreateBuffer_ptr), real_clRetainMemObject(CLSymbols::get().clRetainMemObject_ptr), real_clReleaseMemObject(CLSymbols::get().clReleaseMemObject_ptr),
      real_clSVMAlloc(CLSymbols::get().clSVMAlloc_ptr), real_clSVMFree(CLSymbols::get().clSVMFree_ptr), _allocations(), _svm_allocations(), _start(), _end(), _now()
{
    switch(scale_factor)
    {
        case ScaleFactor::NONE:
            _scale_factor = 1;
            _unit         = "";
            break;
        case ScaleFactor::SCALE_1K:
            _scale_factor = 1000;
            _unit         = "K ";
            break;
        case ScaleFactor::SCALE_1M:
            _scale_factor = 1000000;
            _unit         = "M ";
            break;
        default:
            ARM_COMPUTE_ERROR("Invalid scale");
    }
}

void OpenCLMemoryUsage::test_start()
{
    _now = Stats();

    ARM_COMPUTE_ERROR_ON(CLSymbols::get().clCreateBuffer_ptr == nullptr);
    CLSymbols::get().clCreateBuffer_ptr = [this](
                                              cl_context   context,
                                              cl_mem_flags flags,
                                              size_t       size,
                                              void        *host_ptr,
                                              cl_int *     errcode_ret)
    {
        cl_mem retval = this->real_clCreateBuffer(context, flags, size, host_ptr, errcode_ret);
        if(host_ptr != nullptr)
        {
            // If it's an SVM / external allocation;
            size = 0;
        }
        else
        {
            _now.num_allocations++;
            _now.in_use += size;
            _now.total_allocated += size;
            if(_now.in_use > _now.max_in_use)
            {
                _now.max_in_use = _now.in_use;
            }
        }
        this->_allocations[retval] = Allocation(size);
        return retval;
    };
    ARM_COMPUTE_ERROR_ON(CLSymbols::get().clRetainMemObject_ptr == nullptr);
    CLSymbols::get().clRetainMemObject_ptr = [this](cl_mem memobj)
    {
        cl_int retval = this->real_clRetainMemObject(memobj);
        this->_allocations[memobj].refcount++;
        return retval;
    };
    ARM_COMPUTE_ERROR_ON(CLSymbols::get().clReleaseMemObject_ptr == nullptr);
    CLSymbols::get().clReleaseMemObject_ptr = [this](cl_mem memobj)
    {
        cl_int      retval = this->real_clRetainMemObject(memobj);
        Allocation &alloc  = this->_allocations[memobj];
        if(--alloc.refcount == 0)
        {
            _now.in_use -= alloc.size;
        }
        return retval;
    };

    //Only intercept the function if it exists:
    if(CLSymbols::get().clSVMAlloc_ptr != nullptr)
    {
        CLSymbols::get().clSVMAlloc_ptr = [this](cl_context context, cl_svm_mem_flags flags, size_t size, cl_uint alignment)
        {
            void *retval = this->real_clSVMAlloc(context, flags, size, alignment);
            if(retval != nullptr)
            {
                _svm_allocations[retval] = size;
                _now.num_allocations++;
                _now.in_use += size;
                _now.total_allocated += size;
                if(_now.in_use > _now.max_in_use)
                {
                    _now.max_in_use = _now.in_use;
                }
            }
            return retval;
        };
    }

    //Only intercept the function if it exists:
    if(CLSymbols::get().clSVMFree_ptr != nullptr)
    {
        CLSymbols::get().clSVMFree_ptr = [this](cl_context context, void *svm_pointer)
        {
            this->real_clSVMFree(context, svm_pointer);
            auto iterator = _svm_allocations.find(svm_pointer);
            if(iterator != _svm_allocations.end())
            {
                size_t size = iterator->second;
                _svm_allocations.erase(iterator);
                _now.in_use -= size;
            }
        };
    }
}

void OpenCLMemoryUsage::start()
{
    _start = _now;
}
void OpenCLMemoryUsage::stop()
{
    _end = _now;
}

void OpenCLMemoryUsage::test_stop()
{
    // Restore real function
    CLSymbols::get().clCreateBuffer_ptr     = real_clCreateBuffer;
    CLSymbols::get().clRetainMemObject_ptr  = real_clRetainMemObject;
    CLSymbols::get().clReleaseMemObject_ptr = real_clReleaseMemObject;
    CLSymbols::get().clSVMAlloc_ptr         = real_clSVMAlloc;
    CLSymbols::get().clSVMFree_ptr          = real_clSVMFree;
}

Instrument::MeasurementsMap OpenCLMemoryUsage::measurements() const
{
    MeasurementsMap measurements;
    measurements.emplace("Num buffers allocated per run", Measurement(_end.num_allocations - _start.num_allocations, ""));
    measurements.emplace("Total memory allocated per run", Measurement((_end.total_allocated - _start.total_allocated) / _scale_factor, _unit));
    measurements.emplace("Memory in use at start of run", Measurement(_start.in_use / _scale_factor, _unit));

    return measurements;
}
Instrument::MeasurementsMap OpenCLMemoryUsage::test_measurements() const
{
    MeasurementsMap measurements;
    measurements.emplace("Num buffers", Measurement(_now.num_allocations, ""));
    measurements.emplace("Total memory allocated", Measurement(_now.total_allocated / _scale_factor, _unit));
    measurements.emplace("Max memory allocated", Measurement(_now.max_in_use / _scale_factor, _unit));
    measurements.emplace("Memory leaked", Measurement(_now.in_use / _scale_factor, _unit));

    size_t num_programs = CLKernelLibrary::get().get_built_programs().size();
    size_t total_size   = 0;
    for(auto const &it : CLKernelLibrary::get().get_built_programs())
    {
        std::vector<size_t> binary_sizes = it.second.getInfo<CL_PROGRAM_BINARY_SIZES>();
        total_size                       = std::accumulate(binary_sizes.begin(), binary_sizes.end(), total_size);
    }

    measurements.emplace("Num programs in cache", Measurement(num_programs, ""));
    measurements.emplace("Total programs memory in cache", Measurement(total_size / _scale_factor, _unit));

    return measurements;
}
} // namespace framework
} // namespace test
} // namespace arm_compute
