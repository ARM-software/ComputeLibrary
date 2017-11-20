/*
 * Copyright (c) 2016, 2017 ARM Limited.
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
#ifndef __ARM_COMPUTE_OPENCL_H__
#define __ARM_COMPUTE_OPENCL_H__

#include <string>
#include <utility>

/* Configure the Khronos C++ wrapper to target OpenCL 1.2: */
#ifndef ARM_NO_EXCEPTIONS
#define CL_HPP_ENABLE_EXCEPTIONS
#endif // ARM_NO_EXCEPTIONS
#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_TARGET_OPENCL_VERSION 110
#define CL_HPP_MINIMUM_OPENCL_VERSION 110
#include <CL/cl2.hpp>

namespace cl
{
static const NDRange Range_128_1 = NDRange(128, 1);
} // namespace cl

namespace arm_compute
{
bool opencl_is_available();

class CLSymbols final
{
private:
    CLSymbols() = default;
    void load_symbols(void *handle);

public:
    static CLSymbols &get();
    bool load(const std::string &library);
    bool load_default();

    using clBuildProgram_func            = cl_int (*)(cl_program, cl_uint, const cl_device_id *, const char *, void (*pfn_notify)(cl_program, void *), void *);
    using clEnqueueNDRangeKernel_func    = cl_int (*)(cl_command_queue, cl_kernel, cl_uint, const size_t *, const size_t *, const size_t *, cl_uint, const cl_event *, cl_event *);
    using clSetKernelArg_func            = cl_int (*)(cl_kernel, cl_uint, size_t, const void *);
    using clRetainMemObject_func         = cl_int (*)(cl_mem);
    using clReleaseMemObject_func        = cl_int (*)(cl_mem);
    using clEnqueueUnmapMemObject_func   = cl_int (*)(cl_command_queue, cl_mem, void *, cl_uint, const cl_event *, cl_event *);
    using clRetainCommandQueue_func      = cl_int (*)(cl_command_queue command_queue);
    using clReleaseContext_func          = cl_int (*)(cl_context);
    using clReleaseEvent_func            = cl_int (*)(cl_event);
    using clEnqueueWriteBuffer_func      = cl_int (*)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, const void *, cl_uint, const cl_event *, cl_event *);
    using clEnqueueReadBuffer_func       = cl_int (*)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, void *, cl_uint, const cl_event *, cl_event *);
    using clGetProgramBuildInfo_func     = cl_int (*)(cl_program, cl_device_id, cl_program_build_info, size_t, void *, size_t *);
    using clRetainProgram_func           = cl_int (*)(cl_program program);
    using clEnqueueMapBuffer_func        = void *(*)(cl_command_queue, cl_mem, cl_bool, cl_map_flags, size_t, size_t, cl_uint, const cl_event *, cl_event *, cl_int *);
    using clReleaseCommandQueue_func     = cl_int (*)(cl_command_queue);
    using clCreateProgramWithBinary_func = cl_program (*)(cl_context, cl_uint, const cl_device_id *, const size_t *, const unsigned char **, cl_int *, cl_int *);
    using clRetainContext_func           = cl_int (*)(cl_context context);
    using clReleaseProgram_func          = cl_int (*)(cl_program program);
    using clFlush_func                   = cl_int (*)(cl_command_queue command_queue);
    using clFinish_func                  = cl_int (*)(cl_command_queue command_queue);
    using clGetProgramInfo_func          = cl_int (*)(cl_program, cl_program_info, size_t, void *, size_t *);
    using clCreateKernel_func            = cl_kernel (*)(cl_program, const char *, cl_int *);
    using clRetainKernel_func            = cl_int (*)(cl_kernel kernel);
    using clCreateBuffer_func            = cl_mem (*)(cl_context, cl_mem_flags, size_t, void *, cl_int *);
    using clCreateProgramWithSource_func = cl_program (*)(cl_context, cl_uint, const char **, const size_t *, cl_int *);
    using clReleaseKernel_func           = cl_int (*)(cl_kernel kernel);
    using clGetDeviceInfo_func           = cl_int (*)(cl_device_id, cl_device_info, size_t, void *, size_t *);
    using clGetDeviceIDs_func            = cl_int (*)(cl_platform_id, cl_device_type, cl_uint, cl_device_id *, cl_uint *);
    using clRetainEvent_func             = cl_int (*)(cl_event);
    using clGetPlatformIDs_func          = cl_int (*)(cl_uint, cl_platform_id *, cl_uint *);
    using clGetKernelWorkGroupInfo_func  = cl_int (*)(cl_kernel, cl_device_id, cl_kernel_work_group_info, size_t, void *, size_t *);

    clBuildProgram_func            clBuildProgram            = nullptr;
    clEnqueueNDRangeKernel_func    clEnqueueNDRangeKernel    = nullptr;
    clSetKernelArg_func            clSetKernelArg            = nullptr;
    clReleaseKernel_func           clReleaseKernel           = nullptr;
    clCreateProgramWithSource_func clCreateProgramWithSource = nullptr;
    clCreateBuffer_func            clCreateBuffer            = nullptr;
    clRetainKernel_func            clRetainKernel            = nullptr;
    clCreateKernel_func            clCreateKernel            = nullptr;
    clGetProgramInfo_func          clGetProgramInfo          = nullptr;
    clFlush_func                   clFlush                   = nullptr;
    clFinish_func                  clFinish                  = nullptr;
    clReleaseProgram_func          clReleaseProgram          = nullptr;
    clRetainContext_func           clRetainContext           = nullptr;
    clCreateProgramWithBinary_func clCreateProgramWithBinary = nullptr;
    clReleaseCommandQueue_func     clReleaseCommandQueue     = nullptr;
    clEnqueueMapBuffer_func        clEnqueueMapBuffer        = nullptr;
    clRetainProgram_func           clRetainProgram           = nullptr;
    clGetProgramBuildInfo_func     clGetProgramBuildInfo     = nullptr;
    clEnqueueReadBuffer_func       clEnqueueReadBuffer       = nullptr;
    clEnqueueWriteBuffer_func      clEnqueueWriteBuffer      = nullptr;
    clReleaseEvent_func            clReleaseEvent            = nullptr;
    clReleaseContext_func          clReleaseContext          = nullptr;
    clRetainCommandQueue_func      clRetainCommandQueue      = nullptr;
    clEnqueueUnmapMemObject_func   clEnqueueUnmapMemObject   = nullptr;
    clRetainMemObject_func         clRetainMemObject         = nullptr;
    clReleaseMemObject_func        clReleaseMemObject        = nullptr;
    clGetDeviceInfo_func           clGetDeviceInfo           = nullptr;
    clGetDeviceIDs_func            clGetDeviceIDs            = nullptr;
    clRetainEvent_func             clRetainEvent             = nullptr;
    clGetPlatformIDs_func          clGetPlatformIDs          = nullptr;
    clGetKernelWorkGroupInfo_func  clGetKernelWorkGroupInfo  = nullptr;

private:
    std::pair<bool, bool> _loaded{ false, false };
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_OPENCL_H__ */
