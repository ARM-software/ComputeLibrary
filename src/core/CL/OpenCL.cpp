/*
 * Copyright (c) 2017 ARM Limited.
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

#include "arm_compute/core/CL/OpenCL.h"

#include <dlfcn.h>
#include <iostream>

using clBuildProgram_func            = cl_int (*)(cl_program, cl_uint, const cl_device_id *, const char *, void (*pfn_notify)(cl_program, void *), void *);
using clEnqueueNDRangeKernel_func    = cl_int (*)(cl_command_queue, cl_kernel, cl_uint, const size_t *, const size_t *, const size_t *, cl_uint, const cl_event *, cl_event *);
using clSetKernelArg_func            = cl_int (*)(cl_kernel, cl_uint, size_t, const void *);
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
using clGetProgramInfo_func          = cl_int (*)(cl_program, cl_program_info, size_t, void *, size_t *);
using clCreateKernel_func            = cl_kernel (*)(cl_program, const char *, cl_int *);
using clRetainKernel_func            = cl_int (*)(cl_kernel kernel);
using clCreateBuffer_func            = cl_mem (*)(cl_context, cl_mem_flags, size_t, void *, cl_int *);
using clCreateProgramWithSource_func = cl_program (*)(cl_context, cl_uint, const char **, const size_t *, cl_int *);
using clReleaseKernel_func           = cl_int (*)(cl_kernel kernel);
using clGetDeviceInfo_func           = cl_int (*)(cl_device_id, cl_device_info, size_t, void *, size_t *);
using clGetDeviceIDs_func            = cl_int (*)(cl_platform_id, cl_device_type, cl_uint, cl_device_id *, cl_uint *);

class CLSymbols
{
private:
    CLSymbols()
    {
        void *handle = dlopen("libOpenCL.so", RTLD_LAZY | RTLD_LOCAL);
        if(handle == nullptr)
        {
            std::cerr << "Can't load libOpenCL.so: " << dlerror() << std::endl;
        }
        else
        {
            clBuildProgram            = reinterpret_cast<clBuildProgram_func>(dlsym(handle, "clBuildProgram"));
            clEnqueueNDRangeKernel    = reinterpret_cast<clEnqueueNDRangeKernel_func>(dlsym(handle, "clEnqueueNDRangeKernel"));
            clSetKernelArg            = reinterpret_cast<clSetKernelArg_func>(dlsym(handle, "clSetKernelArg"));
            clReleaseKernel           = reinterpret_cast<clReleaseKernel_func>(dlsym(handle, "clReleaseKernel"));
            clCreateProgramWithSource = reinterpret_cast<clCreateProgramWithSource_func>(dlsym(handle, "clCreateProgramWithSource"));
            clCreateBuffer            = reinterpret_cast<clCreateBuffer_func>(dlsym(handle, "clCreateBuffer"));
            clRetainKernel            = reinterpret_cast<clRetainKernel_func>(dlsym(handle, "clRetainKernel"));
            clCreateKernel            = reinterpret_cast<clCreateKernel_func>(dlsym(handle, "clCreateKernel"));
            clGetProgramInfo          = reinterpret_cast<clGetProgramInfo_func>(dlsym(handle, "clGetProgramInfo"));
            clFlush                   = reinterpret_cast<clFlush_func>(dlsym(handle, "clFlush"));
            clReleaseProgram          = reinterpret_cast<clReleaseProgram_func>(dlsym(handle, "clReleaseProgram"));
            clRetainContext           = reinterpret_cast<clRetainContext_func>(dlsym(handle, "clRetainContext"));
            clCreateProgramWithBinary = reinterpret_cast<clCreateProgramWithBinary_func>(dlsym(handle, "clCreateProgramWithBinary"));
            clReleaseCommandQueue     = reinterpret_cast<clReleaseCommandQueue_func>(dlsym(handle, "clReleaseCommandQueue"));
            clEnqueueMapBuffer        = reinterpret_cast<clEnqueueMapBuffer_func>(dlsym(handle, "clEnqueueMapBuffer"));
            clRetainProgram           = reinterpret_cast<clRetainProgram_func>(dlsym(handle, "clRetainProgram"));
            clGetProgramBuildInfo     = reinterpret_cast<clGetProgramBuildInfo_func>(dlsym(handle, "clGetProgramBuildInfo"));
            clEnqueueReadBuffer       = reinterpret_cast<clEnqueueReadBuffer_func>(dlsym(handle, "clEnqueueReadBuffer"));
            clEnqueueWriteBuffer      = reinterpret_cast<clEnqueueWriteBuffer_func>(dlsym(handle, "clEnqueueWriteBuffer"));
            clReleaseEvent            = reinterpret_cast<clReleaseEvent_func>(dlsym(handle, "clReleaseEvent"));
            clReleaseContext          = reinterpret_cast<clReleaseContext_func>(dlsym(handle, "clReleaseContext"));
            clRetainCommandQueue      = reinterpret_cast<clRetainCommandQueue_func>(dlsym(handle, "clRetainCommandQueue"));
            clEnqueueUnmapMemObject   = reinterpret_cast<clEnqueueUnmapMemObject_func>(dlsym(handle, "clEnqueueUnmapMemObject"));
            clReleaseMemObject        = reinterpret_cast<clReleaseMemObject_func>(dlsym(handle, "clReleaseMemObject"));
            clGetDeviceInfo           = reinterpret_cast<clGetDeviceInfo_func>(dlsym(handle, "clGetDeviceInfo"));
            clGetDeviceIDs            = reinterpret_cast<clGetDeviceIDs_func>(dlsym(handle, "clGetDeviceIDs"));
            dlclose(handle);
        }
    }

public:
    static CLSymbols &get()
    {
        static CLSymbols symbols = CLSymbols();
        return symbols;
    }

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
    clReleaseMemObject_func        clReleaseMemObject        = nullptr;
    clGetDeviceInfo_func           clGetDeviceInfo           = nullptr;
    clGetDeviceIDs_func            clGetDeviceIDs            = nullptr;
};

bool arm_compute::opencl_is_available()
{
    return CLSymbols::get().clBuildProgram != nullptr;
}

cl_int clBuildProgram(
    cl_program          program,
    cl_uint             num_devices,
    const cl_device_id *device_list,
    const char         *options,
    void(CL_CALLBACK *pfn_notify)(cl_program program, void *user_data),
    void *user_data)
{
    auto func = CLSymbols::get().clBuildProgram;
    if(func != nullptr)
    {
        return func(program, num_devices, device_list, options, pfn_notify, user_data);
    }
    else
    {
        return CL_OUT_OF_RESOURCES;
    }
}

cl_int clEnqueueNDRangeKernel(
    cl_command_queue command_queue,
    cl_kernel        kernel,
    cl_uint          work_dim,
    const size_t    *global_work_offset,
    const size_t    *global_work_size,
    const size_t    *local_work_size,
    cl_uint          num_events_in_wait_list,
    const cl_event *event_wait_list,
    cl_event        *event)
{
    auto func = CLSymbols::get().clEnqueueNDRangeKernel;
    if(func != nullptr)
    {
        return func(command_queue, kernel, work_dim, global_work_offset, global_work_size, local_work_size, num_events_in_wait_list, event_wait_list, event);
    }
    else
    {
        return CL_OUT_OF_RESOURCES;
    }
}

cl_int clSetKernelArg(
    cl_kernel   kernel,
    cl_uint     arg_index,
    size_t      arg_size,
    const void *arg_value)
{
    auto func = CLSymbols::get().clSetKernelArg;
    if(func != nullptr)
    {
        return func(kernel, arg_index, arg_size, arg_value);
    }
    else
    {
        return CL_OUT_OF_RESOURCES;
    }
}

cl_int clReleaseMemObject(cl_mem memobj)
{
    auto func = CLSymbols::get().clReleaseMemObject;
    if(func != nullptr)
    {
        return func(memobj);
    }
    else
    {
        return CL_OUT_OF_RESOURCES;
    }
}

cl_int clEnqueueUnmapMemObject(
    cl_command_queue command_queue,
    cl_mem           memobj,
    void            *mapped_ptr,
    cl_uint          num_events_in_wait_list,
    const cl_event *event_wait_list,
    cl_event        *event)
{
    auto func = CLSymbols::get().clEnqueueUnmapMemObject;
    if(func != nullptr)
    {
        return func(command_queue, memobj, mapped_ptr, num_events_in_wait_list, event_wait_list, event);
    }
    else
    {
        return CL_OUT_OF_RESOURCES;
    }
}

cl_int clRetainCommandQueue(cl_command_queue command_queue)
{
    auto func = CLSymbols::get().clRetainCommandQueue;
    if(func != nullptr)
    {
        return func(command_queue);
    }
    else
    {
        return CL_OUT_OF_RESOURCES;
    }
}

cl_int clReleaseContext(cl_context context)
{
    auto func = CLSymbols::get().clReleaseContext;
    if(func != nullptr)
    {
        return func(context);
    }
    else
    {
        return CL_OUT_OF_RESOURCES;
    }
}
cl_int clReleaseEvent(cl_event event)
{
    auto func = CLSymbols::get().clReleaseEvent;
    if(func != nullptr)
    {
        return func(event);
    }
    else
    {
        return CL_OUT_OF_RESOURCES;
    }
}

cl_int clEnqueueWriteBuffer(
    cl_command_queue command_queue,
    cl_mem           buffer,
    cl_bool          blocking_write,
    size_t           offset,
    size_t           size,
    const void      *ptr,
    cl_uint          num_events_in_wait_list,
    const cl_event *event_wait_list,
    cl_event        *event)
{
    auto func = CLSymbols::get().clEnqueueWriteBuffer;
    if(func != nullptr)
    {
        return func(command_queue, buffer, blocking_write, offset, size, ptr, num_events_in_wait_list, event_wait_list, event);
    }
    else
    {
        return CL_OUT_OF_RESOURCES;
    }
}

cl_int clEnqueueReadBuffer(
    cl_command_queue command_queue,
    cl_mem           buffer,
    cl_bool          blocking_read,
    size_t           offset,
    size_t           size,
    void            *ptr,
    cl_uint          num_events_in_wait_list,
    const cl_event *event_wait_list,
    cl_event        *event)
{
    auto func = CLSymbols::get().clEnqueueReadBuffer;
    if(func != nullptr)
    {
        return func(command_queue, buffer, blocking_read, offset, size, ptr, num_events_in_wait_list, event_wait_list, event);
    }
    else
    {
        return CL_OUT_OF_RESOURCES;
    }
}

cl_int clGetProgramBuildInfo(
    cl_program            program,
    cl_device_id          device,
    cl_program_build_info param_name,
    size_t                param_value_size,
    void                 *param_value,
    size_t               *param_value_size_ret)
{
    auto func = CLSymbols::get().clGetProgramBuildInfo;
    if(func != nullptr)
    {
        return func(program, device, param_name, param_value_size, param_value, param_value_size_ret);
    }
    else
    {
        return CL_OUT_OF_RESOURCES;
    }
}

cl_int clRetainProgram(cl_program program)
{
    auto func = CLSymbols::get().clRetainProgram;
    if(func != nullptr)
    {
        return func(program);
    }
    else
    {
        return CL_OUT_OF_RESOURCES;
    }
}

void *clEnqueueMapBuffer(
    cl_command_queue command_queue,
    cl_mem           buffer,
    cl_bool          blocking_map,
    cl_map_flags     map_flags,
    size_t           offset,
    size_t           size,
    cl_uint          num_events_in_wait_list,
    const cl_event *event_wait_list,
    cl_event        *event,
    cl_int          *errcode_ret)
{
    auto func = CLSymbols::get().clEnqueueMapBuffer;
    if(func != nullptr)
    {
        return func(command_queue, buffer, blocking_map, map_flags, offset, size, num_events_in_wait_list, event_wait_list, event, errcode_ret);
    }
    else
    {
        if(errcode_ret != nullptr)
        {
            *errcode_ret = CL_OUT_OF_RESOURCES;
        }
        return nullptr;
    }
}

cl_int clReleaseCommandQueue(cl_command_queue command_queue)
{
    auto func = CLSymbols::get().clReleaseCommandQueue;
    if(func != nullptr)
    {
        return func(command_queue);
    }
    else
    {
        return CL_OUT_OF_RESOURCES;
    }
}

cl_program clCreateProgramWithBinary(
    cl_context            context,
    cl_uint               num_devices,
    const cl_device_id   *device_list,
    const size_t         *lengths,
    const unsigned char **binaries,
    cl_int               *binary_status,
    cl_int               *errcode_ret)
{
    auto func = CLSymbols::get().clCreateProgramWithBinary;
    if(func != nullptr)
    {
        return func(context, num_devices, device_list, lengths, binaries, binary_status, errcode_ret);
    }
    else
    {
        if(errcode_ret != nullptr)
        {
            *errcode_ret = CL_OUT_OF_RESOURCES;
        }
        return nullptr;
    }
}

cl_int clRetainContext(cl_context context)
{
    auto func = CLSymbols::get().clRetainContext;
    if(func != nullptr)
    {
        return func(context);
    }
    else
    {
        return CL_OUT_OF_RESOURCES;
    }
}

cl_int clReleaseProgram(cl_program program)
{
    auto func = CLSymbols::get().clReleaseProgram;
    if(func != nullptr)
    {
        return func(program);
    }
    else
    {
        return CL_OUT_OF_RESOURCES;
    }
}

cl_int clFlush(cl_command_queue command_queue)
{
    auto func = CLSymbols::get().clFlush;
    if(func != nullptr)
    {
        return func(command_queue);
    }
    else
    {
        return CL_OUT_OF_RESOURCES;
    }
}

cl_int clGetProgramInfo(
    cl_program      program,
    cl_program_info param_name,
    size_t          param_value_size,
    void           *param_value,
    size_t         *param_value_size_ret)
{
    auto func = CLSymbols::get().clGetProgramInfo;
    if(func != nullptr)
    {
        return func(program, param_name, param_value_size, param_value, param_value_size_ret);
    }
    else
    {
        return CL_OUT_OF_RESOURCES;
    }
}

cl_kernel clCreateKernel(
    cl_program  program,
    const char *kernel_name,
    cl_int     *errcode_ret)
{
    auto func = CLSymbols::get().clCreateKernel;
    if(func != nullptr)
    {
        return func(program, kernel_name, errcode_ret);
    }
    else
    {
        if(errcode_ret != nullptr)
        {
            *errcode_ret = CL_OUT_OF_RESOURCES;
        }
        return nullptr;
    }
}

cl_int clRetainKernel(cl_kernel kernel)
{
    auto func = CLSymbols::get().clRetainKernel;
    if(func != nullptr)
    {
        return func(kernel);
    }
    else
    {
        return CL_OUT_OF_RESOURCES;
    }
}

cl_mem clCreateBuffer(
    cl_context   context,
    cl_mem_flags flags,
    size_t       size,
    void        *host_ptr,
    cl_int      *errcode_ret)
{
    auto func = CLSymbols::get().clCreateBuffer;
    if(func != nullptr)
    {
        return func(context, flags, size, host_ptr, errcode_ret);
    }
    else
    {
        if(errcode_ret != nullptr)
        {
            *errcode_ret = CL_OUT_OF_RESOURCES;
        }
        return nullptr;
    }
}

cl_program clCreateProgramWithSource(
    cl_context    context,
    cl_uint       count,
    const char **strings,
    const size_t *lengths,
    cl_int       *errcode_ret)
{
    auto func = CLSymbols::get().clCreateProgramWithSource;
    if(func != nullptr)
    {
        return func(context, count, strings, lengths, errcode_ret);
    }
    else
    {
        if(errcode_ret != nullptr)
        {
            *errcode_ret = CL_OUT_OF_RESOURCES;
        }
        return nullptr;
    }
}

cl_int clReleaseKernel(cl_kernel kernel)
{
    auto func = CLSymbols::get().clReleaseKernel;
    if(func != nullptr)
    {
        return func(kernel);
    }
    else
    {
        return CL_OUT_OF_RESOURCES;
    }
}

cl_int clGetDeviceIDs(cl_platform_id platform,
                      cl_device_type device_type,
                      cl_uint        num_entries,
                      cl_device_id *devices,
                      cl_uint       *num_devices)
{
    auto func = CLSymbols::get().clGetDeviceIDs;
    if(func != nullptr)
    {
        return func(platform, device_type, num_entries, devices, num_devices);
    }
    else
    {
        return CL_OUT_OF_RESOURCES;
    }
}

cl_int clGetDeviceInfo(cl_device_id   device,
                       cl_device_info param_name,
                       size_t         param_value_size,
                       void          *param_value,
                       size_t        *param_value_size_ret)
{
    auto func = CLSymbols::get().clGetDeviceInfo;
    if(func != nullptr)
    {
        return func(device, param_name, param_value_size, param_value, param_value_size_ret);
    }
    else
    {
        return CL_OUT_OF_RESOURCES;
    }
}
