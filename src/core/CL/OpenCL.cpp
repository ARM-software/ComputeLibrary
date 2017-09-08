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

namespace arm_compute
{
CLSymbols &CLSymbols::get()
{
    static CLSymbols symbols;
    return symbols;
}

bool CLSymbols::load_default()
{
    static const std::vector<std::string> libraries{ "libOpenCL.so", "libGLES_mali.so", "libmali.so" };

    if(_loaded.first)
    {
        return _loaded.second;
    }

    // Indicate that default loading has been tried
    _loaded.first = true;

    for(const auto &lib : libraries)
    {
        if(load(lib))
        {
            return true;
        }
    }

    std::cerr << "Couldn't find any OpenCL library.\n";
    return false;
}

bool CLSymbols::load(const std::string &library)
{
    void *handle = dlopen(library.c_str(), RTLD_LAZY | RTLD_LOCAL);

    if(handle == nullptr)
    {
        std::cerr << "Can't load " << library << ": " << dlerror() << "\n";
        // Set status of loading to failed
        _loaded.second = false;
        return false;
    }

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
    clFinish                  = reinterpret_cast<clFinish_func>(dlsym(handle, "clFinish"));
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
    clRetainMemObject         = reinterpret_cast<clRetainMemObject_func>(dlsym(handle, "clRetainMemObject"));
    clReleaseMemObject        = reinterpret_cast<clReleaseMemObject_func>(dlsym(handle, "clReleaseMemObject"));
    clGetDeviceInfo           = reinterpret_cast<clGetDeviceInfo_func>(dlsym(handle, "clGetDeviceInfo"));
    clGetDeviceIDs            = reinterpret_cast<clGetDeviceIDs_func>(dlsym(handle, "clGetDeviceIDs"));

    dlclose(handle);

    // Disable default loading and set status to successful
    _loaded = std::make_pair(true, true);

    return true;
}

bool opencl_is_available()
{
    CLSymbols::get().load_default();
    return CLSymbols::get().clBuildProgram != nullptr;
}
} // namespace arm_compute

cl_int clBuildProgram(
    cl_program          program,
    cl_uint             num_devices,
    const cl_device_id *device_list,
    const char         *options,
    void(CL_CALLBACK *pfn_notify)(cl_program program, void *user_data),
    void *user_data)
{
    arm_compute::CLSymbols::get().load_default();
    auto func = arm_compute::CLSymbols::get().clBuildProgram;
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
    arm_compute::CLSymbols::get().load_default();
    auto func = arm_compute::CLSymbols::get().clEnqueueNDRangeKernel;
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
    arm_compute::CLSymbols::get().load_default();
    auto func = arm_compute::CLSymbols::get().clSetKernelArg;
    if(func != nullptr)
    {
        return func(kernel, arg_index, arg_size, arg_value);
    }
    else
    {
        return CL_OUT_OF_RESOURCES;
    }
}

cl_int clRetainMemObject(cl_mem memobj)
{
    arm_compute::CLSymbols::get().load_default();
    auto func = arm_compute::CLSymbols::get().clRetainMemObject;
    if(func != nullptr)
    {
        return func(memobj);
    }
    else
    {
        return CL_OUT_OF_RESOURCES;
    }
}

cl_int clReleaseMemObject(cl_mem memobj)
{
    arm_compute::CLSymbols::get().load_default();
    auto func = arm_compute::CLSymbols::get().clReleaseMemObject;
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
    arm_compute::CLSymbols::get().load_default();
    auto func = arm_compute::CLSymbols::get().clEnqueueUnmapMemObject;
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
    arm_compute::CLSymbols::get().load_default();
    auto func = arm_compute::CLSymbols::get().clRetainCommandQueue;
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
    arm_compute::CLSymbols::get().load_default();
    auto func = arm_compute::CLSymbols::get().clReleaseContext;
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
    arm_compute::CLSymbols::get().load_default();
    auto func = arm_compute::CLSymbols::get().clReleaseEvent;
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
    arm_compute::CLSymbols::get().load_default();
    auto func = arm_compute::CLSymbols::get().clEnqueueWriteBuffer;
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
    arm_compute::CLSymbols::get().load_default();
    auto func = arm_compute::CLSymbols::get().clEnqueueReadBuffer;
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
    arm_compute::CLSymbols::get().load_default();
    auto func = arm_compute::CLSymbols::get().clGetProgramBuildInfo;
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
    arm_compute::CLSymbols::get().load_default();
    auto func = arm_compute::CLSymbols::get().clRetainProgram;
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
    arm_compute::CLSymbols::get().load_default();
    auto func = arm_compute::CLSymbols::get().clEnqueueMapBuffer;
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
    arm_compute::CLSymbols::get().load_default();
    auto func = arm_compute::CLSymbols::get().clReleaseCommandQueue;
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
    arm_compute::CLSymbols::get().load_default();
    auto func = arm_compute::CLSymbols::get().clCreateProgramWithBinary;
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
    arm_compute::CLSymbols::get().load_default();
    auto func = arm_compute::CLSymbols::get().clRetainContext;
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
    arm_compute::CLSymbols::get().load_default();
    auto func = arm_compute::CLSymbols::get().clReleaseProgram;
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
    arm_compute::CLSymbols::get().load_default();
    auto func = arm_compute::CLSymbols::get().clFlush;
    if(func != nullptr)
    {
        return func(command_queue);
    }
    else
    {
        return CL_OUT_OF_RESOURCES;
    }
}

cl_int clFinish(cl_command_queue command_queue)
{
    arm_compute::CLSymbols::get().load_default();
    auto func = arm_compute::CLSymbols::get().clFinish;
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
    arm_compute::CLSymbols::get().load_default();
    auto func = arm_compute::CLSymbols::get().clGetProgramInfo;
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
    arm_compute::CLSymbols::get().load_default();
    auto func = arm_compute::CLSymbols::get().clCreateKernel;
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
    arm_compute::CLSymbols::get().load_default();
    auto func = arm_compute::CLSymbols::get().clRetainKernel;
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
    arm_compute::CLSymbols::get().load_default();
    auto func = arm_compute::CLSymbols::get().clCreateBuffer;
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
    arm_compute::CLSymbols::get().load_default();
    auto func = arm_compute::CLSymbols::get().clCreateProgramWithSource;
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
    arm_compute::CLSymbols::get().load_default();
    auto func = arm_compute::CLSymbols::get().clReleaseKernel;
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
    arm_compute::CLSymbols::get().load_default();
    auto func = arm_compute::CLSymbols::get().clGetDeviceIDs;
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
    arm_compute::CLSymbols::get().load_default();
    auto func = arm_compute::CLSymbols::get().clGetDeviceInfo;
    if(func != nullptr)
    {
        return func(device, param_name, param_value_size, param_value, param_value_size_ret);
    }
    else
    {
        return CL_OUT_OF_RESOURCES;
    }
}
