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

#define LOAD_FUNCTION_PTR(func_name, handle) \
    func_name##_ptr = reinterpret_cast<decltype(func_name) *>(dlsym(handle, #func_name));

    LOAD_FUNCTION_PTR(clCreateContextFromType, handle);
    LOAD_FUNCTION_PTR(clCreateCommandQueue, handle);
    LOAD_FUNCTION_PTR(clGetContextInfo, handle);
    LOAD_FUNCTION_PTR(clBuildProgram, handle);
    LOAD_FUNCTION_PTR(clEnqueueNDRangeKernel, handle);
    LOAD_FUNCTION_PTR(clSetKernelArg, handle);
    LOAD_FUNCTION_PTR(clReleaseKernel, handle);
    LOAD_FUNCTION_PTR(clCreateProgramWithSource, handle);
    LOAD_FUNCTION_PTR(clCreateBuffer, handle);
    LOAD_FUNCTION_PTR(clRetainKernel, handle);
    LOAD_FUNCTION_PTR(clCreateKernel, handle);
    LOAD_FUNCTION_PTR(clGetProgramInfo, handle);
    LOAD_FUNCTION_PTR(clFlush, handle);
    LOAD_FUNCTION_PTR(clFinish, handle);
    LOAD_FUNCTION_PTR(clReleaseProgram, handle);
    LOAD_FUNCTION_PTR(clRetainContext, handle);
    LOAD_FUNCTION_PTR(clCreateProgramWithBinary, handle);
    LOAD_FUNCTION_PTR(clReleaseCommandQueue, handle);
    LOAD_FUNCTION_PTR(clEnqueueMapBuffer, handle);
    LOAD_FUNCTION_PTR(clRetainProgram, handle);
    LOAD_FUNCTION_PTR(clGetProgramBuildInfo, handle);
    LOAD_FUNCTION_PTR(clEnqueueReadBuffer, handle);
    LOAD_FUNCTION_PTR(clEnqueueWriteBuffer, handle);
    LOAD_FUNCTION_PTR(clReleaseEvent, handle);
    LOAD_FUNCTION_PTR(clReleaseContext, handle);
    LOAD_FUNCTION_PTR(clRetainCommandQueue, handle);
    LOAD_FUNCTION_PTR(clEnqueueUnmapMemObject, handle);
    LOAD_FUNCTION_PTR(clRetainMemObject, handle);
    LOAD_FUNCTION_PTR(clReleaseMemObject, handle);
    LOAD_FUNCTION_PTR(clGetDeviceInfo, handle);
    LOAD_FUNCTION_PTR(clGetDeviceIDs, handle);
    LOAD_FUNCTION_PTR(clRetainEvent, handle);
    LOAD_FUNCTION_PTR(clGetPlatformIDs, handle);
    LOAD_FUNCTION_PTR(clGetKernelWorkGroupInfo, handle);

#undef LOAD_FUNCTION_PTR

    //Don't call dlclose(handle) or all the symbols will be unloaded !

    // Disable default loading and set status to successful
    _loaded = std::make_pair(true, true);

    return true;
}

bool opencl_is_available()
{
    CLSymbols::get().load_default();
    return CLSymbols::get().clBuildProgram_ptr != nullptr;
}
} // namespace arm_compute

cl_int clGetContextInfo(cl_context      context,
                        cl_context_info param_name,
                        size_t          param_value_size,
                        void           *param_value,
                        size_t         *param_value_size_ret)
{
    arm_compute::CLSymbols::get().load_default();
    auto func = arm_compute::CLSymbols::get().clGetContextInfo_ptr;
    if(func != nullptr)
    {
        return func(context, param_name, param_value_size, param_value, param_value_size_ret);
    }
    else
    {
        return CL_OUT_OF_RESOURCES;
    }
}

cl_command_queue clCreateCommandQueue(cl_context                  context,
                                      cl_device_id                device,
                                      cl_command_queue_properties properties,
                                      cl_int                     *errcode_ret)
{
    arm_compute::CLSymbols::get().load_default();
    auto func = arm_compute::CLSymbols::get().clCreateCommandQueue_ptr;
    if(func != nullptr)
    {
        return func(context, device, properties, errcode_ret);
    }
    else
    {
        return nullptr;
    }
}

cl_context clCreateContextFromType(const cl_context_properties *properties,
                                   cl_device_type               device_type,
                                   void (*pfn_notify)(const char *, const void *, size_t, void *),
                                   void   *user_data,
                                   cl_int *errcode_ret)
{
    arm_compute::CLSymbols::get().load_default();
    auto func = arm_compute::CLSymbols::get().clCreateContextFromType_ptr;
    if(func != nullptr)
    {
        return func(properties, device_type, pfn_notify, user_data, errcode_ret);
    }
    else
    {
        return nullptr;
    }
}

cl_int clBuildProgram(
    cl_program          program,
    cl_uint             num_devices,
    const cl_device_id *device_list,
    const char         *options,
    void(CL_CALLBACK *pfn_notify)(cl_program program, void *user_data),
    void *user_data)
{
    arm_compute::CLSymbols::get().load_default();
    auto func = arm_compute::CLSymbols::get().clBuildProgram_ptr;
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
    auto func = arm_compute::CLSymbols::get().clEnqueueNDRangeKernel_ptr;
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
    auto func = arm_compute::CLSymbols::get().clSetKernelArg_ptr;
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
    auto func = arm_compute::CLSymbols::get().clRetainMemObject_ptr;
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
    auto func = arm_compute::CLSymbols::get().clReleaseMemObject_ptr;
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
    auto func = arm_compute::CLSymbols::get().clEnqueueUnmapMemObject_ptr;
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
    auto func = arm_compute::CLSymbols::get().clRetainCommandQueue_ptr;
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
    auto func = arm_compute::CLSymbols::get().clReleaseContext_ptr;
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
    auto func = arm_compute::CLSymbols::get().clReleaseEvent_ptr;
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
    auto func = arm_compute::CLSymbols::get().clEnqueueWriteBuffer_ptr;
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
    auto func = arm_compute::CLSymbols::get().clEnqueueReadBuffer_ptr;
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
    auto func = arm_compute::CLSymbols::get().clGetProgramBuildInfo_ptr;
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
    auto func = arm_compute::CLSymbols::get().clRetainProgram_ptr;
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
    auto func = arm_compute::CLSymbols::get().clEnqueueMapBuffer_ptr;
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
    auto func = arm_compute::CLSymbols::get().clReleaseCommandQueue_ptr;
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
    auto func = arm_compute::CLSymbols::get().clCreateProgramWithBinary_ptr;
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
    auto func = arm_compute::CLSymbols::get().clRetainContext_ptr;
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
    auto func = arm_compute::CLSymbols::get().clReleaseProgram_ptr;
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
    auto func = arm_compute::CLSymbols::get().clFlush_ptr;
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
    auto func = arm_compute::CLSymbols::get().clFinish_ptr;
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
    auto func = arm_compute::CLSymbols::get().clGetProgramInfo_ptr;
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
    auto func = arm_compute::CLSymbols::get().clCreateKernel_ptr;
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
    auto func = arm_compute::CLSymbols::get().clRetainKernel_ptr;
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
    auto func = arm_compute::CLSymbols::get().clCreateBuffer_ptr;
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
    auto func = arm_compute::CLSymbols::get().clCreateProgramWithSource_ptr;
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
    auto func = arm_compute::CLSymbols::get().clReleaseKernel_ptr;
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
    auto func = arm_compute::CLSymbols::get().clGetDeviceIDs_ptr;
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
    auto func = arm_compute::CLSymbols::get().clGetDeviceInfo_ptr;
    if(func != nullptr)
    {
        return func(device, param_name, param_value_size, param_value, param_value_size_ret);
    }
    else
    {
        return CL_OUT_OF_RESOURCES;
    }
}

cl_int clRetainEvent(cl_event event)
{
    arm_compute::CLSymbols::get().load_default();
    auto func = arm_compute::CLSymbols::get().clRetainEvent_ptr;
    if(func != nullptr)
    {
        return func(event);
    }
    else
    {
        return CL_OUT_OF_RESOURCES;
    }
}

cl_int clGetPlatformIDs(cl_uint num_entries, cl_platform_id *platforms, cl_uint *num_platforms)
{
    arm_compute::CLSymbols::get().load_default();
    auto func = arm_compute::CLSymbols::get().clGetPlatformIDs_ptr;
    if(func != nullptr)
    {
        return func(num_entries, platforms, num_platforms);
    }
    else
    {
        return CL_OUT_OF_RESOURCES;
    }
}

cl_int
clGetKernelWorkGroupInfo(cl_kernel                 kernel,
                         cl_device_id              device,
                         cl_kernel_work_group_info param_name,
                         size_t                    param_value_size,
                         void                     *param_value,
                         size_t                   *param_value_size_ret)
{
    arm_compute::CLSymbols::get().load_default();
    auto func = arm_compute::CLSymbols::get().clGetKernelWorkGroupInfo_ptr;
    if(func != nullptr)
    {
        return func(kernel, device, param_name, param_value_size, param_value, param_value_size_ret);
    }
    else
    {
        return CL_OUT_OF_RESOURCES;
    }
}
