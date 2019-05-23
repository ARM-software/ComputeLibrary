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
#ifndef __ARM_COMPUTE_OPENCL_H__
#define __ARM_COMPUTE_OPENCL_H__

#include <string>
#include <utility>

/* Configure the Khronos C++ wrapper to target OpenCL 1.2: */
#ifndef ARM_COMPUTE_NO_EXCEPTIONS
#define CL_HPP_ENABLE_EXCEPTIONS
#endif // ARM_COMPUTE_NO_EXCEPTIONS
#define CL_TARGET_OPENCL_VERSION 200
#define CL_HPP_TARGET_OPENCL_VERSION 110
#define CL_HPP_MINIMUM_OPENCL_VERSION 110
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++"
#pragma GCC diagnostic ignored "-Wignored-qualifiers"
#if defined(__GNUG__) && __GNUG__ >= 8
#pragma GCC diagnostic ignored "-Wcatch-value"
#endif // defined(__GNUG__) && __GNUG__ >= 8
#include <CL/cl2.hpp>
#pragma GCC diagnostic pop

namespace cl
{
static const NDRange Range_128_1 = NDRange(128, 1);
} // namespace cl

namespace arm_compute
{
/** Check if OpenCL is available.
 *
 * @return True if OpenCL is available.
 */
bool opencl_is_available();

/** Class for loading OpenCL symbols. */
class CLSymbols final
{
private:
    CLSymbols() = default;
    void load_symbols(void *handle);

public:
    /** Get the static instance of CLSymbols.
     *
     * @return The static instance of CLSymbols.
     */
    static CLSymbols &get();
    /** Load symbols from the given OpenCL library path.
     *
     * @param[in] library Path to the OpenCL library.
     *
     * @return True if loading the library is successful.
     */
    bool load(const std::string &library);
    /** Load symbols from any of the default OpenCL library names.
     *
     * @return True if loading any library is successful.
     */
    bool load_default();

#define DECLARE_FUNCTION_PTR(func_name) \
    std::function<decltype(func_name)> func_name##_ptr = nullptr

    DECLARE_FUNCTION_PTR(clCreateContext);
    DECLARE_FUNCTION_PTR(clCreateContextFromType);
    DECLARE_FUNCTION_PTR(clCreateCommandQueue);
    DECLARE_FUNCTION_PTR(clGetContextInfo);
    DECLARE_FUNCTION_PTR(clBuildProgram);
    DECLARE_FUNCTION_PTR(clEnqueueNDRangeKernel);
    DECLARE_FUNCTION_PTR(clSetKernelArg);
    DECLARE_FUNCTION_PTR(clReleaseKernel);
    DECLARE_FUNCTION_PTR(clCreateProgramWithSource);
    DECLARE_FUNCTION_PTR(clCreateBuffer);
    DECLARE_FUNCTION_PTR(clRetainKernel);
    DECLARE_FUNCTION_PTR(clCreateKernel);
    DECLARE_FUNCTION_PTR(clGetProgramInfo);
    DECLARE_FUNCTION_PTR(clFlush);
    DECLARE_FUNCTION_PTR(clFinish);
    DECLARE_FUNCTION_PTR(clReleaseProgram);
    DECLARE_FUNCTION_PTR(clRetainContext);
    DECLARE_FUNCTION_PTR(clCreateProgramWithBinary);
    DECLARE_FUNCTION_PTR(clReleaseCommandQueue);
    DECLARE_FUNCTION_PTR(clEnqueueMapBuffer);
    DECLARE_FUNCTION_PTR(clRetainProgram);
    DECLARE_FUNCTION_PTR(clGetProgramBuildInfo);
    DECLARE_FUNCTION_PTR(clEnqueueReadBuffer);
    DECLARE_FUNCTION_PTR(clEnqueueWriteBuffer);
    DECLARE_FUNCTION_PTR(clReleaseEvent);
    DECLARE_FUNCTION_PTR(clReleaseContext);
    DECLARE_FUNCTION_PTR(clRetainCommandQueue);
    DECLARE_FUNCTION_PTR(clEnqueueUnmapMemObject);
    DECLARE_FUNCTION_PTR(clRetainMemObject);
    DECLARE_FUNCTION_PTR(clReleaseMemObject);
    DECLARE_FUNCTION_PTR(clGetDeviceInfo);
    DECLARE_FUNCTION_PTR(clGetDeviceIDs);
    DECLARE_FUNCTION_PTR(clGetMemObjectInfo);
    DECLARE_FUNCTION_PTR(clRetainEvent);
    DECLARE_FUNCTION_PTR(clGetPlatformIDs);
    DECLARE_FUNCTION_PTR(clGetKernelWorkGroupInfo);
    DECLARE_FUNCTION_PTR(clGetCommandQueueInfo);
    DECLARE_FUNCTION_PTR(clGetKernelInfo);
    DECLARE_FUNCTION_PTR(clGetEventProfilingInfo);
    DECLARE_FUNCTION_PTR(clSVMAlloc);
    DECLARE_FUNCTION_PTR(clSVMFree);
    DECLARE_FUNCTION_PTR(clEnqueueSVMMap);
    DECLARE_FUNCTION_PTR(clEnqueueSVMUnmap);
    DECLARE_FUNCTION_PTR(clEnqueueMarker);
    DECLARE_FUNCTION_PTR(clWaitForEvents);

    // Third-party extensions
    DECLARE_FUNCTION_PTR(clImportMemoryARM);

#undef DECLARE_FUNCTION_PTR

private:
    std::pair<bool, bool> _loaded{ false, false };
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_OPENCL_H__ */
