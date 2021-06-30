/*
 * Copyright (c) 2016-2021 Arm Limited.
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
#include "arm_compute/core/CL/CLKernelLibrary.h"

#include "arm_compute/core/Error.h"
#include "src/core/gpu/cl/ClKernelLibrary.h"

#include <algorithm>
#include <array>
#include <fstream>
#include <utility>
#include <vector>

namespace arm_compute
{
CLKernelLibrary::CLKernelLibrary()
    : _compile_context()
{
    opencl_is_available(); // Make sure the OpenCL symbols are initialised *before* the CLKernelLibrary is built
}

CLKernelLibrary &CLKernelLibrary::get()
{
    static CLKernelLibrary _kernel_library;
    return _kernel_library;
}

Kernel CLKernelLibrary::create_kernel(const std::string &kernel_name, const std::set<std::string> &build_options_set) const
{
    const opencl::ClKernelLibrary &klib = opencl::ClKernelLibrary::get();

    const std::string  program_name = klib.program_name(kernel_name);
    auto               program      = klib.program(program_name);
    const std::string &kernel_path  = CLKernelLibrary::get().get_kernel_path();

    return _compile_context.create_kernel(kernel_name, program_name, program.program, kernel_path, build_options_set, program.is_binary);
}

std::string CLKernelLibrary::get_program_name(const std::string &kernel_name) const
{
    return opencl::ClKernelLibrary::get().program_name(kernel_name);
}

void CLKernelLibrary::init(std::string kernel_path, cl::Context context, cl::Device device)
{
    _compile_context = CLCompileContext(context, device);
    opencl::ClKernelLibrary::get().set_kernel_path(kernel_path);
}

void CLKernelLibrary::set_kernel_path(const std::string &kernel_path)
{
    opencl::ClKernelLibrary::get().set_kernel_path(kernel_path);
}

cl::Context &CLKernelLibrary::context()
{
    return _compile_context.context();
}

const cl::Device &CLKernelLibrary::get_device()
{
    return _compile_context.get_device();
}

void CLKernelLibrary::set_device(cl::Device device)
{
    _compile_context.set_device(device);
}

void CLKernelLibrary::set_context(cl::Context context)
{
    _compile_context.set_context(context);
}

std::string CLKernelLibrary::get_kernel_path()
{
    return opencl::ClKernelLibrary::get().kernel_path();
}

void CLKernelLibrary::clear_programs_cache()
{
    _compile_context.clear_programs_cache();
}

const std::map<std::string, cl::Program> &CLKernelLibrary::get_built_programs() const
{
    return _compile_context.get_built_programs();
}

void CLKernelLibrary::add_built_program(const std::string &built_program_name, const cl::Program &program)
{
    _compile_context.add_built_program(built_program_name, program);
}

bool CLKernelLibrary::fp16_supported() const
{
    return _compile_context.fp16_supported();
}

bool CLKernelLibrary::int64_base_atomics_supported() const
{
    return _compile_context.int64_base_atomics_supported();
}

bool CLKernelLibrary::is_wbsm_supported()
{
    return _compile_context.is_wbsm_supported();
}

std::pair<std::string, bool> CLKernelLibrary::get_program(const std::string &program_name) const
{
    auto program_info = opencl::ClKernelLibrary::get().program(program_name);
    return std::make_pair(std::move(program_info.program), program_info.is_binary);
}

size_t CLKernelLibrary::max_local_workgroup_size(const cl::Kernel &kernel) const
{
    return _compile_context.max_local_workgroup_size(kernel);
}

cl::NDRange CLKernelLibrary::default_ndrange() const
{
    return _compile_context.default_ndrange();
}

std::string CLKernelLibrary::get_device_version()
{
    return _compile_context.get_device_version();
}

cl_uint CLKernelLibrary::get_num_compute_units()
{
    return _compile_context.get_num_compute_units();
}

CLCompileContext &CLKernelLibrary::get_compile_context()
{
    return _compile_context;
}
} // namespace arm_compute
