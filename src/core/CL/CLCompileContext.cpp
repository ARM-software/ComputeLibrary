/*
 * Copyright (c) 2020-2023 Arm Limited.
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
#include "arm_compute/core/CL/CLCompileContext.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Utils.h"

#include "support/StringSupport.h"

#include <regex>

namespace arm_compute
{
CLBuildOptions::CLBuildOptions() : _build_opts()
{
}

void CLBuildOptions::add_option(std::string option)
{
    _build_opts.emplace(std::move(option));
}

void CLBuildOptions::add_option_if(bool cond, std::string option)
{
    if (cond)
    {
        add_option(std::move(option));
    }
}

void CLBuildOptions::add_option_if_else(bool cond, std::string option_true, std::string option_false)
{
    (cond) ? add_option(std::move(option_true)) : add_option(std::move(option_false));
}

void CLBuildOptions::add_options(const StringSet &options)
{
    _build_opts.insert(options.begin(), options.end());
}

void CLBuildOptions::add_options_if(bool cond, const StringSet &options)
{
    if (cond)
    {
        add_options(options);
    }
}

const CLBuildOptions::StringSet &CLBuildOptions::options() const
{
    return _build_opts;
}

bool CLBuildOptions::operator==(const CLBuildOptions &other) const
{
    return _build_opts == other._build_opts;
}

Program::Program() : _context(), _device(), _is_binary(false), _name(), _source(), _binary()
{
}

Program::Program(cl::Context context, std::string name, std::string source)
    : _context(std::move(context)),
      _device(),
      _is_binary(false),
      _name(std::move(name)),
      _source(std::move(source)),
      _binary()
{
}

Program::Program(cl::Context context, cl::Device device, std::string name, std::vector<unsigned char> binary)
    : _context(std::move(context)),
      _device(std::move(device)),
      _is_binary(true),
      _name(std::move(name)),
      _source(),
      _binary(std::move(binary))
{
}

Program::operator cl::Program() const
{
    if (_is_binary)
    {
        return cl::Program(_context, {_device}, {_binary});
    }
    else
    {
        return cl::Program(_context, _source, false);
    }
}

bool Program::build(const cl::Program &program, const std::string &build_options)
{
    try
    {
        return program.build(build_options.c_str()) == CL_SUCCESS;
    }
    catch (const cl::Error &e)
    {
        cl_int     err        = CL_SUCCESS;
        const auto build_info = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(&err);

        for (auto &pair : build_info)
        {
            std::cerr << pair.second << std::endl;
        }

        return false;
    }
}

cl::Program Program::build(const std::string &build_options) const
{
    cl::Program cl_program = static_cast<cl::Program>(*this);
    build(cl_program, build_options);
    return cl_program;
}

Kernel::Kernel() : _name(), _kernel()
{
}

Kernel::Kernel(std::string name, const cl::Program &program)
    : _name(std::move(name)), _kernel(cl::Kernel(program, _name.c_str()))
{
}
CLCompileContext::CLCompileContext()
    : _context(), _device(), _programs_map(), _built_programs_map(), _is_wbsm_supported()
{
}

CLCompileContext::CLCompileContext(cl::Context context, const cl::Device &device)
    : _context(), _device(), _programs_map(), _built_programs_map(), _is_wbsm_supported()
{
    _context           = std::move(context);
    _device            = CLDevice(device);
    _is_wbsm_supported = get_wbsm_support_info(device);
}

Kernel CLCompileContext::create_kernel(const std::string &kernel_name,
                                       const std::string &program_name,
                                       const std::string &program_source,
                                       const std::string &kernel_path,
                                       const StringSet   &build_options_set,
                                       bool               is_binary) const
{
    const std::string build_options      = generate_build_options(build_options_set, kernel_path);
    const std::string built_program_name = program_name + "_" + build_options;
    auto              built_program_it   = _built_programs_map.find(built_program_name);
    cl::Program       cl_program;

    if (_built_programs_map.end() != built_program_it)
    {
        // If program has been built, retrieve to create kernel from it
        cl_program = built_program_it->second;
    }
    else
    {
        Program program = load_program(program_name, program_source, is_binary);

        // Build program
        cl_program = program.build(build_options);

        // Add built program to internal map
        _built_programs_map.emplace(built_program_name, cl_program);
    }

    // Create and return kernel
    return Kernel(kernel_name, cl_program);
}

const Program &
CLCompileContext::load_program(const std::string &program_name, const std::string &program_source, bool is_binary) const
{
    const auto program_it = _programs_map.find(program_name);

    if (program_it != _programs_map.end())
    {
        return program_it->second;
    }

    Program program;

#ifdef EMBEDDED_KERNELS
    ARM_COMPUTE_UNUSED(is_binary);
    program = Program(_context, program_name, program_source);
#else  /* EMBEDDED_KERNELS */
    if (is_binary)
    {
        program = Program(_context, _device.cl_device(), program_name,
                          std::vector<unsigned char>(program_source.begin(), program_source.end()));
    }
    else
    {
        program = Program(_context, program_name, program_source);
    }
#endif /* EMBEDDED_KERNELS */

    // Insert program to program map
    const auto new_program = _programs_map.emplace(program_name, std::move(program));

    return new_program.first->second;
}

void CLCompileContext::set_context(cl::Context context)
{
    _context = std::move(context);
    if (_context.get() != nullptr)
    {
        const auto cl_devices = _context.getInfo<CL_CONTEXT_DEVICES>();

        if (!cl_devices.empty())
        {
            _device = CLDevice(cl_devices[0]);
        }
    }
}

std::string CLCompileContext::generate_build_options(const StringSet   &build_options_set,
                                                     const std::string &kernel_path) const
{
    std::string concat_str;
    bool        ext_supported = false;
    std::string ext_buildopts;

#if defined(ARM_COMPUTE_DEBUG_ENABLED)
    // Enable debug properties in CL kernels
    concat_str += " -DARM_COMPUTE_DEBUG_ENABLED";
#endif // defined(ARM_COMPUTE_DEBUG_ENABLED)

    GPUTarget gpu_arch = get_arch_from_target(_device.target());
    concat_str +=
        " -DGPU_ARCH=" + support::cpp11::to_string(static_cast<std::underlying_type<GPUTarget>::type>(gpu_arch));

    if (_device.supported("cl_khr_fp16"))
    {
        concat_str += " -DARM_COMPUTE_OPENCL_FP16_ENABLED=1 ";
    }

    if (_device.supported("cl_arm_integer_dot_product_int8") || _device.supported("cl_khr_integer_dot_product"))
    {
        concat_str += " -DARM_COMPUTE_OPENCL_DOT8_ENABLED=1 ";
    }

    if (_device.supported("cl_arm_integer_dot_product_accumulate_int8"))
    {
        concat_str += " -DARM_COMPUTE_OPENCL_DOT8_ACC_ENABLED=1 ";
    }

    std::tie(ext_supported, ext_buildopts) = _device.is_non_uniform_workgroup_supported();

    if (ext_supported)
    {
        concat_str += ext_buildopts;
    }
    else
    {
        ARM_COMPUTE_ERROR("Non uniform workgroup size is not supported!!");
    }

    if (gpu_arch != GPUTarget::UNKNOWN && gpu_arch != GPUTarget::MIDGARD && get_ddk_version() >= 11)
    {
        concat_str += " -DUNROLL_WITH_PRAGMA ";
    }

    std::string build_options = stringify_set(build_options_set, kernel_path) + concat_str;

    return build_options;
}

bool CLCompileContext::fp16_supported() const
{
    return _device.supported("cl_khr_fp16");
}

std::string CLCompileContext::stringify_set(const StringSet &s, const std::string &kernel_path) const
{
    std::string concat_set;
#ifndef EMBEDDED_KERNELS
    concat_set += "-I" + kernel_path + " ";
#else  /* EMBEDDED_KERNELS */
    ARM_COMPUTE_UNUSED(kernel_path);
#endif /* EMBEDDED_KERNELS */

    // Concatenate set
    for (const auto &el : s)
    {
        concat_set += " " + el;
    }

    return concat_set;
}

void CLCompileContext::add_built_program(const std::string &built_program_name, const cl::Program &program) const
{
    _built_programs_map.emplace(built_program_name, program);
}

void CLCompileContext::clear_programs_cache()
{
    _programs_map.clear();
    _built_programs_map.clear();
}

const std::map<std::string, cl::Program> &CLCompileContext::get_built_programs() const
{
    return _built_programs_map;
}

cl::Context &CLCompileContext::context()
{
    return _context;
}

const cl::Device &CLCompileContext::get_device() const
{
    return _device.cl_device();
}

void CLCompileContext::set_device(cl::Device device)
{
    _is_wbsm_supported = get_wbsm_support_info(device);
    _device            = std::move(device);
}

cl::NDRange CLCompileContext::default_ndrange() const
{
    GPUTarget   _target = get_target_from_device(_device.cl_device());
    cl::NDRange default_range;

    switch (_target)
    {
        case GPUTarget::MIDGARD:
        case GPUTarget::T600:
        case GPUTarget::T700:
        case GPUTarget::T800:
            default_range = cl::NDRange(128u, 1);
            break;
        default:
            default_range = cl::NullRange;
    }

    return default_range;
}

bool CLCompileContext::int64_base_atomics_supported() const
{
    return _device.supported("cl_khr_int64_base_atomics");
}

bool CLCompileContext::is_wbsm_supported() const
{
    return _is_wbsm_supported;
}

size_t CLCompileContext::max_local_workgroup_size(const cl::Kernel &kernel) const
{
    size_t result;

    size_t err = kernel.getWorkGroupInfo(_device.cl_device(), CL_KERNEL_WORK_GROUP_SIZE, &result);
    ARM_COMPUTE_ERROR_ON_MSG(err != 0,
                             "clGetKernelWorkGroupInfo failed to return the maximum workgroup size for the kernel");
    ARM_COMPUTE_UNUSED(err);

    return result;
}

std::string CLCompileContext::get_device_version() const
{
    return _device.device_version();
}

cl_uint CLCompileContext::get_num_compute_units() const
{
    return _device.compute_units();
}

int32_t CLCompileContext::get_ddk_version() const
{
    const std::string device_version = _device.device_version();
    const std::regex  ddk_regex("r([0-9]*)p[0-9]");
    std::smatch       ddk_match;

    if (std::regex_search(device_version, ddk_match, ddk_regex))
    {
        return std::stoi(ddk_match[1]);
    }

    return -1;
}
GPUTarget CLCompileContext::get_gpu_target() const
{
    return _device.target();
}
} // namespace arm_compute
