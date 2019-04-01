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
#ifndef __ARM_COMPUTE_CLKERNELLIBRARY_H__
#define __ARM_COMPUTE_CLKERNELLIBRARY_H__

#include "arm_compute/core/CL/OpenCL.h"

#include <map>
#include <set>
#include <string>
#include <utility>

namespace arm_compute
{
/** Build options */
class CLBuildOptions
{
    using StringSet = std::set<std::string>;

public:
    /** Default constructor. */
    CLBuildOptions();
    /** Adds option to the existing build option list
     *
     * @param[in] option Option to add
     */
    void add_option(std::string option);
    /** Adds option if a given condition is true;
     *
     * @param[in] cond   Condition to check
     * @param[in] option Option to add if condition is true
     */
    void add_option_if(bool cond, std::string option);
    /** Adds first option if condition is true else the second one
     *
     * @param[in] cond         Condition to check
     * @param[in] option_true  Option to add if condition is true
     * @param[in] option_false Option to add if condition is false
     */
    void add_option_if_else(bool cond, std::string option_true, std::string option_false);
    /** Appends given build options to the current's objects options.
     *
     * @param[in] options Build options to append
     */
    void add_options(const StringSet &options);
    /** Appends given build options to the current's objects options if a given condition is true.
     *
     * @param[in] cond    Condition to check
     * @param[in] options Option to add if condition is true
     */
    void add_options_if(bool cond, const StringSet &options);
    /** Gets the current options list set
     *
     * @return Build options set
     */
    const StringSet &options() const;

private:
    StringSet _build_opts; /**< Build options set */
};
/** Program class */
class Program
{
public:
    /** Default constructor. */
    Program();
    /** Construct program from source file.
     *
     * @param[in] context CL context used to create the program.
     * @param[in] name    Program name.
     * @param[in] source  Program source.
     */
    Program(cl::Context context, std::string name, std::string source);
    /** Construct program from binary file.
     *
     * @param[in] context CL context used to create the program.
     * @param[in] device  CL device for which the programs are created.
     * @param[in] name    Program name.
     * @param[in] binary  Program binary.
     */
    Program(cl::Context context, cl::Device device, std::string name, std::vector<unsigned char> binary);
    /** Default Copy Constructor. */
    Program(const Program &) = default;
    /** Default Move Constructor. */
    Program(Program &&) = default;
    /** Default copy assignment operator */
    Program &operator=(const Program &) = default;
    /** Default move assignment operator */
    Program &operator=(Program &&) = default;
    /** Returns program name.
     *
     * @return Program's name.
     */
    std::string name() const
    {
        return _name;
    }
    /** User-defined conversion to the underlying CL program.
     *
     * @return The CL program object.
     */
    explicit operator cl::Program() const;
    /** Build the given CL program.
     *
     * @param[in] program       The CL program to build.
     * @param[in] build_options Options to build the CL program.
     *
     * @return True if the CL program builds successfully.
     */
    static bool build(const cl::Program &program, const std::string &build_options = "");
    /** Build the underlying CL program.
     *
     * @param[in] build_options Options used to build the CL program.
     *
     * @return A reference to itself.
     */
    cl::Program build(const std::string &build_options = "") const;

private:
    cl::Context                _context;   /**< Underlying CL context. */
    cl::Device                 _device;    /**< CL device for which the programs are created. */
    bool                       _is_binary; /**< Create program from binary? */
    std::string                _name;      /**< Program name. */
    std::string                _source;    /**< Source code for the program. */
    std::vector<unsigned char> _binary;    /**< Binary from which to create the program. */
};

/** Kernel class */
class Kernel
{
public:
    /** Default Constructor. */
    Kernel();
    /** Default Copy Constructor. */
    Kernel(const Kernel &) = default;
    /** Default Move Constructor. */
    Kernel(Kernel &&) = default;
    /** Default copy assignment operator */
    Kernel &operator=(const Kernel &) = default;
    /** Default move assignment operator */
    Kernel &operator=(Kernel &&) = default;
    /** Constructor.
     *
     * @param[in] name    Kernel name.
     * @param[in] program Built program.
     */
    Kernel(std::string name, const cl::Program &program);
    /** Returns kernel name.
     *
     * @return Kernel's name.
     */
    std::string name() const
    {
        return _name;
    }
    /** Returns OpenCL kernel.
     *
     * @return OpenCL Kernel.
     */
    explicit operator cl::Kernel() const
    {
        return _kernel;
    }

private:
    std::string _name;   /**< Kernel name */
    cl::Kernel  _kernel; /**< OpenCL Kernel */
};

/** CLKernelLibrary class */
class CLKernelLibrary
{
    using StringSet = std::set<std::string>;

private:
    /** Default Constructor. */
    CLKernelLibrary();

public:
    /** Prevent instances of this class from being copied */
    CLKernelLibrary(const CLKernelLibrary &) = delete;
    /** Prevent instances of this class from being copied */
    const CLKernelLibrary &operator=(const CLKernelLibrary &) = delete;
    /** Access the KernelLibrary singleton.
     * @return The KernelLibrary instance.
     */
    static CLKernelLibrary &get();
    /** Initialises the kernel library.
     *
     * @param[in] kernel_path Path of the directory from which kernel sources are loaded.
     * @param[in] context     CL context used to create programs.
     * @param[in] device      CL device for which the programs are created.
     */
    void init(std::string kernel_path, cl::Context context, cl::Device device)
    {
        _kernel_path = std::move(kernel_path);
        _context     = std::move(context);
        _device      = std::move(device);
    }
    /** Sets the path that the kernels reside in.
     *
     * @param[in] kernel_path Path of the kernel.
     */
    void set_kernel_path(const std::string &kernel_path)
    {
        _kernel_path = kernel_path;
    };
    /** Gets the path that the kernels reside in.
     */
    std::string get_kernel_path()
    {
        return _kernel_path;
    };
    /** Gets the source of the selected program.
     *
     * @param[in] program_name Program name.
     *
     * @return Source of the selected program.
     */
    std::string get_program_source(const std::string &program_name);
    /** Sets the CL context used to create programs.
     *
     * @note Setting the context also resets the device to the
     *       first one available in the new context.
     *
     * @param[in] context A CL context.
     */
    void set_context(cl::Context context)
    {
        _context = std::move(context);
        if(_context.get() == nullptr)
        {
            _device = cl::Device();
        }
        else
        {
            const auto cl_devices = _context.getInfo<CL_CONTEXT_DEVICES>();

            if(cl_devices.empty())
            {
                _device = cl::Device();
            }
            else
            {
                _device = cl_devices[0];
            }
        }
    }

    /** Accessor for the associated CL context.
     *
     * @return A CL context.
     */
    cl::Context &context()
    {
        return _context;
    }

    /** Gets the CL device for which the programs are created. */
    cl::Device &get_device()
    {
        return _device;
    }

    /** Sets the CL device for which the programs are created.
     *
     * @param[in] device A CL device.
     */
    void set_device(cl::Device device)
    {
        _device = std::move(device);
    }

    /** Return the device version
     *
     * @return The content of CL_DEVICE_VERSION
     */
    std::string get_device_version();
    /** Creates a kernel from the kernel library.
     *
     * @param[in] kernel_name       Kernel name.
     * @param[in] build_options_set Kernel build options as a set.
     *
     * @return The created kernel.
     */
    Kernel create_kernel(const std::string &kernel_name, const StringSet &build_options_set = {}) const;
    /** Find the maximum number of local work items in a workgroup can be supported for the kernel.
     *
     */
    size_t max_local_workgroup_size(const cl::Kernel &kernel) const;
    /** Return the default NDRange for the device.
     *
     */
    cl::NDRange default_ndrange() const;

    /** Clear the library's cache of binary programs
     */
    void clear_programs_cache()
    {
        _programs_map.clear();
        _built_programs_map.clear();
    }

    /** Access the cache of built OpenCL programs */
    const std::map<std::string, cl::Program> &get_built_programs() const
    {
        return _built_programs_map;
    }

    /** Add a new built program to the cache
     *
     * @param[in] built_program_name Name of the program
     * @param[in] program            Built program to add to the cache
     */
    void add_built_program(const std::string &built_program_name, const cl::Program &program);

    /** Returns true if FP16 is supported by the CL device
     *
     * @return true if the CL device supports FP16
     */
    bool fp16_supported() const;

    /** Returns true if int64_base_atomics extension is supported by the CL device
     *
     * @return true if the CL device supports int64_base_atomics extension
     */
    bool int64_base_atomics_supported() const;

private:
    /** Load program and its dependencies.
     *
     * @param[in] program_name Name of the program to load.
     */
    const Program &load_program(const std::string &program_name) const;
    /** Concatenates contents of a set into a single string.
     *
     * @param[in] s Input set to concatenate.
     *
     * @return Concatenated string.
     */
    std::string stringify_set(const StringSet &s) const;

    cl::Context _context;                                                /**< Underlying CL context. */
    cl::Device  _device;                                                 /**< Underlying CL device. */
    std::string _kernel_path;                                            /**< Path to the kernels folder. */
    mutable std::map<std::string, const Program>    _programs_map;       /**< Map with all already loaded program data. */
    mutable std::map<std::string, cl::Program>      _built_programs_map; /**< Map with all already built program data. */
    static const std::map<std::string, std::string> _kernel_program_map; /**< Map that associates kernel names with programs. */
    static const std::map<std::string, std::string> _program_source_map; /**< Contains sources for all programs.
                                                                              Used for compile-time kernel inclusion. >*/
};
}
#endif /* __ARM_COMPUTE_CLKERNELLIBRARY_H__ */
