/*
 * Copyright (c) 2020-2022 Arm Limited.
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
#ifndef ARM_COMPUTE_CLCOMPILECONTEXT_H
#define ARM_COMPUTE_CLCOMPILECONTEXT_H

#include "arm_compute/core/CL/CLDevice.h"
#include "arm_compute/core/CL/OpenCL.h"

#include <map>
#include <set>
#include <string>
#include <utility>

namespace arm_compute
{
/** Build options */
class CLBuildOptions final
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

    bool operator==(const CLBuildOptions &other) const;

private:
    StringSet _build_opts; /**< Build options set */
};

/** Program class */
class Program final
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
    /** Returns program binary data.
     *
     * @return Program's binary data.
     */
    const std::vector<unsigned char> &binary() const
    {
        return _binary;
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
class Kernel final
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

/** CLCompileContext class */
class CLCompileContext final
{
    using StringSet = std::set<std::string>;

public:
    /** Constructor */
    CLCompileContext();
    /** Constructor
     *
     * @param[in] context A CL context.
     * @param[in] device  A CL device.
     * */
    CLCompileContext(cl::Context context, const cl::Device &device);

    /** Accessor for the associated CL context.
     *
     * @return A CL context.
     */
    cl::Context &context();

    /** Sets the CL context used to create programs.
     *
     * @note Setting the context also resets the device to the
     *       first one available in the new context.
     *
     * @param[in] context A CL context.
     */
    void set_context(cl::Context context);

    /** Gets the CL device for which the programs are created. */
    const cl::Device &get_device() const;

    /** Sets the CL device for which the programs are created.
     *
     * @param[in] device A CL device.
     */
    void set_device(cl::Device device);

    /** Creates an OpenCL kernel.
     *
     * @param[in] kernel_name       Kernel name.
     * @param[in] program_name      Program name.
     * @param[in] program_source    Program source.
     * @param[in] kernel_path       CL kernel path.
     * @param[in] build_options_set Kernel build options as a set.
     * @param[in] is_binary         Flag to indicate if the program source is binary.
     *
     * @return The created kernel.
     */
    Kernel create_kernel(const std::string &kernel_name, const std::string &program_name, const std::string &program_source,
                         const std::string &kernel_path, const StringSet &build_options_set, bool is_binary) const;

    /** Clear the library's cache of binary programs
     */
    void clear_programs_cache();

    /** Access the cache of built OpenCL programs */
    const std::map<std::string, cl::Program> &get_built_programs() const;

    /** Add a new built program to the cache
     *
     * @param[in] built_program_name Name of the program
     * @param[in] program            Built program to add to the cache
     */
    void add_built_program(const std::string &built_program_name, const cl::Program &program) const;

    /** Returns true if FP16 is supported by the CL device
     *
     * @return true if the CL device supports FP16
     */
    bool fp16_supported() const;

    /** Return the maximum number of compute units in the device
     *
     * @return The content of CL_DEVICE_MAX_COMPUTE_UNITS
     */
    cl_uint get_num_compute_units() const;
    /** Find the maximum number of local work items in a workgroup can be supported for the kernel.
     *
     */
    size_t max_local_workgroup_size(const cl::Kernel &kernel) const;
    /** Return the default NDRange for the device.
     *
     */
    cl::NDRange default_ndrange() const;
    /** Return the device version
     *
     * @return The content of CL_DEVICE_VERSION
     */
    std::string get_device_version() const;

    /** Returns true if int64_base_atomics extension is supported by the CL device
     *
     * @return true if the CL device supports int64_base_atomics extension
     */
    bool int64_base_atomics_supported() const;

    /* Returns true if the workgroup batch size modifier parameter is supported on the cl device
    *
    * @return true if the workgroup batch size modifier parameter is supported, false otherwise
    */
    bool is_wbsm_supported() const;

    /** Return the DDK version. If the DDK version cannot be detected, return -1.
     *
     * @return The DDK version.
     */
    int32_t get_ddk_version() const;

    /** Return the Gpu target of the associated device
     *
     * @return GPUTarget
     */
    GPUTarget get_gpu_target() const;

private:
    /** Load program and its dependencies.
     *
     * @param[in] program_name   Name of the program to load.
     * @param[in] program_source Source of the program.
     * @param[in] is_binary      Flag to indicate if the program source is binary.
     */
    const Program &load_program(const std::string &program_name, const std::string &program_source, bool is_binary) const;

    /** Generates the build options given a string of user defined ones
     *
     * @param[in] build_options User defined build options
     * @param[in] kernel_path   Path of the CL kernels
     *
     * @return Generated build options
     */
    std::string generate_build_options(const StringSet &build_options, const std::string &kernel_path) const;

    /** Concatenates contents of a set into a single string.
     *
     * @param[in] s           Input set to concatenate.
     * @param[in] kernel_path Path of the CL kernels
     *
     * @return Concatenated string.
     */
    std::string stringify_set(const StringSet &s, const std::string &kernel_path) const;

    cl::Context _context;                                             /**< Underlying CL context. */
    CLDevice    _device;                                              /**< Underlying CL device. */
    mutable std::map<std::string, const Program> _programs_map;       /**< Map with all already loaded program data. */
    mutable std::map<std::string, cl::Program>   _built_programs_map; /**< Map with all already built program data. */
    bool _is_wbsm_supported;                                          /**< Support of worksize batch size modifier support boolean*/
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_CLCOMPILECONTEXT_H */
