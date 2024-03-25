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
#ifndef ARM_COMPUTE_CLKERNELLIBRARY_H
#define ARM_COMPUTE_CLKERNELLIBRARY_H

#include "arm_compute/core/CL/CLCompileContext.h"
#include "arm_compute/core/CL/OpenCL.h"

#include <map>
#include <set>
#include <string>
#include <utility>

namespace arm_compute
{
/** CLKernelLibrary class */
class CLKernelLibrary final
{
private:
    /** Default Constructor. */
    CLKernelLibrary();
    /** Prevent instances of this class from being copied */
    CLKernelLibrary(const CLKernelLibrary &) = delete;
    /** Prevent instances of this class from being copied */
    const CLKernelLibrary &operator=(const CLKernelLibrary &) = delete;

public:
    /** Access the KernelLibrary singleton.
     * This method has been deprecated and will be removed in future releases
     * @return The KernelLibrary instance.
     */
    static CLKernelLibrary &get();
    /** Initialises the kernel library.
     *
     * @param[in] kernel_path Path of the directory from which kernel sources are loaded.
     * @param[in] context     CL context used to create programs.
     * @param[in] device      CL device for which the programs are created.
     */
    void init(std::string kernel_path, cl::Context context, cl::Device device);
    /** Sets the path that the kernels reside in.
     *
     * @param[in] kernel_path Path of the kernel.
     */
    void set_kernel_path(const std::string &kernel_path);
    /** Gets the path that the kernels reside in.
     */
    std::string get_kernel_path();
    /** Gets the source of the selected program.
     *
     * @param[in] program_name Program name.
     *
     * @return A pair with the source (false) or the binary (true), of the selected program.
     */
    std::pair<std::string, bool> get_program(const std::string &program_name) const;

    /** Accessor for the associated CL context.
     *
     * @return A CL context.
     */
    cl::Context &context();

    /** Gets the CL device for which the programs are created. */
    const cl::Device &get_device();

    /** Sets the CL device for which the programs are created.
     *
     * @param[in] device A CL device.
     */
    void set_device(cl::Device device);

    /** Return the device version
     *
     * @return The content of CL_DEVICE_VERSION
     */
    std::string get_device_version();
    /** Return the maximum number of compute units in the device
     *
     * @return The content of CL_DEVICE_MAX_COMPUTE_UNITS
     */
    cl_uint get_num_compute_units();
    /** Creates a kernel from the kernel library.
     *
     * @param[in] kernel_name       Kernel name.
     * @param[in] build_options_set Kernel build options as a set.
     *
     * @return The created kernel.
     */
    Kernel create_kernel(const std::string &kernel_name, const std::set<std::string> &build_options_set = {}) const;
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
    void clear_programs_cache();

    /** Access the cache of built OpenCL programs */
    const std::map<std::string, cl::Program> &get_built_programs() const;

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

    /** Returns the program name given a kernel name
     *
     * @return Program name
     */
    std::string get_program_name(const std::string &kernel_name) const;

    /* Returns true if the workgroup batch size modifier parameter is supported on the cl device
    *
    * @return true if the workgroup batch size modifier parameter is supported, false otherwise
    */
    bool is_wbsm_supported();

    /** Sets the CL context used to create programs.
     *
     * @note Setting the context also resets the device to the
     *       first one available in the new context.
     *
     * @param[in] context A CL context.
     */
    void set_context(cl::Context context);

    /** Gets the compile context used
     *
     * @return The used compile context
     */
    CLCompileContext &get_compile_context();

private:
    CLCompileContext _compile_context; /**< Compile Context. */
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_CLKERNELLIBRARY_H */
