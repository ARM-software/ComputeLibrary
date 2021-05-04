/*
 * Copyright (c) 2021 Arm Limited.
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
#ifndef ARM_COMPUTE_CL_KERNEL_LIBRARY_H
#define ARM_COMPUTE_CL_KERNEL_LIBRARY_H

#include <map>
#include <string>
#include <tuple>

namespace arm_compute
{
namespace opencl
{
/** ClKernelLibrary contains all the OpenCL kernels that are used throughout the library
 *
 * @note Kernel library is a singleton to reduce memory requirements
 * @note Sole responsibility is just to provide access to the kernel string,
 *       does not perform any compilation and relevant tasks
 */
class ClKernelLibrary final
{
private:
    /** Default Constructor */
    ClKernelLibrary() = default;
    /** Prevent instances of this class from being copied */
    ClKernelLibrary(const ClKernelLibrary &) = delete;
    /** Prevent instances of this class from being copied */
    const ClKernelLibrary &operator=(const ClKernelLibrary &) = delete;

public:
    /** Structure to encapsulte program related information */
    struct ClProgramInfo
    {
        std::string program{};          /**< Program raw string */
        bool        is_binary{ false }; /**< Flag that indicates if is in binary format */
    };

public:
    /** Access the KernelLibrary singleton
     *
     * @return The KernelLibrary instance
     */
    static ClKernelLibrary &get();
    /** Sets the path that the kernels reside in
     *
     * @param[in] kernel_path Path of the kernel
     */
    void set_kernel_path(std::string kernel_path);
    /** Gets the path that the kernels reside in
     */
    const std::string &kernel_path() const;
    /** Gets the source of the selected program
     *
     * @param[in] program_name Program name
     *
     * @return A pair with the source (false) or the binary (true), of the selected program
     */
    ClProgramInfo program(const std::string &program_name) const;
    /** Returns the program name given a kernel name
     *
     * @return Program name
     */
    std::string program_name(const std::string &kernel_name) const;

private:
    std::string _kernel_path{};                                                 /**< Path to the kernels folder. */
    mutable std::map<std::string, std::string>      _decompressed_source_map{}; /**< Map holding the decompressed files when compression is used */
    static const std::map<std::string, std::string> _kernel_program_map;        /**< Map that associates kernel names with programs. */
    static const std::map<std::string, std::string> _program_source_map;        /**< Contains sources for all programs.
                                                                                     Used for compile-time kernel inclusion. >*/
};
} // namespace opencl
} // namespace arm_compute
#endif /* ARM_COMPUTE_CL_KERNEL_LIBRARY_H */
