/*
 * Copyright (c) 2023 Arm Limited.
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

#ifndef CKW_INCLUDE_CKW_KERNELWRITER_H
#define CKW_INCLUDE_CKW_KERNELWRITER_H

#include "ckw/types/TargetArchitecture.h"
#include "ckw/types/TargetLanguage.h"
#include <memory>
#include <string>

namespace ckw
{

class Kernel;

/** A kernel writer.
 *
 * This class is used to construct a new kernel by defining arguments, declaring variable and writing code.
 *
 * Use @ref KernelWriter::create_instance method to create the kernel writer for the specific target architecture and language.
 *
 * After having finished constructing the kernel, call @ref KernelWriter::emit_kernel to get the kernel object.
 */
class KernelWriter
{
public:
    // =============================================================================================
    // Construtors and destructor
    // =============================================================================================

    /** Initialize a new instance of @ref KernelWriter class for the specific architecture and language.
     *
     * Supported target architectures and languages:
     *
     * Architecture                  | Languages                    |
     * ------------------------------|------------------------------|
     * GpuArmMaliValhall             | OpenCL                       |
     *
     * @param[in] architecture The architecture on which the kernel is executed.
     * @param[in] language     The language to write the kernel.
     */
    static std::unique_ptr<KernelWriter> create_instance(TargetArchitecture architecture, TargetLanguage language);

    /** Destructor */
    virtual ~KernelWriter();

    // =============================================================================================
    // Misc
    // =============================================================================================

    /** Write the line comment in debug build.
     * This function does not take effect on release build.
     *
     * The comment must only contain one line (i.e. no newline character is allowed).
     *
     * @param[in] text The comment to be written.
     */
    virtual void comment(const std::string &text) = 0;

    // =============================================================================================
    // Code generation
    // =============================================================================================

    /** Emit the kernel object.
     *
     * @param[in] name The name of the kernel object to be generated.
     */
    virtual std::unique_ptr<Kernel> emit_kernel(const std::string &name) = 0;
};

} // namespace ckw

#endif // CKW_INCLUDE_CKW_KERNELWRITER_H
