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

#ifndef CKW_SRC_CL_CLKERNELWRITER_H
#define CKW_SRC_CL_CLKERNELWRITER_H

#include "ckw/KernelWriter.h"

#include <utility>

namespace ckw
{

/** OpenCL kernel writer. */
class CLKernelWriter : public KernelWriter
{
public:
    // =============================================================================================
    // Construtors and destructor
    // =============================================================================================

    /** Initialize a new instance of @ref CLKernelWriter class. */
    CLKernelWriter();

    /** Destructor */
    ~CLKernelWriter();

    // =============================================================================================
    // Misc
    // =============================================================================================

    void comment(const std::string &text) override;

    // =============================================================================================
    // Code generation
    // =============================================================================================

    std::unique_ptr<Kernel> emit_kernel(const std::string &name) override;

    /** Declare a tile given name and tile information
     *
     * Similar to @ref KernelWriter::declare_tile()
    */
    TileOperand &declare_tile(const ::std::string &name, const TileInfo &tile_info) override;


protected:
    /** Append the specified code to the kernel body source code. */
    template <typename T, typename... TArgs>
    void append_code(T &&code, TArgs &&...args)
    {
        append_code(std::forward<T>(code));
        append_code(std::forward<TArgs>(args)...);
    }

    /** Append the specified code to the kernel body source code. */
    template <typename T>
    void append_code(T &&code)
    {
        _body_source_code += std::forward<T>(code);
    }

    /** Get the current kernel body source code. */
    const std::string &body_source_code() const;

    /** Add a tile operand to the kernel and return it */
    TileOperand &add_operand(const std::string &code, const TileInfo &tile_info) override;

private:
    /** This string contains the kernel body source code, not the full CL source code.
     * The full source code will only be generated when the user calls @ref KernelWriter::emit_kernel.
     *
     * In order to add code to this, use @ref CLKernelWriter::append_code.
     * Do not attempt to concatenate and alter this string directly.
     */
    std::string _body_source_code{};
};

} // namespace ckw

#endif // CKW_SRC_CL_CLKERNELWRITER_H
