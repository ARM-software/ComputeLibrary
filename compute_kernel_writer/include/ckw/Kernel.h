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

#ifndef CKW_INCLUDE_CKW_KERNEL_H
#define CKW_INCLUDE_CKW_KERNEL_H

#include <string>

namespace ckw
{

// Forward Declerations
class TileInfo;
class ITileOperand;

enum class TargetLanguage;

/** The kernel that has been emitted by the kernel writer.
 *
 * It contains all the necessary information to compile and execute the kernel.
 */
class Kernel
{
public:
    virtual ~Kernel();

    /** Initialize a new instance of @ref Kernel class with all emitted kernel information.
     *
     * @param[in] language    The target language of the kernel.
     * @param[in] source_code The source code of the kernel.
     */
    Kernel(TargetLanguage language, const std::string &source_code);

    /** Get the target language. */
    TargetLanguage target_language() const;

    /** Get the source code. */
    const std::string &source_code() const;

    /** Add a tile operand */
    virtual ITileOperand &add_operand(const std::string &name, const TileInfo &tile_info) = 0;

private:
    TargetLanguage _language;
    std::string    _source_code;
};

} // namespace ckw

#endif // CKW_INCLUDE_CKW_KERNEL_H
