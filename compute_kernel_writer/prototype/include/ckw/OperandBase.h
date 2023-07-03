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

#ifndef CKW_PROTOTYPE_INCLUDE_CKW_OPERANDBASE_H
#define CKW_PROTOTYPE_INCLUDE_CKW_OPERANDBASE_H

#include "ckw/Types.h"
#include <string>

namespace ckw
{
namespace prototype
{
class IGpuKernelWriter;

class Operand;
} // namespace prototype

/** The base class for all operands. */
class OperandBase
{
public:
    /** Constructor
     *
     * @param[in] name The name of the operand.
     */
    explicit OperandBase(const ::std::string &name);

    /** Destructor */
    virtual ~OperandBase();

    /** (Internal use only) Create the implementation operand.
     *
     * @param[in] writer The implementation kernel writer.
     */
    virtual prototype::Operand create_impl_operand(prototype::IGpuKernelWriter *writer) const = 0;

    /** Get the name of the operand. */
    const ::std::string &name() const;

    /** Set the name of the operand. */
    OperandBase &name(const ::std::string &name);

    /** Get the data type of the operand. */
    virtual DataType data_type() const = 0;

    /** Get whether the operand is compile-time constant. */
    virtual bool is_constant() const = 0;

private:
    ::std::string _name;
};

} // namespace ckw

#endif // CKW_PROTOTYPE_INCLUDE_CKW_OPERANDBASE_H
