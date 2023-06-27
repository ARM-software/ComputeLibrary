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

#ifndef CKW_PROTOTYPE_INCLUDE_CKW_KERNEL_H
#define CKW_PROTOTYPE_INCLUDE_CKW_KERNEL_H

#include "ckw/KernelArgument.h"
#include "ckw/OperandBase.h"
#include "ckw/types/GpuTargetLanguage.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace ckw
{

class TileOperand;

namespace prototype
{
class GpuKernelWriterDataHolder;
} // namespace prototype

/** The target for kernel writer to write into. */
class Kernel
{
public:
    /** Constructor
     *
     * @param[in] name     The name of the kernel function.
     * @param[in] language The programming language to write the kernel.
     */
    Kernel(const char *name, GpuTargetLanguage language);

    /** Destructor */
    ~Kernel();

    /** Get the name of the kernel function. */
    const std::string &name() const;

    /** Get the list of kernel arguments. */
    ::std::vector<KernelArgument> arguments() const;

    /** (Internal use only) Register the tile operand.
     *
     * @param operand The tile operand to be registered.
     */
    TileOperand &register_operand(::std::unique_ptr<TileOperand> operand);

    /** (Internal use only) Register the tensor operand.
     *
     * @param operand The tensor operand to be registered.
     */
    TensorOperand &register_operand(::std::unique_ptr<TensorOperand> operand);

    /** (Internal use only) Get the implementation data. */
    prototype::GpuKernelWriterDataHolder *impl();

private:
    ::std::string                                             _name;
    ::std::unique_ptr<prototype::GpuKernelWriterDataHolder>   _kernel;
    ::std::map<::std::string, ::std::unique_ptr<OperandBase>> _operands;
    ::std::map<int32_t, TensorOperand *>                      _tensor_id_operands;
};

} // namespace ckw

#endif // CKW_PROTOTYPE_INCLUDE_CKW_KERNEL_H
