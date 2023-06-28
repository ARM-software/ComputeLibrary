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

#include "ckw/Kernel.h"
#include "ckw/TensorOperand.h"
#include "ckw/types/GpuTargetLanguage.h"
#include "src/Prototype.h"

namespace ckw
{

Kernel::Kernel(GpuTargetLanguage language)
    : Kernel{"unnamed", language}
{
}

Kernel::Kernel(const char *name, GpuTargetLanguage language)
    : _name(name), _kernel(std::make_unique<prototype::GpuKernelWriterDataHolder>(language)), _operands{}, _tensor_id_operands{}
{
}


Kernel::~Kernel()
{
}

const std::string &Kernel::name() const
{
    return _name;
}

void Kernel::name(const std::string& name)
{
    _name = name;
}
std::vector<KernelArgument> Kernel::arguments() const
{
    std::vector<KernelArgument> arguments;

    const auto impl_args = _kernel->arguments.tensor_argument_declarations();

    for(auto tensor_arg : impl_args)
    {
        auto tensor = _tensor_id_operands.at(tensor_arg->format().id);
        arguments.push_back(*tensor);

        for(auto component_arg : tensor_arg->component_declarations())
        {
            switch(component_arg)
            {
                case TensorComponentType::OffsetFirstElement:
                    arguments.push_back(tensor->offset_first_element_in_bytes());
                    break;

                case TensorComponentType::Stride1:
                    arguments.push_back(tensor->stride1());
                    break;

                case TensorComponentType::Stride2:
                    arguments.push_back(tensor->stride2());
                    break;

                case TensorComponentType::Stride3:
                    arguments.push_back(tensor->stride3());
                    break;

                case TensorComponentType::Stride4:
                    arguments.push_back(tensor->stride4());
                    break;

                case TensorComponentType::Dim0:
                    arguments.push_back(tensor->dim0());
                    break;

                case TensorComponentType::Dim1:
                    arguments.push_back(tensor->dim1());
                    break;

                case TensorComponentType::Dim2:
                    arguments.push_back(tensor->dim2());
                    break;

                case TensorComponentType::Dim3:
                    arguments.push_back(tensor->dim3());
                    break;

                case TensorComponentType::Dim4:
                    arguments.push_back(tensor->dim4());
                    break;

                case TensorComponentType::Dim1xDim2:
                    arguments.push_back(tensor->dim1_dim2());
                    break;

                case TensorComponentType::Dim1xDim2xDim3:
                    arguments.push_back(tensor->dim1_dim2_dim3());
                    break;

                default:
                    CKW_ASSERT(false);
            }
        }
    }

    return arguments;
}

TileOperand &Kernel::register_operand(std::unique_ptr<TileOperand> operand)
{
    const auto &name = operand->name();
    auto        ptr  = operand.get();

    CKW_ASSERT(_operands.find(name) == _operands.end());
    _operands[name] = std::move(operand);

    return *ptr;
}

TensorOperand &Kernel::register_operand(std::unique_ptr<TensorOperand> operand)
{
    const auto  id   = operand->info().id();
    const auto &name = operand->name();
    auto        ptr  = operand.get();

    CKW_ASSERT(_tensor_id_operands.find(id) == _tensor_id_operands.end());
    CKW_ASSERT(_operands.find(name) == _operands.end());

    _tensor_id_operands[id] = operand.get();
    _operands[name]         = std::move(operand);

    return *ptr;
}

prototype::GpuKernelWriterDataHolder *Kernel::impl()
{
    return _kernel.get();
}

} // namespace ckw
