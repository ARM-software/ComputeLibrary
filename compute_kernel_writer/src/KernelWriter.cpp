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

#include "ckw/KernelWriter.h"
#include "ckw/Error.h"
#include "ckw/TileOperand.h"
#include "ckw/types/TargetArchitecture.h"
#include "ckw/types/TargetLanguage.h"
#include "src/cl/CLKernelWriter.h"
#include "src/cl/CLTensorArgument.h"
#include "src/cl/CLTile.h"

namespace ckw
{

KernelWriter::~KernelWriter() = default;

std::unique_ptr<KernelWriter> KernelWriter::create_instance(TargetArchitecture architecture, TargetLanguage language)
{
    CKW_UNUSED(architecture);
    switch(language)
    {
        case TargetLanguage::OpenCL:
            // Currently this is the oldest and the only supported GPU architecture.
            CKW_ASSERT(architecture == TargetArchitecture::GpuArmMaliValhall);
            return std::make_unique<CLKernelWriter>();

        default:
            CKW_THROW_MSG("Language not supported!");
    }
}

int32_t KernelWriter::id_space() const
{
    return _id_space;
}

std::string KernelWriter::generate_full_name(const std::string &name) const
{
    return "G" + std::to_string(id_space()) + "__" + name;
}

TileOperand KernelWriter::create_tile_operand(ITile &tile)
{
    return TileOperand(tile);
}

ITile &KernelWriter::get_tile(const TileOperand &operand)
{
    return operand._tile;
}

TensorOperand KernelWriter::create_tensor_operand(ITensor &tensor)
{
    return TensorOperand(tensor);
}

ITensor &KernelWriter::get_tensor(const TensorOperand &operand)
{
    return operand._tensor;
}

const std::vector<std::vector<std::string>> &KernelWriter::get_values(const ConstantData &data)
{
    return data.values();
}

DataType KernelWriter::get_data_type(const ConstantData &data)
{
    return data.data_type();
}

} // namespace ckw
