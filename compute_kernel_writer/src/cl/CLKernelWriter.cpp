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

#include "src/cl/CLKernelWriter.h"
#include "ckw/Error.h"
#include "ckw/TileOperand.h"
#include "src/cl/CLHelpers.h"
#include "src/cl/CLTile.h"
#include <cstdint>

namespace ckw
{

CLKernelWriter::CLKernelWriter()  = default;
CLKernelWriter::~CLKernelWriter() = default;

std::unique_ptr<Kernel> CLKernelWriter::emit_kernel(const std::string &name)
{
    CKW_UNUSED(name);
    CKW_THROW_MSG("Not implemented!");
}

void CLKernelWriter::comment(const std::string &text)
{
#ifdef COMPUTE_KERNEL_WRITER_DEBUG_ENABLED

    CKW_ASSERT(text.find("\n") == text.npos);
    CKW_ASSERT(text.find("\r") == text.npos);

    append_code("// ", text, "\n");

#else // COMPUTE_KERNEL_WRITER_DEBUG_ENABLED

    CKW_UNUSED(text);

#endif // COMPUTE_KERNEL_WRITER_DEBUG_ENABLED
}

const std::string &CLKernelWriter::body_source_code() const
{
    return _body_source_code;
}

TileOperand CLKernelWriter::declare_tile(const std::string &name, const TileInfo &tile_info)
{
    const std::string fullname = generate_full_name(name);

    const int32_t  height    = tile_info.height();
    const int32_t  width     = tile_info.width();
    const DataType data_type = tile_info.data_type();

    for(int32_t row = 0; row < height; ++row)
    {
        const std::string cl_type = cl_get_variable_datatype_as_string(data_type, width);
        append_code(cl_type, " ", fullname, std::to_string(row), ";\n");
    }

    auto       tile    = std::make_unique<CLTile>(name, tile_info);
    const auto operand = create_tile_operand(*tile);

    _tiles.insert(std::move(tile));

    return operand;
}

} // namespace ckw
