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

#include "ExampleKernelWriter.h"

#include "ckw/Error.h"
#include "ckw/TileInfo.h"

#include "ExampleComponentArgument.h"

ExampleKernelWriter::ExampleKernelWriter(ckw::Kernel &kernel) : KernelWriter(kernel)
{
}

void ExampleKernelWriter::op_load_once(ExampleComponentArgument *tensor_or_tile, const ckw::TensorTileSampler &sampler)
{
    if (!tensor_or_tile->has_tile())
    {
        CKW_ASSERT(tensor_or_tile->has_tensor());

        auto &tensor = tensor_or_tile->tensor();

        const auto tile_name = tensor.name() + "_tile";
        auto      &tile =
            declare_tile(tile_name.c_str(), ckw::TileInfo(tensor.data_type(), sampler.height(), sampler.width()));

        op_load(tile, tensor, sampler);

        tensor_or_tile->init_virtual_tensor(tile, sampler);
    }
}
