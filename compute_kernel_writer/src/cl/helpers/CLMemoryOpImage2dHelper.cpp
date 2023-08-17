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
#include "src/cl/helpers/CLMemoryOpImage2dHelper.h"

#include "ckw/Error.h"
#include "ckw/TensorSampler.h"
#include "ckw/types/MemoryOperation.h"
#include "ckw/types/TensorStorageType.h"

#include "src/ITensor.h"
#include "src/Tensor3dMapper.h"
#include "src/cl/CLKernelWriter.h"
#include "src/cl/CLTensorArgument.h"
#include "src/cl/CLTile.h"

namespace ckw
{
void CLMemoryOpImage2dHelper::initialize(const CLTile *dst, const CLTile *x, const CLTile *z, const CLTile *b)
{
    CKW_ASSERT(validate(_writer, _tensor, _sampler, _mapper.get(), _op, dst));

    _dst           = dst;
    _ls_width_full = dst->info().width();
    _coord_x       = x->scalar(0, 0).str;
    _coord_z       = z->scalar(0, 0).str;
    _coord_b       = b->scalar(0, 0).str;
}

void CLMemoryOpImage2dHelper::write_row(int32_t row_id, const std::string &coord_y)
{
    // The only check required is on Y.
    out_of_bound_initialize_y(coord_y);

    const std::string dst     = _dst->vector(row_id).str;
    const std::string sampler = to_ls_image2d_sampler();
    const std::string coord   = to_ls_image2d_address(_coord_x, coord_y, _coord_z, _coord_b);
    const std::string ls_buf  = to_ls_image2d(_op, _ls_width_full, dst, sampler, coord);

    _writer->op_write_raw_code(ls_buf + ";\n");

    out_of_bound_finalize_y();
}

void CLMemoryOpImage2dHelper::finalize()
{
}

bool CLMemoryOpImage2dHelper::validate(const CLKernelWriter *writer, const ITensor *tensor, const TensorSampler *sampler, const Tensor3dMapper *mapper, MemoryOperation op, const CLTile *dst)
{
    CKW_UNUSED(writer, tensor, mapper);

    if(dst->info().width() != 4)
    {
        return false;
    }
    if(sampler->address_mode_x() != TensorSamplerAddressModeX::None)
    {
        return false;
    }
    if(sampler->address_mode_z() != TensorSamplerAddressModeZ::None)
    {
        return false;
    }
    if(sampler->storage() != TensorStorageType::Texture2dReadOnly && op == MemoryOperation::Load)
    {
        return false;
    }
    if(sampler->storage() != TensorStorageType::Texture2dWriteOnly && op == MemoryOperation::Store)
    {
        return false;
    }
    if((dst->info().data_type() != DataType::Fp32) && (dst->info().data_type() != DataType::Fp16))
    {
        return false;
    }
    return true;
}

void CLMemoryOpImage2dHelper::out_of_bound_initialize_y(const std::string &coord)
{
    CKW_UNUSED(coord);

    const TensorSamplerAddressModeY address_mode_y = _sampler->address_mode_y();
    switch(address_mode_y)
    {
        case TensorSamplerAddressModeY::SkipLessThanZero:
            _writer->op_write_raw_code("if(" + coord + " >= 0)\n{\n");
            break;
        case TensorSamplerAddressModeY::ClampToBorderMaxOnly:
        case TensorSamplerAddressModeY::None:
            break;
        default:
            CKW_THROW_MSG("Unsupported address mode for Y dimension");
    }
}

void CLMemoryOpImage2dHelper::out_of_bound_finalize_y()
{
    const TensorSamplerAddressModeY address_mode_y = _sampler->address_mode_y();
    switch(address_mode_y)
    {
        case TensorSamplerAddressModeY::SkipLessThanZero:
            _writer->op_write_raw_code("}\n");
            break;
        case TensorSamplerAddressModeY::ClampToBorderMaxOnly:
        case TensorSamplerAddressModeY::None:
            break;
        default:
            CKW_THROW_MSG("Unsupported address mode for Y dimension");
    }
}

std::string CLMemoryOpImage2dHelper::to_ls_image2d(MemoryOperation op, int32_t vector_width, const std::string &data, const std::string &sampler, const std::string &address) const
{
    CKW_UNUSED(vector_width);

    const TensorStorageType tensor_storage = _sampler->storage();
    const std::string image2d_obj    = _tensor->storage(tensor_storage).val;
    const std::string post_fix = _dst->info().data_type() == DataType::Fp32 ? "f" : "h";

    switch(op)
    {
        case MemoryOperation::Load:
            return data + " = read_image" + post_fix + "(" + image2d_obj + ", " + sampler + ", " + address + ")";
            break;
        case MemoryOperation::Store:
            return "write_image" + post_fix + "(" + image2d_obj + ", " + address + ", " + data + ")";
        default:
            CKW_THROW_MSG("Unsupported MemoryOperation");
    }
}

std::string CLMemoryOpImage2dHelper::to_ls_image2d_sampler() const
{
    const auto address_mode_y = _sampler->address_mode_y();

    switch(address_mode_y)
    {
        case TensorSamplerAddressModeY::None:
            return "CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST";
        case TensorSamplerAddressModeY::SkipLessThanZero:
        case TensorSamplerAddressModeY::ClampToBorderMaxOnly:
            return "CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST";
        default:
            CKW_THROW_MSG("Unsupported address_mode_coord");
    }
}

std::string CLMemoryOpImage2dHelper::to_ls_image2d_address(const std::string &x, const std::string &y, const std::string &z,
                                                           const std::string &b) const
{
    std::string coord_x = "(" + x + ") >> 2";
    std::string coord_y = "(";

    if(y != "0")
    {
        coord_y += y;
    }
    if(z != "0" && (_mapper->dim_z().str != "1"))
    {
        const std::string dim = _mapper->dim_y().str;
        coord_y += " + (";
        coord_y += z + ")";
        coord_y += " * ";
        coord_y += dim;
    }
    if(b != "0" && (_mapper->dim_batch().str != "1"))
    {
        const std::string dim0 = _mapper->dim_y().str;
        const std::string dim1 = _mapper->dim_z().str;
        coord_y += " + (";
        coord_y += b + ")";
        coord_y += " * ";
        coord_y += dim0;
        coord_y += " * ";
        coord_y += dim1;
    }
    coord_y += ")";
    return "(int2)(" + coord_x + ", " + coord_y + ")";
}

} // namespace ckw
