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
#include "src/cl/helpers/CLMemoryOpBufferHelper.h"

#include "ckw/Error.h"
#include "ckw/TensorSampler.h"
#include "ckw/types/MemoryOperation.h"
#include "ckw/types/TensorStorageType.h"

#include "src/ITensor.h"
#include "src/Tensor3dMapper.h"
#include "src/cl/CLHelpers.h"
#include "src/cl/CLKernelWriter.h"
#include "src/cl/CLTensorArgument.h"
#include "src/cl/CLTile.h"

namespace ckw
{
bool CLMemoryOpBufferHelper::validate(const CLKernelWriter *writer, const ITensor *tensor, const TensorSampler *sampler, const Tensor3dMapper *mapper, MemoryOperation op, const CLTile *dst)
{
    CKW_UNUSED(writer, tensor, mapper, op, dst);

    if(sampler->storage() != TensorStorageType::BufferUint8Ptr)
    {
        return false;
    }
    return true;
}

/** Initialization and Finalizing Logic
 *
 *   The meanings of if/elses in different dimensions and how they're constructed:
 *   - x: partial load/store
 *   - y: no load/store operation
 *   - z: no load/store operation
 *   if(x)
 *   {
 *       if(z)
 *       {
 *           if(y)
 *           {
 *               // full load/store width
 *           }
 *           else
 *           {
 *               // no load/store
 *           }
 *       }
 *       else
 *       {
 *           // no load/store
 *       }
 *   }
 *   else
 *   {
 *       if(z)
 *       {
 *           if(y)
 *           {
 *               // partial load/store width
 *           }
 *           else
 *           {
 *               // no load/store
 *           }
 *       }
 *       else
 *       {
 *           // no load/store
 *       }
 *   }
 *
 *  In general, initialize() writes if conditions, and finalize() writes else conditions.
 *  The outermost block is x, then z and then y. This is why, if/else's covering for y are initialized
 *  at each row write. In some addressing modes, such as None, no if/else conditions are written.
 */
void CLMemoryOpBufferHelper::initialize(const CLTile *dst, const CLTile *x, const CLTile *z, const CLTile *b)
{
    _dst           = dst;

    CKW_ASSERT(validate(_writer, _tensor, _sampler, _mapper.get(), _op, _dst));

    _ls_width_full = dst->info().width();
    _coord_x      = x->scalar(0, 0).str;
    _coord_z      = z->scalar(0, 0).str;
    _coord_b      = b->scalar(0, 0).str;
    _coord_orig_z = _coord_z;

    out_of_bound_initialize_x(_coord_x);
    out_of_bound_initialize_z(_coord_z);
}

void CLMemoryOpBufferHelper::write_row(int32_t row_id, const std::string &coord_y)
{
    // The only check required is on Y.
    out_of_bound_initialize_y(coord_y);

    const std::string dst     = _dst->vector(row_id).str;
    const std::string address = to_buffer_address(_coord_x, coord_y, _coord_z, _coord_b);
    const std::string ls_buf  = to_statement(_op, _ls_width_full, dst, address);

    _writer->op_write_raw_code(ls_buf);
    _writer->op_write_raw_code(";\n");

    out_of_bound_finalize_y(dst);

    // The left over load/store will be written in the finalize stage
    if(_ls_width_part.size() != 0)
    {
        int32_t col_start = 0;
        for(int32_t partial_width : _ls_width_part)
        {
            const std::string dst       = _dst->vector(row_id, col_start, partial_width).str;
            const std::string coord_x   = _coord_x + " + " + std::to_string(col_start);
            const std::string address   = to_buffer_address(coord_x, coord_y, _coord_z, _coord_b);
            const std::string statement = to_statement(_op, partial_width, dst, address);
            _leftovers_x.emplace_back(dst, coord_y, statement);

            col_start += partial_width;
        }
    }
}

void CLMemoryOpBufferHelper::finalize()
{
    out_of_bound_finalize_z();
    out_of_bound_finalize_x();
}

void CLMemoryOpBufferHelper::out_of_bound_initialize_x(const std::string &coord)
{
    if(_sampler->address_mode_x() == TensorSamplerAddressModeX::OverlappingMin)
    {
        TensorInfo tensor_info = _tensor->info();
        TensorShape shape      = tensor_info.shape();

        _ls_width_part = cl_decompose_vector_width(shape[0] % _ls_width_full);
        if(_ls_width_part.size() != 0)
        {
            _writer->op_write_raw_code("if(" + coord + " > 0)\n{\n");
        }
    }
}

void CLMemoryOpBufferHelper::out_of_bound_finalize_x()
{
    if(_sampler->address_mode_x() == TensorSamplerAddressModeX::OverlappingMin)
    {
        if(_ls_width_part.size() != 0)
        {
            _writer->op_write_raw_code("}\nelse\n{\n");

            out_of_bound_initialize_z(_coord_orig_z);
            for(LeftoverDescriptor leftover_desc : _leftovers_x)
            {
                out_of_bound_initialize_y(leftover_desc.coord);
                _writer->op_write_raw_code(leftover_desc.statement);
                _writer->op_write_raw_code(";\n");
                out_of_bound_finalize_y(leftover_desc.dst);
            }
            out_of_bound_finalize_z();
            _writer->op_write_raw_code("}\n");
        }
    }
}

void CLMemoryOpBufferHelper::out_of_bound_initialize_y(const std::string &coord)
{
    std::string max = "";

    const TensorSamplerAddressModeY address_mode_y = _sampler->address_mode_y();

    switch(address_mode_y)
    {
        case TensorSamplerAddressModeY::ClampToBorderMaxOnly:
            // Not to be moved outside the case because it marks the relevant tensor component as used even if we dont't use the variable
            max = _mapper->dim_y().str;
            _writer->op_write_raw_code("if(" + coord + " < " + max + ")\n{\n");
            break;
        case TensorSamplerAddressModeY::SkipLessThanZero:
            _writer->op_write_raw_code("if(" + coord + " >= 0)\n{\n");
            break;
        case TensorSamplerAddressModeY::None:
            break;
        default:
            CKW_THROW_MSG("Unsupported address mode for Y dimension");
    }
}

void CLMemoryOpBufferHelper::out_of_bound_finalize_y(const std::string &dst)
{
    const TensorSamplerAddressModeY address_mode_y = _sampler->address_mode_y();

    switch(address_mode_y)
    {
        case TensorSamplerAddressModeY::ClampToBorderMaxOnly:
            _writer->op_write_raw_code("}\nelse\n{\n");
            _writer->op_write_raw_code(dst);
            _writer->op_write_raw_code(" = 0.0f;\n}\n");
            break;
        case TensorSamplerAddressModeY::SkipLessThanZero:
            _writer->op_write_raw_code("}\n");
            break;
        case TensorSamplerAddressModeY::None:
            break;
        default:
            CKW_THROW_MSG("Unsupported address mode for Y dimension");
    }
}

void CLMemoryOpBufferHelper::out_of_bound_initialize_z(const std::string &coord)
{
    CKW_UNUSED(coord);

    const TensorSamplerAddressModeZ address_mode_z = _sampler->address_mode_z();
    switch(address_mode_z)
    {
        case TensorSamplerAddressModeZ::None:
            break;
        default:
            CKW_THROW_MSG("Unsupported address mode for Z dimension");
    }
}

void CLMemoryOpBufferHelper::out_of_bound_finalize_z()
{
    const TensorSamplerAddressModeZ address_mode_z = _sampler->address_mode_z();

    switch(address_mode_z)
    {
        case TensorSamplerAddressModeZ::None:
            break;
        default:
            CKW_THROW_MSG("Unsupported address mode for Z dimension");
    }
}

std::string CLMemoryOpBufferHelper::to_statement(MemoryOperation op, int32_t vector_width, const std::string &data,
                             const std::string &address) const
{
    switch(op)
    {
        case MemoryOperation::Load:
            if(vector_width != 1)
            {
                return data + " = vload" + std::to_string(vector_width) + "(0, " + address + ")";
            }
            else
            {
                return data + " = *(" + address + ")";
            }
            break;
        case MemoryOperation::Store:
            if(vector_width != 1)
            {
                return "vstore" + std::to_string(vector_width) + "(" + data + ", 0, " + address + ")";
            }
            else
            {
                return "*(" + address + ") = " + data;
            }
            break;
        default:
            CKW_THROW_MSG("Unsupported MemoryOperation");
    }

    return "";
}

std::string CLMemoryOpBufferHelper::to_buffer_address(const std::string &x, const std::string &y, const std::string &z,
                                     const std::string &b) const
{
    TensorStorageType tensor_storage = _sampler->storage();
    CKW_ASSERT(tensor_storage == TensorStorageType::BufferUint8Ptr);

    const std::string ptr_buf      = _tensor->storage(tensor_storage).val;
    const std::string dst_type     = cl_data_type_rounded_up_to_valid_vector_width(_dst->info().data_type(), 1);

    std::string address;
    address += "(__global ";
    address += dst_type;
    address += "*)(";
    address += ptr_buf;
    if(x != "0" && (_mapper->dim_x().str != "1"))
    {
        address += " + (";
        address += x + ") * sizeof(" + dst_type + ")";
    }
    if(y != "0")
    {
        const std::string stride_y = _mapper->stride_y().str;
        address += " + (";
        address += y + ")";
        address += " * ";
        address += stride_y;
    }
    if(z != "0" && (_mapper->dim_z().str != "1"))
    {
        const std::string stride_z = _mapper->stride_z().str;
        address += " + (";
        address += z + ")";
        address += " * ";
        address += stride_z;
    }
    if(b != "0" && (_mapper->dim_batch().str != "1"))
    {
        const std::string stride_b = _mapper->stride_batch().str;
        address += " + (";
        address += b + ")";
        address += " * ";
        address += stride_b;
    }
    address += ")";
    return address;
}
} // namespace ckw
