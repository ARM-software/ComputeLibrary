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

#ifndef CKW_VALIDATION_TESTS_CLKERNELWRITEROPLOADSTORETEST_H
#define CKW_VALIDATION_TESTS_CLKERNELWRITEROPLOADSTORETEST_H

#include "ckw/TileInfo.h"
#include "ckw/types/DataType.h"
#include "src/cl/CLKernelWriter.h"
#include "validation/tests/common/KernelWriterInterceptor.h"
#include "validation/tests/common/Common.h"

#include "ckw/TensorSampler.h"
#include "ckw/types/MemoryOperation.h"
#include "ckw/types/TensorSamplerTypes.h"

#include <vector>

namespace ckw
{

class CLKernelWriterOpLoadStoreTest : public ITest
{
private:
    using AddressModeX = TensorSamplerAddressModeX;
    using AddressModeY = TensorSamplerAddressModeY;
    using AddressModeZ = TensorSamplerAddressModeZ;
    using Format = TensorSamplerFormat;
    using Storage = TensorStorageType;

    struct Coordinates
    {
        Coordinates(std::string x, std::string y, std::string z, std::string batch)
            : x(x), y(y), z(z), batch(batch)
        {
        }

        std::string x;
        std::string y;
        std::string z;
        std::string batch;
    };

    struct SamplerData
    {
        SamplerData(Format format, AddressModeX mode_x, AddressModeY mode_y, AddressModeZ mode_z)
        :   format(format), mode_x(mode_x), mode_y(mode_y), mode_z(mode_z)
        {
        }

        Format format;
        AddressModeX mode_x;
        AddressModeY mode_y;
        AddressModeZ mode_z;
    };

    struct Dilations
    {
        Dilations(std::string dilation_x, std::string dilation_y)
        : dilation_x(dilation_x), dilation_y(dilation_y)
        {
        }

        std::string dilation_x;
        std::string dilation_y;
    };

    using CLKernelWriterOpLoadStoreConfig = std::tuple<MemoryOperation, TileInfo, TensorStorageType, SamplerData, Coordinates, Dilations, std::string>;

public:
    CLKernelWriterOpLoadStoreTest()
    {
        // Cases
        const std::string load_fp_2x3_tile = R"_(
G0__tile__0 = vload3(0, (__global float*)(G0__tensor_ptr + (G0__x) * sizeof(float) + (G0__y + 0) * G0__tensor_stride1 + (G0__z) * G0__tensor_stride2 + (G0__b) * G0__tensor_stride3));
G0__tile__1 = vload3(0, (__global float*)(G0__tensor_ptr + (G0__x) * sizeof(float) + (G0__y + 1) * G0__tensor_stride1 + (G0__z) * G0__tensor_stride2 + (G0__b) * G0__tensor_stride3));
)_";
        const std::string load_half_2x4_tile_image_clamp_y = R"_(
G0__tile__0 = read_imageh(G0__tensor_img2d, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST, (int2)((G0__x) >> 2, (G0__y + 0 + (G0__z) * G0__tensor_dim1 + (G0__b) * G0__tensor_dim1 * G0__tensor_dim2)));
G0__tile__1 = read_imageh(G0__tensor_img2d, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST, (int2)((G0__x) >> 2, (G0__y + 1 + (G0__z) * G0__tensor_dim1 + (G0__b) * G0__tensor_dim1 * G0__tensor_dim2)));
)_";
        const std::string store_fp_2x3_tile = R"_(
vstore3(G0__tile__0, 0, (__global float*)(G0__tensor_ptr + (G0__x) * sizeof(float) + (G0__y + 0) * G0__tensor_stride1 + (G0__b) * G0__tensor_stride3));
vstore3(G0__tile__1, 0, (__global float*)(G0__tensor_ptr + (G0__x) * sizeof(float) + (G0__y + 1) * G0__tensor_stride1 + (G0__b) * G0__tensor_stride3));
)_";
        const std::string store_int8_4x4_y_dilation_batch_eq_0 = R"_(
vstore4(G0__tile__0, 0, (__global char*)(G0__tensor_ptr + (((int)(1))) * sizeof(char) + (G0__y + 0 * G0__y_dilation) * G0__tensor_stride1 + (G0__z) * G0__tensor_stride2 + (((int)(0))) * G0__tensor_stride3));
vstore4(G0__tile__1, 0, (__global char*)(G0__tensor_ptr + (((int)(1))) * sizeof(char) + (G0__y + 1 * G0__y_dilation) * G0__tensor_stride1 + (G0__z) * G0__tensor_stride2 + (((int)(0))) * G0__tensor_stride3));
vstore4(G0__tile__2, 0, (__global char*)(G0__tensor_ptr + (((int)(1))) * sizeof(char) + (G0__y + 2 * G0__y_dilation) * G0__tensor_stride1 + (G0__z) * G0__tensor_stride2 + (((int)(0))) * G0__tensor_stride3));
vstore4(G0__tile__3, 0, (__global char*)(G0__tensor_ptr + (((int)(1))) * sizeof(char) + (G0__y + 3 * G0__y_dilation) * G0__tensor_stride1 + (G0__z) * G0__tensor_stride2 + (((int)(0))) * G0__tensor_stride3));
)_";
        // tensor dimension is 10
        const std::string load_fp_2x3_tile_x_overlapping_min_y_eq_0_batch_eq_1 = R"_(
if(G0__x > 0)
{
G0__tile__0 = vload3(0, (__global float*)(G0__tensor_ptr + (G0__x) * sizeof(float) + (((int)(0)) + 0) * G0__tensor_stride1 + (G0__z) * G0__tensor_stride2 + (((int)(1))) * G0__tensor_stride3));
G0__tile__1 = vload3(0, (__global float*)(G0__tensor_ptr + (G0__x) * sizeof(float) + (((int)(0)) + 1) * G0__tensor_stride1 + (G0__z) * G0__tensor_stride2 + (((int)(1))) * G0__tensor_stride3));
}
else
{
G0__tile__0.s0 = *((__global float*)(G0__tensor_ptr + (G0__x + 0) * sizeof(float) + (((int)(0)) + 0) * G0__tensor_stride1 + (G0__z) * G0__tensor_stride2 + (((int)(1))) * G0__tensor_stride3));
G0__tile__1.s0 = *((__global float*)(G0__tensor_ptr + (G0__x + 0) * sizeof(float) + (((int)(0)) + 1) * G0__tensor_stride1 + (G0__z) * G0__tensor_stride2 + (((int)(1))) * G0__tensor_stride3));
}
)_";
        const std::string store_fp_2x3_tile_x_overlapping_min_y_clamp_to_border_max_only = R"_(
if(G0__x > 0)
{
if(G0__y + 0 < G0__tensor_dim1)
{
vstore3(G0__tile__0, 0, (__global float*)(G0__tensor_ptr + (G0__x) * sizeof(float) + (G0__y + 0) * G0__tensor_stride1 + (G0__z) * G0__tensor_stride2 + (G0__b) * G0__tensor_stride3));
}
else
{
G0__tile__0 = 0.0f;
}
if(G0__y + 1 < G0__tensor_dim1)
{
vstore3(G0__tile__1, 0, (__global float*)(G0__tensor_ptr + (G0__x) * sizeof(float) + (G0__y + 1) * G0__tensor_stride1 + (G0__z) * G0__tensor_stride2 + (G0__b) * G0__tensor_stride3));
}
else
{
G0__tile__1 = 0.0f;
}
}
else
{
if(G0__y + 0 < G0__tensor_dim1)
{
*((__global float*)(G0__tensor_ptr + (G0__x + 0) * sizeof(float) + (G0__y + 0) * G0__tensor_stride1 + (G0__z) * G0__tensor_stride2 + (G0__b) * G0__tensor_stride3)) = G0__tile__0.s0;
}
else
{
G0__tile__0.s0 = 0.0f;
}
if(G0__y + 1 < G0__tensor_dim1)
{
*((__global float*)(G0__tensor_ptr + (G0__x + 0) * sizeof(float) + (G0__y + 1) * G0__tensor_stride1 + (G0__z) * G0__tensor_stride2 + (G0__b) * G0__tensor_stride3)) = G0__tile__1.s0;
}
else
{
G0__tile__1.s0 = 0.0f;
}
}
)_";
        const std::string store_half_2x4_tile_x_image_y_dilation = R"_(
write_imageh(G0__tensor_img2d, (int2)((G0__x) >> 2, (((int)(0)) + 0 * G0__y_dilation + (G0__z) * G0__tensor_dim1 + (((int)(1))) * G0__tensor_dim1 * G0__tensor_dim2)), G0__tile__0);
write_imageh(G0__tensor_img2d, (int2)((G0__x) >> 2, (((int)(0)) + 1 * G0__y_dilation + (G0__z) * G0__tensor_dim1 + (((int)(1))) * G0__tensor_dim1 * G0__tensor_dim2)), G0__tile__1);
)_";

        // Configs Bundled
        _configs = {
            // op, tile,  storage, sampler, coordinates, dilation, expected
            {
                MemoryOperation::Load,
                TileInfo(DataType::Fp32, 2, 3),
                TensorStorageType::BufferUint8Ptr,
                SamplerData(Format::Dim0_Dim1_Dim2, AddressModeX::None, AddressModeY::None, AddressModeZ::None),
                Coordinates("x", "y", "z", "b"),
                Dilations("1", "1"),
                load_fp_2x3_tile
            },
            {
                MemoryOperation::Load,
                TileInfo(DataType::Fp16, 2, 4),
                TensorStorageType::Texture2dReadOnly,
                SamplerData(Format::Dim0_Dim1_Dim2, AddressModeX::None, AddressModeY::ClampToBorderMaxOnly, AddressModeZ::None),
                Coordinates("x", "y", "z", "b"),
                Dilations("1", "1"),
                load_half_2x4_tile_image_clamp_y
            },
            {
                MemoryOperation::Store,
                TileInfo(DataType::Fp32, 2, 3),
                TensorStorageType::BufferUint8Ptr,
                SamplerData(Format::Dim0_Dim1xDim2_1,AddressModeX::None, AddressModeY::None, AddressModeZ::None),
                Coordinates("x", "y", "z", "b"),
                Dilations("1", "1"),
                store_fp_2x3_tile
            },
            {
                MemoryOperation::Store,
                TileInfo(DataType::Int8, 4, 4),
                TensorStorageType::BufferUint8Ptr,
                SamplerData(Format::Dim0_Dim1_Dim2, AddressModeX::None, AddressModeY::None, AddressModeZ::None),
                Coordinates("1", "y", "z", "0"),
                Dilations("1", "y_dilation"),
                store_int8_4x4_y_dilation_batch_eq_0
            },
            {
                MemoryOperation::Load,
                TileInfo(DataType::Fp32, 2, 3),
                TensorStorageType::BufferUint8Ptr,
                SamplerData(Format::Dim0_Dim1_Dim2, AddressModeX::OverlappingMin, AddressModeY::None, AddressModeZ::None),
                Coordinates("x", "0", "z", "1"),
                Dilations("1", "1"),
                load_fp_2x3_tile_x_overlapping_min_y_eq_0_batch_eq_1
            },
            {
                MemoryOperation::Store,
                TileInfo(DataType::Fp32, 2, 3),
                TensorStorageType::BufferUint8Ptr,
                SamplerData(Format::Dim0_Dim1_Dim2, AddressModeX::OverlappingMin, AddressModeY::ClampToBorderMaxOnly, AddressModeZ::None),
                Coordinates("x", "y", "z", "b"),
                Dilations("1", "1"),
                store_fp_2x3_tile_x_overlapping_min_y_clamp_to_border_max_only
            },
            {
                MemoryOperation::Store,
                TileInfo(DataType::Fp16, 2, 4),
                TensorStorageType::Texture2dWriteOnly,
                SamplerData(Format::Dim0_Dim1_Dim2, AddressModeX::None, AddressModeY::None, AddressModeZ::None),
                Coordinates("x", "0", "z", "1"),
                Dilations("1", "y_dilation"),
                store_half_2x4_tile_x_image_y_dilation
            }
        };
    }

    TileOperand declare_tile_helper(KernelWriter &writer, std::string tile)
    {
        if(tile == "0" || tile == "1")
        {
            return writer.declare_constant_tile(ConstantData({{std::stoi(tile)}}, DataType::Int32));
        }
        else
        {
            return writer.declare_tile(tile, TileInfo(DataType::Int32));
        }
    }

    bool run() override
    {
        bool all_tests_passed = true;
        int32_t test_idx = 0;

        for(auto _config: _configs)
        {
            KernelWriterInterceptor<CLKernelWriter> writer;

            const MemoryOperation op = std::get<0>(_config);
            const TileInfo tile_info = std::get<1>(_config);
            const Storage storage = std::get<2>(_config);
            const SamplerData sampler_data = std::get<3>(_config);
            const Coordinates coord = std::get<4>(_config);
            const Dilations dilations = std::get<5>(_config);
            const std::string expected_code = std::get<6>(_config).substr(1); // ignore initial newline, which was added for convenience

            TileOperand tile_op = writer.declare_tile("tile", tile_info);
            TileOperand x_op = declare_tile_helper(writer, coord.x);
            TileOperand y_op = declare_tile_helper(writer, coord.y);
            TileOperand z_op = declare_tile_helper(writer, coord.z);
            TileOperand batch_op = declare_tile_helper(writer, coord.batch);
            TileOperand dil_x_op = declare_tile_helper(writer, dilations.dilation_x);
            TileOperand dil_y_op = declare_tile_helper(writer, dilations.dilation_y);

            TensorShape tensor_shape {10, 10, 10, 10};
            TensorInfo tensor_info(tile_info.data_type(), tensor_shape, TensorDataLayout::Nhwc, 0 /* id */);
            TensorOperand tensor_op = writer.declare_tensor_argument("tensor", tensor_info);
            TensorSampler sampler(storage, sampler_data.format, sampler_data.mode_x, sampler_data.mode_y, sampler_data.mode_z);

            const bool no_dilation = (dilations.dilation_x == "1" && dilations.dilation_y == "1");

            writer.start_capture_code();
            if(op == MemoryOperation::Load)
            {
                if(no_dilation)
                {
                    writer.op_load(tile_op, tensor_op, sampler, x_op, y_op, z_op, batch_op);
                }
                else
                {
                    writer.op_load_dilated(tile_op, tensor_op, sampler, x_op, y_op, z_op, batch_op, dil_x_op, dil_y_op);
                }
            }
            else
            {
                if(no_dilation)
                {
                    writer.op_store(tensor_op, tile_op, sampler, x_op, y_op, z_op, batch_op);
                }
                else
                {
                    writer.op_store_dilated(tensor_op, tile_op, sampler, x_op, y_op, z_op, batch_op, dil_x_op, dil_y_op);
                }
            }

            VALIDATE_TEST(writer.check_added_code(expected_code), all_tests_passed, test_idx++);
        }

        return all_tests_passed;
    }

    std::string name() override
    {
        return "CLKernelWriterOpLoadStoreTest";
    }

private:
    std::vector<CLKernelWriterOpLoadStoreConfig> _configs {};
};

} // namespace ckw

#endif // CKW_VALIDATION_TESTS_CLKERNELWRITEROPLOADSTORETEST_H
