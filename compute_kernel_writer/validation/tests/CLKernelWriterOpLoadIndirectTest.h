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

#ifndef CKW_VALIDATION_TESTS_CLKERNELWRITEROPLOADINDIRECTTEST_H
#define CKW_VALIDATION_TESTS_CLKERNELWRITEROPLOADINDIRECTTEST_H

#include "ckw/TileInfo.h"
#include "ckw/types/DataType.h"
#include "ckw/TensorSampler.h"
#include "ckw/types/MemoryOperation.h"
#include "ckw/types/TensorSamplerTypes.h"
#include "src/cl/CLKernelWriter.h"
#include "validation/tests/common/KernelWriterInterceptor.h"
#include "validation/tests/common/Common.h"

#include <vector>

namespace ckw
{

class CLKernelWriterOpLoadIndirectTest : public ITest
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

    using CLKernelWriterOpLoadIndirectConfig = std::tuple<TileInfo, TensorStorageType, SamplerData, Coordinates, std::string>;

public:
    CLKernelWriterOpLoadIndirectTest()
    {
        const std::string fp_2x3_tile = R"_(
G0__tile__0 = vload3(0, (__global float*)(G0__tensor_ptr + (G0__x) * sizeof(float) + (G0__indirect_addr__0) * G0__tensor_stride1 + (G0__z) * G0__tensor_stride2 + (G0__b) * G0__tensor_stride3));
G0__tile__1 = vload3(0, (__global float*)(G0__tensor_ptr + (G0__x) * sizeof(float) + (G0__indirect_addr__1) * G0__tensor_stride1 + (G0__z) * G0__tensor_stride2 + (G0__b) * G0__tensor_stride3));
)_";

        const std::string half_2x4_yz_collapsed_y_clamped_to_border_max_only_image = R"_(
G0__tile__0 = read_imageh(G0__tensor_img2d, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST, (int2)((G0__x) >> 2, (G0__indirect_addr__0 + (G0__b) * G0__tensor_dim1xdim2 * 1)));
G0__tile__1 = read_imageh(G0__tensor_img2d, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST, (int2)((G0__x) >> 2, (G0__indirect_addr__1 + (G0__b) * G0__tensor_dim1xdim2 * 1)));
)_";

        const std::string int_2x4_y_skip_less_than_zero = R"_(
if(G0__indirect_addr__0 >= 0)
{
G0__tile__0 = vload4(0, (__global int*)(G0__tensor_ptr + (G0__x) * sizeof(int) + (G0__indirect_addr__0) * G0__tensor_stride1 + (G0__z) * G0__tensor_stride2 + (G0__b) * G0__tensor_stride3));
}
if(G0__indirect_addr__1 >= 0)
{
G0__tile__1 = vload4(0, (__global int*)(G0__tensor_ptr + (G0__x) * sizeof(int) + (G0__indirect_addr__1) * G0__tensor_stride1 + (G0__z) * G0__tensor_stride2 + (G0__b) * G0__tensor_stride3));
}
)_";

        // tensor shape in x-dim is 10 (thus the 8, 2 vloads in if, else blocks respectively)
        const std::string uint16_3x8_yz_collapsed_b_eq_0_x_overlapping_min_y_skip_less_than_zero = R"_(
if(G0__x > 0)
{
if(G0__indirect_addr__0 >= 0)
{
G0__tile__0 = vload8(0, (__global ushort*)(G0__tensor_ptr + (G0__x) * sizeof(ushort) + (G0__indirect_addr__0) * G0__tensor_stride1 + (G0__0) * G0__tensor_stride3));
}
if(G0__indirect_addr__1 >= 0)
{
G0__tile__1 = vload8(0, (__global ushort*)(G0__tensor_ptr + (G0__x) * sizeof(ushort) + (G0__indirect_addr__1) * G0__tensor_stride1 + (G0__0) * G0__tensor_stride3));
}
if(G0__indirect_addr__2 >= 0)
{
G0__tile__2 = vload8(0, (__global ushort*)(G0__tensor_ptr + (G0__x) * sizeof(ushort) + (G0__indirect_addr__2) * G0__tensor_stride1 + (G0__0) * G0__tensor_stride3));
}
}
else
{
if(G0__indirect_addr__0 >= 0)
{
G0__tile__0.s01 = vload2(0, (__global ushort*)(G0__tensor_ptr + (G0__x + 0) * sizeof(ushort) + (G0__indirect_addr__0) * G0__tensor_stride1 + (G0__0) * G0__tensor_stride3));
}
if(G0__indirect_addr__1 >= 0)
{
G0__tile__1.s01 = vload2(0, (__global ushort*)(G0__tensor_ptr + (G0__x + 0) * sizeof(ushort) + (G0__indirect_addr__1) * G0__tensor_stride1 + (G0__0) * G0__tensor_stride3));
}
if(G0__indirect_addr__2 >= 0)
{
G0__tile__2.s01 = vload2(0, (__global ushort*)(G0__tensor_ptr + (G0__x + 0) * sizeof(ushort) + (G0__indirect_addr__2) * G0__tensor_stride1 + (G0__0) * G0__tensor_stride3));
}
}
)_";

        // Configs Bundled
        _configs = {
            {
                TileInfo(DataType::Fp32, 2, 3),
                TensorStorageType::BufferUint8Ptr,
                SamplerData(Format::Dim0_Dim1_Dim2, AddressModeX::None, AddressModeY::None, AddressModeZ::None),
                Coordinates("x", "y", "z", "b"),
                fp_2x3_tile
            },
            {
                TileInfo(DataType::Fp16, 2, 4),
                TensorStorageType::Texture2dReadOnly,
                SamplerData(Format::Dim0_Dim1xDim2_1, AddressModeX::None, AddressModeY::ClampToBorderMaxOnly, AddressModeZ::None),
                Coordinates("x", "y", "z", "b"),
                half_2x4_yz_collapsed_y_clamped_to_border_max_only_image
            },
            {
                TileInfo(DataType::Int32, 2, 4),
                TensorStorageType::BufferUint8Ptr,
                SamplerData(Format::Dim0_Dim1_Dim2, AddressModeX::None, AddressModeY::SkipLessThanZero, AddressModeZ::None),
                Coordinates("x", "y", "z", "b"),
                int_2x4_y_skip_less_than_zero
            },
            {
                TileInfo(DataType::Uint16, 3, 8),
                TensorStorageType::BufferUint8Ptr,
                SamplerData(Format::Dim0_Dim1xDim2_1, AddressModeX::OverlappingMin, AddressModeY::SkipLessThanZero, AddressModeZ::None),
                Coordinates("x", "y", "z", "0"),
                uint16_3x8_yz_collapsed_b_eq_0_x_overlapping_min_y_skip_less_than_zero
            }
        };
    }

    bool run() override
    {
        bool all_tests_passed = true;
        int32_t test_idx = 0;

        for(auto _config: _configs)
        {
            KernelWriterInterceptor<CLKernelWriter> writer;

            const TileInfo tile_info = std::get<0>(_config);
            const Storage storage = std::get<1>(_config);
            const SamplerData sampler_data = std::get<2>(_config);
            const Coordinates coord = std::get<3>(_config);
            const std::string expected_code = std::get<4>(_config).substr(1); // ignore initial newline, which was added for convenience

            TileOperand tile_op = writer.declare_tile("tile", TileInfo(tile_info.data_type(), tile_info.height(), tile_info.width()));
            TileOperand indirect_addr_op = writer.declare_tile("indirect_addr", TileInfo(DataType::Int32, tile_info.height(), 1)); // (M0, 1)
            TileOperand x_op = writer.declare_tile(coord.x, TileInfo(DataType::Int32));
            TileOperand z_op = writer.declare_tile(coord.z, TileInfo(DataType::Int32));
            TileOperand batch_op = writer.declare_tile(coord.batch, TileInfo(DataType::Int32));

            TensorShape tensor_shape {10, 10, 10, 10};
            TensorInfo tensor_info(tile_info.data_type(), tensor_shape, TensorDataLayout::Nhwc, 0 /* id */);
            TensorOperand tensor_op = writer.declare_tensor_argument("tensor", tensor_info);
            TensorSampler sampler(storage, sampler_data.format, sampler_data.mode_x, sampler_data.mode_y, sampler_data.mode_z);

            writer.start_capture_code();
            writer.op_load_indirect(tile_op, tensor_op, sampler, x_op, indirect_addr_op, z_op, batch_op);

            VALIDATE_TEST(writer.check_added_code(expected_code), all_tests_passed, test_idx++);
        }

        return all_tests_passed;
    }

    std::string name() override
    {
        return "CLKernelWriterOpLoadIndirectTest";
    }

private:
    std::vector<CLKernelWriterOpLoadIndirectConfig> _configs {};
};

} // namespace ckw

#endif // CKW_VALIDATION_TESTS_CLKERNELWRITEROPLOADINDIRECTTEST_H
