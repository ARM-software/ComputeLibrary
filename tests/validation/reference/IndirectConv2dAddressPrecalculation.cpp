/*
 * Copyright (c) 2022 Arm Limited.
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
#include "IndirectConv2dAddressPrecalculation.h"

#include "arm_compute/core/Types.h"

#include "tests/validation/Helpers.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace reference
{
SimpleTensor<int32_t> indirect_conv2d_addr_precalculation(const TensorShape &shape_conv_src, const TensorShape &shape_conv_wei, const TensorShape &shape_conv_dst, const TensorShape &shape_dst,
                                                          const PadStrideInfo &conv_info)
{
    SimpleTensor<int32_t> out{ shape_dst, DataType::S32 };

    constexpr unsigned int width_idx = 1;
    constexpr unsigned int heigh_idx = 2;

    const int src_conv_width  = static_cast<int32_t>(shape_conv_src[width_idx]); // NHWC
    const int src_conv_height = static_cast<int32_t>(shape_conv_src[heigh_idx]); // NHWC
    const int dst_conv_width  = static_cast<int32_t>(shape_conv_dst[width_idx]); // NHWC
    const int wei_conv_width  = static_cast<int32_t>(shape_conv_wei[width_idx]); // NHWC
    const int wei_conv_height = static_cast<int32_t>(shape_conv_wei[heigh_idx]); // NHWC
    const int dst_width       = static_cast<int32_t>(shape_dst[0]);
    const int dst_height      = static_cast<int32_t>(shape_dst[1]);
    const int dst_batch       = static_cast<int32_t>(shape_dst[2]);
    const int ks              = wei_conv_width * wei_conv_height;
    const int stride_x        = static_cast<int32_t>(conv_info.stride().first);
    const int stride_y        = static_cast<int32_t>(conv_info.stride().second);
    const int pad_left        = static_cast<int32_t>(conv_info.pad_left());
    const int pad_top         = static_cast<int32_t>(conv_info.pad_top());

    const int m0 = dst_width / ks;

    for(int z = 0; z < dst_batch; ++z)
    {
        for(int y = 0; y < dst_height; ++y)
        {
            const int mout = y * m0;
            for(int ki = 0; ki < ks; ++ki)
            {
                const int xk = ki % wei_conv_width;
                const int yk = ki / wei_conv_width;
                for(int mi = 0; mi < m0; ++mi)
                {
                    int xi = ((mout + mi) % dst_conv_width) * stride_x;
                    int yi = ((mout + mi) / dst_conv_width) * stride_y;
                    xi -= pad_left;
                    yi -= pad_top;
                    const int x_s = xi + xk;
                    const int y_s = yi + yk;
                    int       my  = x_s + y_s * src_conv_width;
                    my            = my + z * src_conv_width * src_conv_height;
                    my            = x_s >= 0 ? my : -1;
                    my            = x_s < src_conv_width ? my : -1;
                    my            = y_s >= 0 ? my : -1;
                    my            = y_s < src_conv_height ? my : -1;

                    const unsigned int addr_out = mi + ki * m0 + y * (dst_width) + z * (dst_width * dst_height);
                    out[addr_out]               = my;
                }
            }
        }
    }

    return out;
}
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute