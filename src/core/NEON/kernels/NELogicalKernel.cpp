/*
 * Copyright (c) 2020 Arm Limited.
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
#include "src/core/NEON/kernels/NELogicalKernel.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Validate.h"
#include "src/core/common/Validate.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"

#include <arm_neon.h>

namespace arm_compute
{
namespace kernels
{
namespace
{
static const uint8x8_t  c0_x8     = vdup_n_u8(0);
static const uint8x16_t c0_x16    = vdupq_n_u8(0);
static const uint8x8_t  c1_x8     = vdup_n_u8(1);
static const uint8x16_t c1_x16    = vdupq_n_u8(1);
static const int        step      = 16;
static const int        half_step = step / 2;

void neon_logical_and(const uint8_t *src0, const uint8_t *src1, uint8_t *dst, int len)
{
    ARM_COMPUTE_ASSERT_NOT_NULLPTR(src0);
    ARM_COMPUTE_ASSERT_NOT_NULLPTR(src1);
    ARM_COMPUTE_ASSERT_NOT_NULLPTR(dst);
    ARM_COMPUTE_ASSERT(len >= 0);

    for(; len >= step; len -= step)
    {
        vst1q_u8(dst, vandq_u8(vminq_u8(vld1q_u8(src0), c1_x16), vminq_u8(vld1q_u8(src1), c1_x16)));
        src0 += step;
        src1 += step;
        dst += step;
    }

    for(; len >= half_step; len -= half_step)
    {
        vst1_u8(dst, vand_u8(vmin_u8(vld1_u8(src0), c1_x8), vmin_u8(vld1_u8(src1), c1_x8)));
        src0 += half_step;
        src1 += half_step;
        dst += half_step;
    }

    for(; len > 0; --len)
    {
        *dst = (*src0) && (*src1);
        ++src0;
        ++src1;
        ++dst;
    }
}

void neon_logical_and_broadcast(const uint8_t *src, uint8_t broadcast_val, uint8_t *dst, int len)
{
    ARM_COMPUTE_ASSERT_NOT_NULLPTR(src);
    ARM_COMPUTE_ASSERT_NOT_NULLPTR(dst);
    ARM_COMPUTE_ASSERT(len >= 0);

    const auto broadcast_val_clamped_s   = std::min<uint8_t>(broadcast_val, 1);
    const auto broadcast_val_clamped_x16 = vdupq_n_u8(broadcast_val_clamped_s);
    const auto broadcast_val_clamped_x8  = vdup_n_u8(broadcast_val_clamped_s);

    for(; len >= step; len -= step)
    {
        vst1q_u8(dst, vandq_u8(vminq_u8(vld1q_u8(src), c1_x16), broadcast_val_clamped_x16));
        src += step;
        dst += step;
    }

    for(; len >= half_step; len -= half_step)
    {
        vst1_u8(dst, vand_u8(vmin_u8(vld1_u8(src), c1_x8), broadcast_val_clamped_x8));
        src += half_step;
        dst += half_step;
    }

    for(; len > 0; --len)
    {
        *dst = (*src) && broadcast_val_clamped_s;
        ++src;
        ++dst;
    }
}

void neon_logical_or(const uint8_t *src0, const uint8_t *src1, uint8_t *dst, int len)
{
    ARM_COMPUTE_ASSERT_NOT_NULLPTR(src0);
    ARM_COMPUTE_ASSERT_NOT_NULLPTR(src1);
    ARM_COMPUTE_ASSERT_NOT_NULLPTR(dst);
    ARM_COMPUTE_ASSERT(len >= 0);

    for(; len >= step; len -= step)
    {
        vst1q_u8(dst, vorrq_u8(vminq_u8(vld1q_u8(src0), c1_x16), vminq_u8(vld1q_u8(src1), c1_x16)));
        src0 += step;
        src1 += step;
        dst += step;
    }

    for(; len >= half_step; len -= half_step)
    {
        vst1_u8(dst, vorr_u8(vmin_u8(vld1_u8(src0), c1_x8), vmin_u8(vld1_u8(src1), c1_x8)));
        src0 += half_step;
        src1 += half_step;
        dst += half_step;
    }

    for(; len > 0; --len)
    {
        *dst = (*src0) || (*src1);
        ++src0;
        ++src1;
        ++dst;
    }
}

void neon_logical_or_broadcast(const uint8_t *src, uint8_t broadcast_val, uint8_t *dst, int len)
{
    ARM_COMPUTE_ASSERT_NOT_NULLPTR(src);
    ARM_COMPUTE_ASSERT_NOT_NULLPTR(dst);
    ARM_COMPUTE_ASSERT(len >= 0);

    const auto broadcast_val_clamped_s   = std::min<uint8_t>(broadcast_val, 1);
    const auto broadcast_val_clamped_x16 = vdupq_n_u8(broadcast_val_clamped_s);
    const auto broadcast_val_clamped_x8  = vdup_n_u8(broadcast_val_clamped_s);

    for(; len >= step; len -= step)
    {
        vst1q_u8(dst, vorrq_u8(vminq_u8(vld1q_u8(src), c1_x16), broadcast_val_clamped_x16));
        src += step;
        dst += step;
    }

    for(; len >= half_step; len -= half_step)
    {
        vst1_u8(dst, vorr_u8(vmin_u8(vld1_u8(src), c1_x8), broadcast_val_clamped_x8));
        src += half_step;
        dst += half_step;
    }

    for(; len > 0; --len)
    {
        *dst = (*src) || broadcast_val_clamped_s;
        ++src;
        ++dst;
    }
}

void neon_logical_not(const uint8_t *src, uint8_t *dst, int len)
{
    ARM_COMPUTE_ASSERT_NOT_NULLPTR(src);
    ARM_COMPUTE_ASSERT_NOT_NULLPTR(dst);
    ARM_COMPUTE_ASSERT(len >= 0);

    for(; len >= step; len -= step)
    {
        vst1q_u8(dst, vbslq_u8(vceqq_u8(vld1q_u8(src), c0_x16), c1_x16, c0_x16));
        src += step;
        dst += step;
    }

    for(; len >= half_step; len -= half_step)
    {
        vst1_u8(dst, vbsl_u8(vceq_u8(vld1_u8(src), c0_x8), c1_x8, c0_x8));
        src += half_step;
        dst += half_step;
    }

    for(; len > 0; --len)
    {
        *dst = !(*src);
        ++src;
        ++dst;
    }
}

void run_unary(const Window &window, const ITensor *src, ITensor *dst)
{
    Window win{ window };
    win.set(Window::DimX, Window::Dimension(0, 1, 1));
    const auto len = static_cast<int>(window.x().end()) - static_cast<int>(window.x().start());

    Iterator in(src, win);
    Iterator out(dst, win);

    execute_window_loop(win, [&](const Coordinates &)
    {
        neon_logical_not(in.ptr(), out.ptr(), len);
    },
    in, out);
}

void run_binary(const Window &window, const ITensor *src0, const ITensor *src1, ITensor *dst, LogicalOperation op)
{
    Window src0_win = window.broadcast_if_dimension_le_one(src0->info()->tensor_shape());
    Window src1_win = window.broadcast_if_dimension_le_one(src1->info()->tensor_shape());

    Window win{ window };
    win.set(Window::DimX, Window::Dimension(0, 1, 1));

    const bool is_broadcast_across_x = src0->info()->tensor_shape().x() != src1->info()->tensor_shape().x();
    const auto len                   = static_cast<int>(window.x().end()) - static_cast<int>(window.x().start());

    if(is_broadcast_across_x)
    {
        using LogicalBroadcastUKernelPtr        = std::add_pointer<void(const uint8_t *, uint8_t, uint8_t *, int)>::type;
        LogicalBroadcastUKernelPtr logical_func = op == LogicalOperation::Or ? &neon_logical_or_broadcast : &neon_logical_and_broadcast;

        const bool     is_broadcast_input_1 = src1_win.x().step() == 0;
        Window         broadcast_win        = is_broadcast_input_1 ? src1_win : src0_win;
        Window         non_broadcast_win    = !is_broadcast_input_1 ? src1_win : src0_win;
        const ITensor *broadcast_tensor     = is_broadcast_input_1 ? src1 : src0;
        const ITensor *non_broadcast_tensor = !is_broadcast_input_1 ? src1 : src0;
        non_broadcast_win.set(Window::DimX, Window::Dimension(0, 1, 1));

        Iterator broadcast_in(broadcast_tensor, broadcast_win);
        Iterator non_broadcast_in(non_broadcast_tensor, non_broadcast_win);
        Iterator out(dst, win);

        execute_window_loop(win, [&](const Coordinates &)
        {
            const uint8_t broadcast_value = *broadcast_in.ptr();
            logical_func(non_broadcast_in.ptr(), broadcast_value, out.ptr(), len);

        },
        broadcast_in, non_broadcast_in, out);
    }
    else
    {
        using LogicalUKernelPtr        = std::add_pointer<void(const uint8_t *, const uint8_t *, uint8_t *, int)>::type;
        LogicalUKernelPtr logical_func = op == LogicalOperation::Or ? &neon_logical_or : &neon_logical_and;

        src0_win.set(Window::DimX, Window::Dimension(0, 1, 1));
        src1_win.set(Window::DimX, Window::Dimension(0, 1, 1));

        Iterator in0(src0, src0_win);
        Iterator in1(src1, src1_win);
        Iterator out(dst, win);
        execute_window_loop(win, [&](const Coordinates &)
        {
            logical_func(in0.ptr(), in1.ptr(), out.ptr(), len);
        },
        in0, in1, out);
    }
}
} // namespace
const char *NELogicalKernel::name() const
{
    return "NELogicalKernel";
}

void NELogicalKernel::configure(const ITensorInfo *input1, const ITensorInfo *input2, ITensorInfo *output, LogicalOperation op)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input1, output);
    ARM_COMPUTE_ERROR_THROW_ON(validate(input1, input2, output, op));

    _op = op;

    Window      win       = calculate_max_window(*input1, Steps());
    TensorShape out_shape = input1->tensor_shape();
    if(op != LogicalOperation::Not)
    {
        ARM_COMPUTE_ERROR_ON_NULLPTR(input2);
        const std::pair<TensorShape, ValidRegion> broadcast_pair = ITensorInfo::broadcast_shape_and_valid_region(*input1, *input2);
        out_shape = broadcast_pair.first;
        win       = calculate_max_window(broadcast_pair.second, Steps());
    }
    ICPPKernel::configure(win);

    // Auto initialize if empty
    set_shape_if_empty(*output, out_shape);
    set_data_type_if_unknown(*output, input1->data_type());
}

Status NELogicalKernel::validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, LogicalOperation op)
{
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input1, 1, DataType::U8);
    ARM_COMPUTE_RETURN_ERROR_ON(op == LogicalOperation::Unknown);

    TensorShape out_shape = input1->tensor_shape();
    if(op != LogicalOperation::Not)
    {
        out_shape = TensorShape::broadcast_shape(input1->tensor_shape(), input2->tensor_shape());
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(out_shape.total_size() == 0, "Inputs are not broadcast compatible");
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input1, input2);
    }

    // Checks performed when output is configured
    if((output != nullptr) && (output->total_size() != 0))
    {
        ARM_COMPUTE_RETURN_ERROR_ON(detail::have_different_dimensions(out_shape, output->tensor_shape(), 0));
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input1, output);
    }

    return Status{};
}

void NELogicalKernel::run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(tensors.empty());

    const ITensor *src0 = tensors.get_const_tensor(TensorType::ACL_SRC_0);
    const ITensor *src1 = tensors.get_const_tensor(TensorType::ACL_SRC_1);
    ITensor       *dst  = tensors.get_tensor(TensorType::ACL_DST);

    if(_op == LogicalOperation::Not)
    {
        run_unary(window, src0, dst);
    }
    else
    {
        run_binary(window, src0, src1, dst, _op);
    }
}
} // namespace kernels
} // namespace arm_compute
