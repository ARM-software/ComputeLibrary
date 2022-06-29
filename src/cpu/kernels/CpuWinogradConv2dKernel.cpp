/*
 * Copyright (c) 2017-2022 Arm Limited.
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

#include "src/cpu/kernels/CpuWinogradConv2dKernel.h"

namespace arm_compute
{
namespace cpu
{
CpuWinogradConv2dTransformInputKernel::CpuWinogradConv2dTransformInputKernel(arm_conv::winograd::WinogradImpl &w_impl, arm_conv::ConvolutionArgs &_c_args, uint32_t nthreads)
    : _winograd_impl{ w_impl }, _conv_args{ _c_args }, _nthreads{ nthreads }
{
}

void CpuWinogradConv2dTransformInputKernel::run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(window);
    const ITensor *input_nhwc               = tensors.get_const_tensor(TensorType::ACL_SRC);
    const ITensor *winograd_input_transform = tensors.get_const_tensor(TensorType::ACL_DST);
    const ITensor *workspace                = tensors.get_const_tensor(TensorType::ACL_INT);

    const unsigned int width_idx             = 1;
    const unsigned int height_idx            = 2;
    const unsigned int batch_idx             = 3;
    int                element_size_in_bytes = input_nhwc->info()->element_size();
    const auto         src_strides           = input_nhwc->info()->strides_in_bytes();

    const size_t input_row_stride   = src_strides[height_idx] / element_size_in_bytes;
    const size_t input_col_stride   = src_strides[width_idx] / element_size_in_bytes;
    const size_t input_batch_stride = src_strides[batch_idx] / element_size_in_bytes;
    const auto   input_nhwc_ptr     = reinterpret_cast<const void *>(input_nhwc->buffer() + input_nhwc->info()->offset_first_element_in_bytes());
    auto         win_transf_ptr     = reinterpret_cast<void *>(winograd_input_transform->buffer() + winograd_input_transform->info()->offset_first_element_in_bytes());

    _winograd_impl.input_transform->execute(
        _conv_args,
        input_nhwc_ptr,
        input_batch_stride,
        input_row_stride,
        input_col_stride,
        win_transf_ptr,
        _winograd_impl.winograd_spec,
        workspace->buffer(),
        info.thread_id,
        _nthreads);
}

CpuWinogradConv2dTransformOutputKernel::CpuWinogradConv2dTransformOutputKernel(arm_conv::winograd::WinogradImpl &w_impl, arm_conv::ConvolutionArgs &_c_args, uint32_t nthreads)
    : _winograd_impl{ w_impl }, _conv_args{ _c_args }, _nthreads{ nthreads }
{
}

// Inherited methods overridden:
void CpuWinogradConv2dTransformOutputKernel::run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(window);
    const ITensor *dst_nhwc                  = tensors.get_const_tensor(TensorType::ACL_DST);
    const ITensor *winograd_output_transform = tensors.get_const_tensor(TensorType::ACL_SRC_0);
    const ITensor *biases                    = tensors.get_const_tensor(TensorType::ACL_SRC_1);
    const ITensor *workspace                 = tensors.get_tensor(TensorType::ACL_INT);

    const unsigned int width_idx             = 1;
    const unsigned int height_idx            = 2;
    const unsigned int batch_idx             = 3;
    const int          element_size_in_bytes = dst_nhwc->info()->element_size();
    const auto         dst_strides           = dst_nhwc->info()->strides_in_bytes();

    const size_t out_row_stride   = dst_strides[height_idx] / element_size_in_bytes;
    const size_t out_col_stride   = dst_strides[width_idx] / element_size_in_bytes;
    const size_t out_batch_stride = dst_strides[batch_idx] / element_size_in_bytes;
    const auto   wout_transf_ptr  = reinterpret_cast<const void *>(winograd_output_transform->buffer() + winograd_output_transform->info()->offset_first_element_in_bytes());
    auto         dst_nhwc_ptr     = reinterpret_cast<void *>(dst_nhwc->buffer() + dst_nhwc->info()->offset_first_element_in_bytes());
    void        *biases_data_ptr  = nullptr;
    if(biases != nullptr)
    {
        biases_data_ptr = reinterpret_cast<void *>(biases->buffer() + biases->info()->offset_first_element_in_bytes());
    }

    // Output transform
    _winograd_impl.output_transform->execute(
        _conv_args,
        wout_transf_ptr,
        _winograd_impl.winograd_spec,
        biases_data_ptr,
        dst_nhwc_ptr,
        out_batch_stride,
        out_row_stride,
        out_col_stride,
        workspace->buffer(),
        info.thread_id,
        _nthreads);
}

} // namespace cpu
} // namespace arm_compute