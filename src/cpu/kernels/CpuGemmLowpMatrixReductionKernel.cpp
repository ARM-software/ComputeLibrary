/*
 * Copyright (c) 2017-2021 Arm Limited.
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
#include "src/cpu/kernels/CpuGemmLowpMatrixReductionKernel.h"

#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/KernelDescriptors.h"
#include "arm_compute/core/TensorInfo.h"

#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"
#include "src/core/NEON/wrapper/wrapper.h"

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
namespace
{
Status validate_arguments_matrix_a_reduction(const ITensorInfo                 *src,
                                             const ITensorInfo                 *dst,
                                             const GEMMLowpReductionKernelInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src, dst);
    ARM_COMPUTE_ERROR_ON_MSG(info.is_reshaped == true, "Not supported");
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src, 1, DataType::QASYMM8, DataType::QASYMM8_SIGNED,
                                                         DataType::QSYMM8, DataType::QSYMM8_PER_CHANNEL);

    if (dst->total_size() > 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(dst, 1, DataType::S32);
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(
            dst->dimension(0) != src->dimension(1),
            "Output vector must have length equal to the number of rows of the input matrix");
    }
    return Status{};
}
Status validate_arguments_matrix_b_reduction(const ITensorInfo                 *src,
                                             const ITensorInfo                 *dst,
                                             const GEMMLowpReductionKernelInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src, dst);
    ARM_COMPUTE_ERROR_ON_MSG(info.is_reshaped == true, "Not supported");
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src, 1, DataType::QASYMM8, DataType::QASYMM8_SIGNED,
                                                         DataType::QSYMM8, DataType::QSYMM8_PER_CHANNEL);

    if (dst->total_size() > 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(dst, 1, DataType::S32);
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(
            dst->dimension(0) != src->dimension(0),
            "Output vector must have length equal to the number of columns of the input matrix");
    }
    return Status{};
}
} // namespace

void CpuGemmLowpMatrixAReductionKernel::configure(const ITensorInfo                 *src,
                                                  ITensorInfo                       *dst,
                                                  const GEMMLowpReductionKernelInfo &info)
{
    // Perform validate step
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, dst);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments_matrix_a_reduction(src, dst, info));
    _k             = info.k;
    _scalar        = info.scalar;
    _mul_by_scalar = info.mul_by_scalar;

    switch (src->data_type())
    {
        case DataType::QASYMM8:
            _func = &CpuGemmLowpMatrixAReductionKernel::run_internal<uint8_t>;
            break;
        case DataType::QASYMM8_SIGNED:
        case DataType::QSYMM8:
        case DataType::QSYMM8_PER_CHANNEL:
            _func = &CpuGemmLowpMatrixAReductionKernel::run_internal<int8_t>;
            break;
        default:
            ARM_COMPUTE_ERROR("Unsupported data type");
    }

    // Output auto initialization if not yet initialized
    auto_init_if_empty(*dst, TensorShape(src->dimension(1)), 1, DataType::S32);

    Window win = calculate_max_window(*dst, Steps(1));
    ICpuKernel::configure(win);
}

Status CpuGemmLowpMatrixAReductionKernel::validate(const ITensorInfo                 *src,
                                                   const ITensorInfo                 *dst,
                                                   const GEMMLowpReductionKernelInfo &info)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments_matrix_a_reduction(src, dst, info));
    return Status{};
}

template <typename T>
void CpuGemmLowpMatrixAReductionKernel::run_internal(const ITensor             *src,
                                                     ITensor                   *dst,
                                                     const arm_compute::Window &window)
{
    // Intermediate and final accumulator types
    using TIAcc = wrapper::traits::promote_t<T>;
    using TAcc  = wrapper::traits::promote_t<TIAcc>;

    Window collapsed_window = window.collapse_if_possible(IKernel::window(), Window::DimY);

    Window win_input(collapsed_window);
    win_input.set(Window::DimX, Window::Dimension(0, 0, 0));
    win_input.set(Window::DimY, Window::Dimension(0, 0, 0));
    win_input.set(Window::DimZ, Window::Dimension(0, 0, 0));

    Iterator in(src, win_input);
    Iterator out(dst, collapsed_window);

    execute_window_loop(
        collapsed_window,
        [&](const Coordinates &id)
        {
            auto vsum_row = wrapper::vdup_n(static_cast<TAcc>(0), wrapper::traits::vector_128_tag{});
            TAcc sum_row  = 0;

            const T *matrix_a = reinterpret_cast<const T *>(
                (in.ptr() + id.x() * src->info()->strides_in_bytes()[1] + id.y() * src->info()->strides_in_bytes()[2]));

#if __arm__
            asm volatile("PLD [%0, #128*4]" ::"r"(matrix_a));
#endif /* __arm__ */

            int i = 0;
            // This for loop performs 16 accumulations
            for (; i <= (_k - 16); i += 16)
            {
                const auto a0_d8 = wrapper::vloadq(matrix_a + i);

                // Partial accumulations in U16
                const auto tmp_sum0 = wrapper::vaddl(wrapper::vgetlow(a0_d8), wrapper::vgethigh(a0_d8));

                // Accumulate to U32
                vsum_row = wrapper::vadd(vsum_row, wrapper::vpaddl(tmp_sum0));
            }

            // This for loop performs the leftover accumulations
            for (; i < _k; ++i)
            {
                sum_row += static_cast<TAcc>(matrix_a[i]);
            }

#if defined(__aarch64__)
            // Reduction operation available on 64 bit architectures only
            sum_row += wrapper::vaddv(vsum_row);
#else  // __aarch64__
            auto tmp = wrapper::vpadd(wrapper::vgethigh(vsum_row), wrapper::vgetlow(vsum_row));
            tmp      = wrapper::vpadd(tmp, tmp);

            sum_row += wrapper::vgetlane(tmp, 0);
#endif // __aarch64__

            // Multiply by scalar if necessary
            if (_mul_by_scalar)
            {
                sum_row *= _scalar;
            }

            *(reinterpret_cast<int *>(out.ptr())) = static_cast<int32_t>(sum_row);
        },
        in, out);
}

void CpuGemmLowpMatrixAReductionKernel::run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICpuKernel::window(), window);

    auto src = tensors.get_const_tensor(TensorType::ACL_SRC);
    auto dst = tensors.get_tensor(TensorType::ACL_DST);

    (this->*_func)(src, dst, window);
}

const char *CpuGemmLowpMatrixAReductionKernel::name() const
{
    return "CpuGemmLowpMatrixAReductionKernel";
}

void CpuGemmLowpMatrixBReductionKernel::configure(const ITensorInfo                 *src,
                                                  ITensorInfo                       *dst,
                                                  const GEMMLowpReductionKernelInfo &info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, dst);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments_matrix_b_reduction(src, dst, info));

    _k             = info.k;
    _scalar        = info.scalar;
    _mul_by_scalar = info.mul_by_scalar;

    // Configure kernel window
    constexpr unsigned int num_elems_processed_per_iteration = 16;

    switch (src->data_type())
    {
        case DataType::QASYMM8:
            _func = &CpuGemmLowpMatrixBReductionKernel::run_internal<uint8_t>;
            break;
        case DataType::QASYMM8_SIGNED:
        case DataType::QSYMM8:
        case DataType::QSYMM8_PER_CHANNEL:
            _func = &CpuGemmLowpMatrixBReductionKernel::run_internal<int8_t>;
            break;
        default:
            ARM_COMPUTE_ERROR("Unsupported data type");
    }

    // Output auto initialization if not yet initialized
    auto_init_if_empty(*dst, TensorShape(src->dimension(0)), 1, DataType::S32);

    // Configure kernel window
    Window win = calculate_max_window_horizontal(*dst, Steps(num_elems_processed_per_iteration));
    ICpuKernel::configure(win);
}

Status CpuGemmLowpMatrixBReductionKernel::validate(const ITensorInfo                 *src,
                                                   const ITensorInfo                 *dst,
                                                   const GEMMLowpReductionKernelInfo &info)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments_matrix_b_reduction(src, dst, info));
    return Status{};
}

template <typename T>
void CpuGemmLowpMatrixBReductionKernel::run_internal(const ITensor    *src,
                                                     ITensor          *dst,
                                                     const Window     &window,
                                                     const ThreadInfo &info)
{
    // Intermediate and final accumulator types
    using TIAcc = wrapper::traits::promote_t<T>;
    using TAcc  = wrapper::traits::promote_t<TIAcc>;

    Window     collapsed_window = window.collapse_if_possible(IKernel::window(), Window::DimY);
    const auto vec_scalar       = wrapper::vdup_n(static_cast<TAcc>(_scalar), wrapper::traits::vector_128_tag{});

    const auto width_matrix_b = static_cast<int>(src->info()->dimension(0));
    const auto in_b_stride    = static_cast<int>(src->info()->strides_in_bytes()[1]);

    // The implementation computes 16 elements per iteration
    const int window_start_x = 16 * info.thread_id;
    const int window_step_x  = 16 * info.num_threads;
    // Make sure (window_end_x - window_start_x) is a multiple of window_step_x
    const int window_end_x = ceil_to_multiple(width_matrix_b - window_start_x, window_step_x) + window_start_x;

    Window win_out(collapsed_window);
    win_out.set(Window::DimX, Window::Dimension(window_start_x, window_end_x, window_step_x));

    Window win_in(win_out);
    win_in.set(Window::DimY, Window::Dimension(0, 0, 0));
    win_in.set(Window::DimZ, Window::Dimension(0, 0, 0));

    Iterator inb(src, win_in);
    Iterator out(dst, win_out);

    execute_window_loop(
        win_out,
        [&](const Coordinates &id)
        {
            if (id.x() > width_matrix_b)
            {
                return;
            }

            // Note: Since the input is unsigned char, we can safely use unsigned int for the accumulation
            typename wrapper::traits::neon_bitvector<TAcc, wrapper::traits::BitWidth::W128>::type sum_col[4] = {
                wrapper::vdup_n(static_cast<TAcc>(0), wrapper::traits::vector_128_tag{}),
                wrapper::vdup_n(static_cast<TAcc>(0), wrapper::traits::vector_128_tag{}),
                wrapper::vdup_n(static_cast<TAcc>(0), wrapper::traits::vector_128_tag{}),
                wrapper::vdup_n(static_cast<TAcc>(0), wrapper::traits::vector_128_tag{})};

            const auto *matrix_b = reinterpret_cast<const T *>(inb.ptr() + id.y() * src->info()->strides_in_bytes()[2]);

#if __arm__
            asm volatile("PLD [%0, #128*4]" ::"r"(matrix_b));
            asm volatile("PLD [%0, #128*4]" ::"r"(matrix_b + in_b_stride));
#endif /* __arm__ */

            int i = 0;
            // This for loop performs 4 accumulations
            for (; i <= (_k - 4); i += 4)
            {
                const auto b0_u8 = wrapper::vloadq(matrix_b + 0 * in_b_stride);
                const auto b1_u8 = wrapper::vloadq(matrix_b + 1 * in_b_stride);
                const auto b2_u8 = wrapper::vloadq(matrix_b + 2 * in_b_stride);
                const auto b3_u8 = wrapper::vloadq(matrix_b + 3 * in_b_stride);

#if __arm__
                asm volatile("PLD [%0, #128*1]" ::"r"(matrix_b + 1 * in_b_stride));
                asm volatile("PLD [%0, #128*1]" ::"r"(matrix_b + 2 * in_b_stride));
                asm volatile("PLD [%0, #128*1]" ::"r"(matrix_b + 3 * in_b_stride));
                asm volatile("PLD [%0, #128*1]" ::"r"(matrix_b + 4 * in_b_stride));
#endif /* __arm__ */

                // Partial accumulation in 16bit
                typename wrapper::traits::neon_bitvector<TIAcc, wrapper::traits::BitWidth::W128>::type tmp_sum[2] = {
                    wrapper::vdup_n(static_cast<TIAcc>(0), wrapper::traits::vector_128_tag{}),
                    wrapper::vdup_n(static_cast<TIAcc>(0), wrapper::traits::vector_128_tag{})};

                tmp_sum[0] = wrapper::vaddw(tmp_sum[0], wrapper::vgetlow(b1_u8));
                tmp_sum[0] = wrapper::vaddw(tmp_sum[0], wrapper::vgetlow(b0_u8));
                tmp_sum[0] = wrapper::vaddw(tmp_sum[0], wrapper::vgetlow(b2_u8));
                tmp_sum[0] = wrapper::vaddw(tmp_sum[0], wrapper::vgetlow(b3_u8));
                tmp_sum[1] = wrapper::vaddw(tmp_sum[1], wrapper::vgethigh(b0_u8));
                tmp_sum[1] = wrapper::vaddw(tmp_sum[1], wrapper::vgethigh(b1_u8));
                tmp_sum[1] = wrapper::vaddw(tmp_sum[1], wrapper::vgethigh(b2_u8));
                tmp_sum[1] = wrapper::vaddw(tmp_sum[1], wrapper::vgethigh(b3_u8));

                // Accumulate to 32bit
                sum_col[0] = wrapper::vaddw(sum_col[0], wrapper::vgetlow(tmp_sum[0]));
                sum_col[1] = wrapper::vaddw(sum_col[1], wrapper::vgethigh(tmp_sum[0]));
                sum_col[2] = wrapper::vaddw(sum_col[2], wrapper::vgetlow(tmp_sum[1]));
                sum_col[3] = wrapper::vaddw(sum_col[3], wrapper::vgethigh(tmp_sum[1]));

                matrix_b += 4 * in_b_stride;
            }

            // This for loop perfoms the leftover accumulations
            for (; i < _k; ++i)
            {
                const auto b0_b8 = wrapper::vloadq(matrix_b + 0 * in_b_stride);

                // Convert S8 to S16
                const typename wrapper::traits::neon_bitvector<TIAcc, wrapper::traits::BitWidth::W128>::type b0_b16[2]{
                    wrapper::vmovl(wrapper::vgetlow(b0_b8)), wrapper::vmovl(wrapper::vgethigh(b0_b8))};

                // Accumulate to 32bit
                sum_col[0] = wrapper::vaddw(sum_col[0], wrapper::vgetlow(b0_b16[0]));
                sum_col[1] = wrapper::vaddw(sum_col[1], wrapper::vgethigh(b0_b16[0]));
                sum_col[2] = wrapper::vaddw(sum_col[2], wrapper::vgetlow(b0_b16[1]));
                sum_col[3] = wrapper::vaddw(sum_col[3], wrapper::vgethigh(b0_b16[1]));

                matrix_b += in_b_stride;
            }

            // Multiply by scalar if necessary
            if (_mul_by_scalar)
            {
                sum_col[0] = wrapper::vmul(sum_col[0], vec_scalar);
                sum_col[1] = wrapper::vmul(sum_col[1], vec_scalar);
                sum_col[2] = wrapper::vmul(sum_col[2], vec_scalar);
                sum_col[3] = wrapper::vmul(sum_col[3], vec_scalar);
            }

            auto vector_sum_col = reinterpret_cast<int32_t *>(out.ptr());
            if (id.x() + 16 < width_matrix_b)
            {
                wrapper::vstore(vector_sum_col + 0, wrapper::vreinterpret(sum_col[0]));
                wrapper::vstore(vector_sum_col + 4, wrapper::vreinterpret(sum_col[1]));
                wrapper::vstore(vector_sum_col + 8, wrapper::vreinterpret(sum_col[2]));
                wrapper::vstore(vector_sum_col + 12, wrapper::vreinterpret(sum_col[3]));
            }
            else
            {
                auto left_over = width_matrix_b - id.x();
                for (auto k = 0; k < 4 && left_over; ++k)
                {
                    for (auto j = 0; j < 4 && left_over; ++j, --left_over)
                    {
                        *(vector_sum_col + k * 4 + j) = sum_col[k][j];
                    }
                }
            }
        },
        inb, out);
}

void CpuGemmLowpMatrixBReductionKernel::run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICpuKernel::window(), window);

    auto src = tensors.get_const_tensor(TensorType::ACL_SRC);
    auto dst = tensors.get_tensor(TensorType::ACL_DST);

    (this->*_func)(src, dst, window, info);
}

const char *CpuGemmLowpMatrixBReductionKernel::name() const
{
    return "CpuGemmLowpMatrixBReductionKernel";
}
} // namespace kernels
} // namespace cpu
} // namespace arm_compute
