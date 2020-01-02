/*
 * Copyright (c) 2017-2020 ARM Limited.
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
#include "arm_compute/core/NEON/kernels/NEGEMMLowpReductionKernel.h"

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/NEON/wrapper/wrapper.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include <cstddef>
#include <cstdint>

using namespace arm_compute;

namespace arm_compute
{
class Coordinates;
} // namespace arm_compute

namespace
{
Status validate_arguments_matrix_a_reduction(const ITensorInfo *input, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QASYMM8, DataType::QASYMM8_SIGNED);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::S32);

    return Status{};
}
std::pair<Status, Window> validate_and_configure_window_matrix_a_reduction(ITensorInfo *input, ITensorInfo *output, bool is_reshaped)
{
    const unsigned int num_elems_processed_per_iteration = is_reshaped ? 4 : 1;

    Window win = calculate_max_window(*output, Steps(num_elems_processed_per_iteration));

    AccessWindowStatic     input_access(input, 0, 0, ceil_to_multiple(input->dimension(0), 16), input->dimension(1));
    AccessWindowHorizontal output_access(output, 0, num_elems_processed_per_iteration);

    bool window_changed = update_window_and_padding(win, input_access, output_access);

    output_access.set_valid_region(win, ValidRegion(Coordinates(), output->tensor_shape()));

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win);
}

Status validate_arguments_matrix_b_reduction(const ITensorInfo *input, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QASYMM8, DataType::QASYMM8_SIGNED, DataType::QSYMM8_PER_CHANNEL);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::S32);

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window_matrix_b_reduction(ITensorInfo *input, ITensorInfo *output)
{
    constexpr unsigned int num_elems_processed_per_iteration = 16;

    // Configure kernel window
    Window win = calculate_max_window(*output, Steps(num_elems_processed_per_iteration));

    AccessWindowStatic     input_access(input, 0, 0, ceil_to_multiple(input->dimension(0), 16), input->dimension(1));
    AccessWindowHorizontal output_access(output, 0, num_elems_processed_per_iteration);

    bool window_changed = update_window_and_padding(win, input_access, output_access);

    output_access.set_valid_region(win, ValidRegion(Coordinates(), output->tensor_shape()));

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win);
}
} // namespace

INEGEMMLowpReductionKernel::INEGEMMLowpReductionKernel()
    : _input(), _output(), _k(0), _is_reshaped(false)
{
}

void NEGEMMLowpMatrixAReductionKernel::configure(const ITensor *mtx_a, ITensor *vector_sum_row, int32_t num_mtx_a_cols, bool is_interleaved4x4)
{
    // Perform validate step
    ARM_COMPUTE_ERROR_ON_NULLPTR(mtx_a, vector_sum_row);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments_matrix_a_reduction(mtx_a->info(), vector_sum_row->info()));

    _input       = mtx_a;
    _output      = vector_sum_row;
    _k           = num_mtx_a_cols;
    _is_reshaped = is_interleaved4x4;

    // Configure kernel window
    auto win_config = validate_and_configure_window_matrix_a_reduction(_input->info(), _output->info(), _is_reshaped);
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    INEKernel::configure(win_config.second);
}

Status NEGEMMLowpMatrixAReductionKernel::validate(const ITensorInfo *mtx_a, const ITensorInfo *vector_sum_row, int32_t num_mtx_a_cols, bool is_interleaved4x4)
{
    ARM_COMPUTE_UNUSED(num_mtx_a_cols);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments_matrix_a_reduction(mtx_a, vector_sum_row));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window_matrix_a_reduction(mtx_a->clone().get(), vector_sum_row->clone().get(), is_interleaved4x4).first);

    return Status{};
}

template <typename T>
void NEGEMMLowpMatrixAReductionKernel::run_internal(const arm_compute::Window &window)
{
    // Intermediate and final accumulator types
    using TIAcc = wrapper::traits::promote_t<T>;
    using TAcc  = wrapper::traits::promote_t<TIAcc>;

    Window collapsed_window = window.collapse_if_possible(IKernel::window(), Window::DimY);

    Window win_input(collapsed_window);
    win_input.set(Window::DimX, Window::Dimension(0, 0, 0));
    win_input.set(Window::DimY, Window::Dimension(0, 0, 0));
    win_input.set(Window::DimZ, Window::Dimension(0, 0, 0));

    Iterator in(_input, win_input);
    Iterator out(_output, collapsed_window);

    if(_is_reshaped)
    {
        execute_window_loop(collapsed_window, [&](const Coordinates & id)
        {
            // Note: Since the input is unsigned char, we can safely use unsigned int for the accumulation
            auto sum_row = wrapper::vdup_n(static_cast<TAcc>(0), wrapper::traits::vector_128_tag{});

            const T *matrix_a = reinterpret_cast<const T *>((in.ptr() + (id.x() / 4) * _input->info()->strides_in_bytes()[1] + id.y() * _input->info()->strides_in_bytes()[2]));

#if __arm__
            asm volatile("PLD [%0, #128*4]" ::"r"(matrix_a));
#endif /* __arm__ */

            int i = 0;
            // This for loop performs 4 accumulations
            for(; i <= (_k - 4); i += 4)
            {
                const auto a0_d8 = wrapper::vloadq(matrix_a + i * 4);

                // Convert 8-bit to 16-bit
                typename wrapper::traits::neon_bitvector<TIAcc, wrapper::traits::BitWidth::W64>::type a0_d16[4] =
                {
                    wrapper::vgetlow(wrapper::vmovl(wrapper::vgetlow(a0_d8))),
                    wrapper::vgethigh(wrapper::vmovl(wrapper::vgetlow(a0_d8))),
                    wrapper::vgetlow(wrapper::vmovl((wrapper::vgethigh(a0_d8)))),
                    wrapper::vgethigh(wrapper::vmovl(wrapper::vgethigh(a0_d8)))
                };

                // Accumulate to 16-bit
                a0_d16[0] = wrapper::vadd(a0_d16[0], a0_d16[1]);
                a0_d16[0] = wrapper::vadd(a0_d16[0], a0_d16[2]);
                a0_d16[0] = wrapper::vadd(a0_d16[0], a0_d16[3]);

                // Accumulate to 32-bit
                sum_row = wrapper::vaddw(sum_row, a0_d16[0]);
            }

            // This for loop performs the leftover accumulations
            for(; i < _k; ++i)
            {
                const auto a0_d8 = wrapper::vload(matrix_a + i * 4);

                // Convert U8 to U16
                const auto a0_d16 = wrapper::vgetlow(wrapper::vmovl(a0_d8));

                // Accumulate to U32
                sum_row = wrapper::vaddw(sum_row, a0_d16);
            }

            auto vector_sum_row = reinterpret_cast<int32_t *>(out.ptr());

            wrapper::vstore(vector_sum_row, wrapper::vreinterpret(sum_row));
        },
        in, out);
    }
    else // it is not reshaped
    {
        execute_window_loop(collapsed_window, [&](const Coordinates & id)
        {
            // Note: Since the input is unsigned char, we can safely use unsigned int for the accumulation
            auto vsum_row = wrapper::vdup_n(static_cast<TAcc>(0), wrapper::traits::vector_128_tag{});
            TAcc sum_row  = 0;

            const T *matrix_a = reinterpret_cast<const T *>((in.ptr() + id.x() * _input->info()->strides_in_bytes()[1] + id.y() * _input->info()->strides_in_bytes()[2]));

#if __arm__
            asm volatile("PLD [%0, #128*4]" ::"r"(matrix_a));
#endif /* __arm__ */

            int i = 0;
            // This for loop performs 16 accumulations
            for(; i <= (_k - 16); i += 16)
            {
                const auto a0_d8 = wrapper::vloadq(matrix_a + i);

                // Partial accumulations in U16
                const auto tmp_sum0 = wrapper::vaddl(wrapper::vgetlow(a0_d8), wrapper::vgethigh(a0_d8));

                // Accumulate to U32
                vsum_row = wrapper::vadd(vsum_row, wrapper::vpaddl(tmp_sum0));
            }

            // This for loop performs the leftover accumulations
            for(; i < _k; ++i)
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

            *(reinterpret_cast<int *>(out.ptr())) = static_cast<int32_t>(sum_row);
        },
        in, out);
    }
}

void NEGEMMLowpMatrixAReductionKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    switch(_input->info()->data_type())
    {
        case DataType::QASYMM8:
            run_internal<uint8_t>(window);
            break;
        case DataType::QASYMM8_SIGNED:
        case DataType::QSYMM8_PER_CHANNEL:
            run_internal<int8_t>(window);
            break;
        default:
            ARM_COMPUTE_ERROR("Unsupported data type");
    }
}

void NEGEMMLowpMatrixBReductionKernel::configure(const ITensor *mtx_b, ITensor *vector_sum_col, int32_t num_mtx_b_rows, bool is_transposed1xW)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(mtx_b, vector_sum_col);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments_matrix_b_reduction(mtx_b->info(), vector_sum_col->info()));

    _input       = mtx_b;
    _output      = vector_sum_col;
    _k           = num_mtx_b_rows;
    _is_reshaped = is_transposed1xW;

    // Configure kernel window
    auto win_config = validate_and_configure_window_matrix_b_reduction(_input->info(), _output->info());
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    INEKernel::configure(win_config.second);
}

Status NEGEMMLowpMatrixBReductionKernel::validate(const ITensorInfo *mtx_b, const ITensorInfo *vector_sum_col, int32_t num_mtx_b_rows, bool is_transposed1xW)
{
    ARM_COMPUTE_UNUSED(num_mtx_b_rows);
    ARM_COMPUTE_UNUSED(is_transposed1xW);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments_matrix_b_reduction(mtx_b, vector_sum_col));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window_matrix_b_reduction(mtx_b->clone().get(), vector_sum_col->clone().get()).first);

    return Status{};
}

template <typename T>
void NEGEMMLowpMatrixBReductionKernel::run_internal(const Window &window, const ThreadInfo &info)
{
    // Intermediate and final accumulator types
    using TIAcc = wrapper::traits::promote_t<T>;
    using TAcc  = wrapper::traits::promote_t<TIAcc>;

    Window collapsed_window = window.collapse_if_possible(IKernel::window(), Window::DimY);

    if(_is_reshaped)
    {
        Window win_input(collapsed_window);
        win_input.set(Window::DimX, Window::Dimension(0, 0, 0));
        win_input.set(Window::DimY, Window::Dimension(0, 0, 0));
        win_input.set(Window::DimZ, Window::Dimension(0, 0, 0));

        Iterator in(_input, win_input);
        Iterator out(_output, collapsed_window);

        execute_window_loop(collapsed_window, [&](const Coordinates & id)
        {
            // Note: Since the input is unsigned char, we can safely use unsigned int for the accumulation
            typename wrapper::traits::neon_bitvector<TAcc, wrapper::traits::BitWidth::W128>::type sum_col[4] =
            {
                wrapper::vdup_n(static_cast<TAcc>(0), wrapper::traits::vector_128_tag{}),
                wrapper::vdup_n(static_cast<TAcc>(0), wrapper::traits::vector_128_tag{}),
                wrapper::vdup_n(static_cast<TAcc>(0), wrapper::traits::vector_128_tag{}),
                wrapper::vdup_n(static_cast<TAcc>(0), wrapper::traits::vector_128_tag{})
            };

            const auto *matrix_b = reinterpret_cast<const T *>(in.ptr() + (id.x() / 16) * _input->info()->strides_in_bytes()[1] + id.y() * _input->info()->strides_in_bytes()[2]);

#if __arm__
            asm volatile("PLD [%0, #128*4]" ::"r"(matrix_b));
#endif /* __arm__ */

            int i = 0;
            for(; i < _k; ++i)
            {
                const auto b0_b8 = wrapper::vloadq(matrix_b + i * 16);

                // Convert 8bit to 16bit
                const typename wrapper::traits::neon_bitvector<TIAcc, wrapper::traits::BitWidth::W128>::type b0_b16[2] =
                {
                    wrapper::vmovl(wrapper::vgetlow(b0_b8)),
                    wrapper::vmovl(wrapper::vgethigh(b0_b8))
                };

                // Accumulate to U32
                sum_col[0] = wrapper::vaddw(sum_col[0], wrapper::vgetlow(b0_b16[0]));
                sum_col[1] = wrapper::vaddw(sum_col[1], wrapper::vgethigh(b0_b16[0]));
                sum_col[2] = wrapper::vaddw(sum_col[2], wrapper::vgetlow(b0_b16[1]));
                sum_col[3] = wrapper::vaddw(sum_col[3], wrapper::vgethigh(b0_b16[1]));
            }

            auto vector_sum_col = reinterpret_cast<int32_t *>(out.ptr());

            wrapper::vstore(vector_sum_col + 0, wrapper::vreinterpret(sum_col[0]));
            wrapper::vstore(vector_sum_col + 4, wrapper::vreinterpret(sum_col[1]));
            wrapper::vstore(vector_sum_col + 8, wrapper::vreinterpret(sum_col[2]));
            wrapper::vstore(vector_sum_col + 12, wrapper::vreinterpret(sum_col[3]));
        },
        in, out);
    }
    else // it is not reshaped
    {
        const auto width_matrix_b = static_cast<int>(_input->info()->dimension(0));
        const auto in_b_stride    = static_cast<int>(_input->info()->strides_in_bytes()[1]);

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

        Iterator inb(_input, win_in);
        Iterator out(_output, win_out);

        execute_window_loop(win_out, [&](const Coordinates & id)
        {
            if(id.x() > width_matrix_b)
            {
                return;
            }

            // Note: Since the input is unsigned char, we can safely use unsigned int for the accumulation
            typename wrapper::traits::neon_bitvector<TAcc, wrapper::traits::BitWidth::W128>::type sum_col[4] =
            {
                wrapper::vdup_n(static_cast<TAcc>(0), wrapper::traits::vector_128_tag{}),
                wrapper::vdup_n(static_cast<TAcc>(0), wrapper::traits::vector_128_tag{}),
                wrapper::vdup_n(static_cast<TAcc>(0), wrapper::traits::vector_128_tag{}),
                wrapper::vdup_n(static_cast<TAcc>(0), wrapper::traits::vector_128_tag{})
            };

            const auto *matrix_b = reinterpret_cast<const T *>(inb.ptr() + id.y() * _input->info()->strides_in_bytes()[2]);

#if __arm__
            asm volatile("PLD [%0, #128*4]" ::"r"(matrix_b));
            asm volatile("PLD [%0, #128*4]" ::"r"(matrix_b + in_b_stride));
#endif /* __arm__ */

            int i = 0;
            // This for loop performs 4 accumulations
            for(; i <= (_k - 4); i += 4)
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
                typename wrapper::traits::neon_bitvector<TIAcc, wrapper::traits::BitWidth::W128>::type tmp_sum[2] =
                {
                    wrapper::vdup_n(static_cast<TIAcc>(0), wrapper::traits::vector_128_tag{}),
                    wrapper::vdup_n(static_cast<TIAcc>(0), wrapper::traits::vector_128_tag{})
                };

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
            for(; i < _k; ++i)
            {
                const auto b0_b8 = wrapper::vloadq(matrix_b + 0 * in_b_stride);

                // Convert S8 to S16
                const typename wrapper::traits::neon_bitvector<TIAcc, wrapper::traits::BitWidth::W128>::type b0_b16[2]
                {
                    wrapper::vmovl(wrapper::vgetlow(b0_b8)),
                    wrapper::vmovl(wrapper::vgethigh(b0_b8))
                };

                // Accumulate to 32bit
                sum_col[0] = wrapper::vaddw(sum_col[0], wrapper::vgetlow(b0_b16[0]));
                sum_col[1] = wrapper::vaddw(sum_col[1], wrapper::vgethigh(b0_b16[0]));
                sum_col[2] = wrapper::vaddw(sum_col[2], wrapper::vgetlow(b0_b16[1]));
                sum_col[3] = wrapper::vaddw(sum_col[3], wrapper::vgethigh(b0_b16[1]));

                matrix_b += in_b_stride;
            }

            auto vector_sum_col = reinterpret_cast<int32_t *>(out.ptr());

            wrapper::vstore(vector_sum_col + 0, wrapper::vreinterpret(sum_col[0]));
            wrapper::vstore(vector_sum_col + 4, wrapper::vreinterpret(sum_col[1]));
            wrapper::vstore(vector_sum_col + 8, wrapper::vreinterpret(sum_col[2]));
            wrapper::vstore(vector_sum_col + 12, wrapper::vreinterpret(sum_col[3]));
        },
        inb, out);
    }
}

void NEGEMMLowpMatrixBReductionKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    switch(_input->info()->data_type())
    {
        case DataType::QASYMM8:
            run_internal<uint8_t>(window, info);
            break;
        case DataType::QASYMM8_SIGNED:
        case DataType::QSYMM8_PER_CHANNEL:
            run_internal<int8_t>(window, info);
            break;
        default:
            ARM_COMPUTE_ERROR("Unsupported data type");
    }
}
