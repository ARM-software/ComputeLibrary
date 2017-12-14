/*
 * Copyright (c) 2017 ARM Limited.
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
#ifndef ARM_COMPUTE_TEST_FIXED_POINT_NEON_TARGET
#define ARM_COMPUTE_TEST_FIXED_POINT_NEON_TARGET

#include "arm_compute/core/NEON/NEFixedPoint.h"

#include "tests/Globals.h"
#include "tests/Types.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
template <typename TensorType, typename AccessorType, typename T>
void compute_target_impl(const TensorShape &shape, DataType dt, FixedPointOp op, int fixed_point_position, TensorType &src, TensorType &dst)
{
    Window window;

    switch(dt)
    {
        case DataType::QS8:
        {
            constexpr unsigned int num_elems_processed_per_iteration = 16;
            window                                                   = calculate_max_window(*src.info(), Steps(num_elems_processed_per_iteration));
            AccessWindowHorizontal input_access(src.info(), 0, num_elems_processed_per_iteration);
            AccessWindowHorizontal output_access(dst.info(), 0, num_elems_processed_per_iteration);
            update_window_and_padding(window, input_access, output_access);
            break;
        }
        case DataType::QS16:
        {
            constexpr unsigned int num_elems_processed_per_iteration = 8;
            window                                                   = calculate_max_window(*src.info(), Steps(num_elems_processed_per_iteration));
            AccessWindowHorizontal input_access(src.info(), 0, num_elems_processed_per_iteration);
            AccessWindowHorizontal output_access(dst.info(), 0, num_elems_processed_per_iteration);
            update_window_and_padding(window, input_access, output_access);
            break;
        }
        default:
            ARM_COMPUTE_ERROR("Not Supported");
            break;
    }

    int min;
    int max;
    switch(op)
    {
        case FixedPointOp::EXP:
        {
            // Fill tensors. Keep the range between [-1.0, 1.0) so the result won't
            // overflow.
            min = -(1 << (fixed_point_position - 1));
            max = (1 << (fixed_point_position - 1));
            break;
        }
        case FixedPointOp::INV_SQRT:
        {
            if(dt == DataType::QS8)
            {
                // Fill tensors. Keep the range between [1, 127).
                min = 1;
                max = 127;
            }
            else
            {
                // Fill tensors. Keep the range between [1, 0x7FFF)
                min = 1;
                max = 0x7FFF;
            }
            break;
        }
        case FixedPointOp::LOG:
        {
            if(dt == DataType::QS8)
            {
                // Fill tensors. Keep the range between [(1 << (fixed_point_position - 1), 63) so the result won't
                // overflow. E.g. for Q2.5 ln(0.001) = -6.9, which cannot be represented.
                min = (1 << (fixed_point_position - 1));
                max = 0x3F;
            }
            else
            {
                // Fill tensors. Keep the range between [(1 << (fixed_point_position - 1), 0x3FFF) so the result won't
                // overflow.
                min = (1 << (fixed_point_position - 1));
                max = 0x3FFF;
            }
            break;
        }
        case FixedPointOp::RECIPROCAL:
        {
            if(dt == DataType::QS8)
            {
                // Fill tensors. Keep the range between [15, 100) so the result won't
                // overflow. E.g. for Q2.5 reciprocal(0.001) = 1000, which cannot be represented.
                min = 15;
                max = 0x7F;
            }
            else
            {
                // Fill tensors. Keep the range between [15, 0x7FFF) so the result won't
                // overflow.
                min = 15;
                max = 0x7FFF;
            }
            break;
        }
        default:
            ARM_COMPUTE_ERROR("Not Supported");
            break;
    }

    std::uniform_int_distribution<> distribution(min, max);
    library->fill(AccessorType(src), distribution, 0);

    Iterator input(&src, window);
    Iterator output(&dst, window);

    const auto loop_function = [&](const Coordinates & id)
    {
        switch(dt)
        {
            case DataType::QS8:
            {
                const qint8x16_t qs8in = vld1q_s8(reinterpret_cast<const qint8_t *>(input.ptr()));
                switch(op)
                {
                    case FixedPointOp::EXP:
                    {
                        // Use saturated exp
                        vst1q_s8(reinterpret_cast<qint8_t *>(output.ptr()), vqexpq_qs8(qs8in, fixed_point_position));
                        break;
                    }
                    case FixedPointOp::INV_SQRT:
                    {
                        vst1q_s8(reinterpret_cast<qint8_t *>(output.ptr()), vqinvsqrtq_qs8(qs8in, fixed_point_position));
                        break;
                    }
                    case FixedPointOp::LOG:
                    {
                        vst1q_s8(reinterpret_cast<qint8_t *>(output.ptr()), vlogq_qs8(qs8in, fixed_point_position));
                        break;
                    }
                    case FixedPointOp::RECIPROCAL:
                    {
                        vst1q_s8(reinterpret_cast<qint8_t *>(output.ptr()), vrecipq_qs8(qs8in, fixed_point_position));
                        break;
                    }
                    default:
                        ARM_COMPUTE_ERROR("Not Supported");
                        break;
                }
                break;
            }
            case DataType::QS16:
            {
                const qint16x8_t qs16in = vld1q_qs16(reinterpret_cast<const qint16_t *>(input.ptr()));
                switch(op)
                {
                    case FixedPointOp::EXP:
                    {
                        // Use saturated exp
                        vst1q_qs16(reinterpret_cast<qint16_t *>(output.ptr()), vqexpq_qs16(qs16in, fixed_point_position));
                        break;
                    }
                    case FixedPointOp::INV_SQRT:
                    {
                        vst1q_qs16(reinterpret_cast<qint16_t *>(output.ptr()), vqinvsqrtq_qs16(qs16in, fixed_point_position));
                        break;
                    }
                    case FixedPointOp::LOG:
                    {
                        vst1q_qs16(reinterpret_cast<qint16_t *>(output.ptr()), vlogq_qs16(qs16in, fixed_point_position));
                        break;
                    }
                    case FixedPointOp::RECIPROCAL:
                    {
                        vst1q_qs16(reinterpret_cast<qint16_t *>(output.ptr()), vqrecipq_qs16(qs16in, fixed_point_position));
                        break;
                    }
                    default:
                        ARM_COMPUTE_ERROR("Not Supported");
                        break;
                }
                break;
            }
            default:
                ARM_COMPUTE_ERROR("Not Supported");
                break;
        }
    };

    execute_window_loop(window, loop_function, input, output);
}
} // namespace
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_FIXED_POINT_NEON_TARGET */
