/*
 * Copyright (c) 2017-2019 ARM Limited.
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
#include "arm_compute/core/NEON/kernels/NEGEMMMatrixAccumulateBiasesKernel.h"

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/CPP/Validate.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/NEON/NEFixedPoint.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include <arm_neon.h>
#include <cstddef>
#include <cstdint>

using namespace arm_compute;

namespace
{
inline Status validate_arguments(const ITensorInfo *accum, const ITensorInfo *biases)
{
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(accum);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(accum, 1, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(biases, accum);
    ARM_COMPUTE_RETURN_ERROR_ON(biases->num_dimensions() > 1);
    ARM_COMPUTE_RETURN_ERROR_ON(biases->dimension(0) != accum->dimension(0));

    return Status{};
}

inline std::pair<Status, Window> validate_and_configure_window(ITensorInfo *accum, ITensorInfo *biases)
{
    constexpr unsigned int num_elems_processed_per_iteration = 16;

    // Configure kernel window
    Window win = calculate_max_window(*accum, Steps(num_elems_processed_per_iteration));

    bool window_changed = update_window_and_padding(win,
                                                    AccessWindowHorizontal(accum, 0, num_elems_processed_per_iteration),
                                                    AccessWindowStatic(biases, 0, 0, ceil_to_multiple(biases->dimension(0), num_elems_processed_per_iteration), biases->tensor_shape().y()));

    AccessWindowHorizontal output_access(accum, 0, num_elems_processed_per_iteration);

    // Set the valid region for the accum tensor
    Coordinates coord;
    coord.set_num_dimensions(accum->num_dimensions());
    output_access.set_valid_region(win, ValidRegion(coord, accum->tensor_shape()));

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win);
}
} // namespace

NEGEMMMatrixAccumulateBiasesKernel::NEGEMMMatrixAccumulateBiasesKernel()
    : _accum(nullptr), _biases(nullptr)
{
}

void NEGEMMMatrixAccumulateBiasesKernel::configure(ITensor *accum, const ITensor *biases)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(accum, biases);

    // Perform validate step
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(accum->info(), biases->info()));

    _biases = biases;
    _accum  = accum;

    // Configure kernel window
    auto win_config = validate_and_configure_window(accum->info(), biases->info());
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    INEKernel::configure(win_config.second);
}

Status NEGEMMMatrixAccumulateBiasesKernel::validate(const ITensorInfo *accum, const ITensorInfo *biases)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(accum, biases));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(accum->clone().get(), biases->clone().get()).first);

    return Status{};
}

void NEGEMMMatrixAccumulateBiasesKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    Window win_biases;
    win_biases.set(Window::DimX, Window::Dimension(window.x().start(), window.x().end(), window.x().step()));
    win_biases.set(Window::DimY, Window::Dimension(0, 1, 1));

    Iterator in0_out(_accum, window);
    Iterator in1(_biases, win_biases);

    switch(_accum->info()->data_type())
    {
        case DataType::F32:
        {
            execute_window_loop(window, [&](const Coordinates &)
            {
                const float32x4x4_t accum  = vld4q_f32(reinterpret_cast<const float *>(in0_out.ptr()));
                const float32x4x4_t biases = vld4q_f32(reinterpret_cast<const float *>(in1.ptr()));
                const float32x4x4_t res =
                {
                    {
                        vaddq_f32(accum.val[0], biases.val[0]),
                        vaddq_f32(accum.val[1], biases.val[1]),
                        vaddq_f32(accum.val[2], biases.val[2]),
                        vaddq_f32(accum.val[3], biases.val[3])
                    }
                };

                vst4q_f32(reinterpret_cast<float *>(in0_out.ptr()), res);
            },
            in0_out, in1);
            break;
        }
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        case DataType::F16:
        {
            execute_window_loop(window, [&](const Coordinates &)
            {
                const float16x8x2_t accum  = vld2q_f16(reinterpret_cast<const float16_t *>(in0_out.ptr()));
                const float16x8x2_t biases = vld2q_f16(reinterpret_cast<const float16_t *>(in1.ptr()));
                const float16x8x2_t res =
                {
                    {
                        vaddq_f16(accum.val[0], biases.val[0]),
                        vaddq_f16(accum.val[1], biases.val[1])
                    }
                };

                vst2q_f16(reinterpret_cast<float16_t *>(in0_out.ptr()), res);
            },
            in0_out, in1);
            break;
        }
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
        default:
            ARM_COMPUTE_ERROR("Data type not supported");
            break;
    }
}
