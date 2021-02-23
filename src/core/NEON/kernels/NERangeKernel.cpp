/*
 * Copyright (c) 2018-2021 Arm Limited.
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
#include "src/core/NEON/kernels/NERangeKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/IAccessWindow.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Validate.h"
#include "src/core/NEON/NEAsymm.h"
#include "src/core/NEON/wrapper/wrapper.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"

#include "arm_compute/core/Utils.h"

namespace arm_compute
{
namespace
{
template <typename T>
void range_function(ITensor *output, float start, float step, const Window &window)
{
    /** Neon vector tag type. */
    using ExactTagType = typename wrapper::traits::neon_bitvector<T, wrapper::traits::BitWidth::W128>::tag_type;

    const auto step_vec  = wrapper::vdup_n(static_cast<T>(step), ExactTagType{});
    const auto start_vec = wrapper::vdup_n(static_cast<T>(start), ExactTagType{});
    auto       id_vec    = wrapper::vdup_n(static_cast<T>(0.f), ExactTagType{});

    const auto window_start_x = static_cast<int>(window.x().start());
    const auto window_end_x   = static_cast<int>(window.x().end());
    const int  window_step_x  = 16 / sizeof(T);

    Window win{ window };
    win.set(Window::DimX, Window::Dimension(0, 1, 1));
    Iterator output_it(output, win);

    execute_window_loop(win, [&](const Coordinates &)
    {
        int        x       = window_start_x;
        const auto out_ptr = reinterpret_cast<T *>(output_it.ptr());
        for(; x <= (window_end_x - window_step_x); x += window_step_x)
        {
            for(int count = 0; count < window_step_x; ++count)
            {
                id_vec = wrapper::vsetlane(static_cast<T>(x + count), id_vec, count);
            }

            // start + step * id
            const auto res_vec = wrapper::vmla(start_vec, id_vec, step_vec);
            wrapper::vstore(out_ptr + x, res_vec);
        }

        // Compute left-over elements
        for(; x < window_end_x; ++x)
        {
            const auto res = start + x * step;
            *(out_ptr + x) = res;
        }

    },
    output_it);
}

Status validate_arguments(const ITensorInfo &output, const float start, const float end, const float step)
{
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&output,
                                                         1,
                                                         DataType::U8, DataType::S8,
                                                         DataType::U16, DataType::S16,
                                                         DataType::U32, DataType::S32,
                                                         DataType::F16, DataType::F32);

    ARM_COMPUTE_RETURN_ERROR_ON_MSG((start == end), "start of the requested sequence must not be equal to the end");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(((start < end) && (step <= 0)), "step must be greater than 0 when start < end");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(((start > end) && (step >= 0)), "step must be less than 0 when start > end");

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(!check_value_range(start, output.data_type(), output.quantization_info()), "start value is outside the range of the data type");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(!check_value_range(end, output.data_type(), output.quantization_info()), "end value is outside the range of the data type");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(!check_value_range(step, output.data_type(), output.quantization_info()), "step value is outside the range of the data type");

    ARM_COMPUTE_RETURN_ERROR_ON_MSG((start == end), "start of the requested sequence must not be equal to the end");

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(output.num_dimensions() != 1, "Output has to be a 1-D tensor");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(output.tensor_shape().total_size() < num_of_elements_in_range(start, end, step), "Output tensor size is incorrect");

    return Status{};
}
} // namespace

NERangeKernel::NERangeKernel()
    : _func(nullptr), _start(0), _end(1), _step(1), _output(nullptr)
{
}

void NERangeKernel::configure(ITensor *output, float start, float end, float step)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(output);

    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(*(output->info()), start, end, step));

    // Auto initialize output if not initialized
    auto_init_if_empty(*output->info(), TensorShape(num_of_elements_in_range(start, end, step)), 1, output->info()->data_type(), output->info()->quantization_info());

    // Configure kernel window
    Window      win = calculate_max_window(*output->info(), Steps());
    Coordinates coord;
    coord.set_num_dimensions(output->info()->num_dimensions());
    output->info()->set_valid_region(ValidRegion(coord, output->info()->tensor_shape()));

    _start  = start;
    _end    = end;
    _step   = step;
    _output = output;
    switch(_output->info()->data_type())
    {
        case DataType::U8:
            _func = &range_function<uint8_t>;
            break;
        case DataType::U16:
            _func = &range_function<uint16_t>;
            break;
        case DataType::U32:
            _func = &range_function<uint32_t>;
            break;
        case DataType::S8:
            _func = &range_function<int8_t>;
            break;
        case DataType::S16:
            _func = &range_function<int16_t>;
            break;
        case DataType::S32:
            _func = &range_function<int32_t>;
            break;
        case DataType::F32:
            _func = &range_function<float>;
            break;
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        case DataType::F16:
            _func = &range_function<float16_t>;
            break;
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        default:
            ARM_COMPUTE_ERROR("Unsupported data type.");
            break;
    }

    INEKernel::configure(win);
}

Status NERangeKernel::validate(const ITensorInfo *output, float start, float end, float step)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(output);

    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(*output, start, end, step));

    return Status{};
}

void NERangeKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(_func == nullptr);

    (*_func)(_output, _start, _step, window);
}
} // namespace arm_compute
