/*
 * Copyright (c) 2018-2019 ARM Limited.
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
#include "arm_compute/core/NEON/kernels/NERangeKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/IAccessWindow.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/NEON/NEAsymm.h"
#include "arm_compute/core/NEON/wrapper/wrapper.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Validate.h"

#include "arm_compute/core/Utils.h"

namespace arm_compute
{
namespace
{
template <typename T>
void range_function(ITensor *output, float start, float step, const Window &window)
{
    const unsigned int num_elems_processed_per_iteration = 16 / sizeof(T);
    /** NEON vector tag type. */
    using ExactTagType = typename wrapper::traits::neon_bitvector<T, wrapper::traits::BitWidth::W128>::tag_type;

    const auto step_vec  = wrapper::vdup_n(static_cast<T>(step), ExactTagType{});
    const auto start_vec = wrapper::vdup_n(static_cast<T>(start), ExactTagType{});
    auto       id_vec    = wrapper::vdup_n(static_cast<T>(0.f), ExactTagType{});

    Iterator output_it(output, window);
    execute_window_loop(window, [&](const Coordinates & id)
    {
        for(unsigned int count = 0; count < num_elems_processed_per_iteration; ++count)
        {
            id_vec = wrapper::vsetlane(static_cast<T>(id.x() + count), id_vec, count);
        }
        // start + step * id
        const auto res_vec = wrapper::vmla(start_vec, id_vec, step_vec);
        const auto out_ptr = reinterpret_cast<T *>(output_it.ptr());
        wrapper::vstore(out_ptr, res_vec);
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

std::pair<Status, Window> validate_and_configure_window(ITensorInfo &output, const float start, const float end, const float step)
{
    const unsigned int num_elems_processed_per_iteration = 16 / output.element_size();

    // Auto initialize output if not initialized
    auto_init_if_empty(output, TensorShape(num_of_elements_in_range(start, end, step)), 1, output.data_type(), output.quantization_info());

    // Configure kernel window
    Window                 win = calculate_max_window(output, Steps(num_elems_processed_per_iteration));
    AccessWindowHorizontal output_access(&output, 0, num_elems_processed_per_iteration);
    bool                   window_changed = update_window_and_padding(win, output_access);
    output_access.set_valid_region(win, ValidRegion(Coordinates(), TensorShape(num_of_elements_in_range(start, end, step))));
    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win);
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

    // Configure kernel window
    auto win_config = validate_and_configure_window(*(output->info()), start, end, step);
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);

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

    INEKernel::configure(win_config.second);
}

Status NERangeKernel::validate(const ITensorInfo *output, float start, float end, float step)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(output);

    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(*output, start, end, step));
    ARM_COMPUTE_RETURN_ON_ERROR((validate_and_configure_window(*(output->clone()), start, end, step)).first);

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