/*
 * Copyright (c) 2018-2020 ARM Limited.
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
#include "arm_compute/core/NEON/kernels/NEUpsampleLayerKernel.h"

#include "arm_compute/core/CPP/Validate.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/core/NEON/wrapper/wrapper.h"

#include <arm_neon.h>

namespace arm_compute
{
namespace
{
template <typename T, int S>
inline T get_data_out(T data, int offset)
{
    T out{ 0 };
    for(int i = 0; i < S / 2; ++i)
    {
        out[2 * i]     = wrapper::vgetlane(data, i + offset);
        out[2 * i + 1] = wrapper::vgetlane(data, i + offset);
    }
    return out;
}

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *output, int num_elems_processed_per_iteration_x, const Size2D &info)
{
    std::pair<Status, Window> win_config;
    switch(input->data_layout())
    {
        case DataLayout::NCHW:
        {
            const int              num_elems_processed_per_iteration_x_out = num_elems_processed_per_iteration_x * info.x();
            Window                 win                                     = calculate_max_window(*output, Steps(num_elems_processed_per_iteration_x_out));
            AccessWindowRectangle  input_access(input, 0, 0, num_elems_processed_per_iteration_x, 1, 0.5f, 0.5f);
            AccessWindowHorizontal output_access(output, 0, num_elems_processed_per_iteration_x_out);
            bool                   window_changed = update_window_and_padding(win, input_access, output_access);
            output_access.set_valid_region(win, output->valid_region());

            Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
            win_config = std::make_pair(err, win);
            break;
        }
        case DataLayout::NHWC:
        {
            Window                 win = calculate_max_window(*output, Steps(num_elems_processed_per_iteration_x));
            AccessWindowHorizontal input_access(input, 0, num_elems_processed_per_iteration_x);
            AccessWindowHorizontal output_access(output, 0, num_elems_processed_per_iteration_x);
            bool                   window_changed = update_window_and_padding(win, input_access, output_access);
            output_access.set_valid_region(win, output->valid_region());

            Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
            win_config = std::make_pair(err, win);
            break;
        }
        default:
        {
            win_config = std::make_pair(ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Unsupported data layout!"), Window{});
        }
    }

    return win_config;
}
} // namespace
NEUpsampleLayerKernel::NEUpsampleLayerKernel()
    : _func(nullptr), _input(nullptr), _output(nullptr), _info(), _num_elems_processed_per_iteration_x()
{
}

Status NEUpsampleLayerKernel::validate(const ITensorInfo *input, const ITensorInfo *output, const Size2D &info, const InterpolationPolicy policy)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_UNUSED(policy);

    const DataLayout data_layout = input->data_layout();
    const int        idx_width   = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const int        idx_height  = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);

    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(input);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QASYMM8_SIGNED, DataType::QASYMM8, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(info.x() != 2 || info.y() != 2, "Only stride 2 is supported");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(policy != InterpolationPolicy::NEAREST_NEIGHBOR, "Only nearest neighbor policy supported");

    // Check output if configured
    if(output->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_LAYOUT(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON(output->dimension(idx_width) != info.x() * input->dimension(idx_width));
        ARM_COMPUTE_RETURN_ERROR_ON(output->dimension(idx_height) != info.y() * input->dimension(idx_height));
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_QUANTIZATION_INFO(input, output);
    }

    const int num_elems_processed_per_iteration_x = 16 / input->element_size();
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input->clone().get(),
                                                              output->clone().get(), num_elems_processed_per_iteration_x, info)
                                .first);
    return Status{};
}

template <typename T, int S>
void NEUpsampleLayerKernel::upsample_nchw(const arm_compute::Window &window)
{
    using VectorType = typename wrapper::traits::neon_vector<T, S>::type;

    Window window_in(window);
    window_in.set(Window::DimX, Window::Dimension(0, _input->info()->dimension(0), _num_elems_processed_per_iteration_x));

    Window window_out(window);
    window_out.set(Window::DimY, Window::Dimension(0, _output->info()->dimension(1), _info.y()));

    Iterator  input(_input, window_in);
    Iterator  output(_output, window_out);
    const int offset_y_out = _output->info()->strides_in_bytes().y() / sizeof(T);

    execute_window_loop(window_out, [&](const Coordinates &)
    {
        const VectorType data      = wrapper::vloadq(reinterpret_cast<const T *>(input.ptr()));
        const VectorType data_out1 = get_data_out<VectorType, S>(data, 0);
        const VectorType data_out2 = get_data_out<VectorType, S>(data, S / 2);
        auto              out       = reinterpret_cast<T *>(output.ptr());

        wrapper::vstore(out, data_out1);
        wrapper::vstore(out + S, data_out2);
        wrapper::vstore(out + offset_y_out, data_out1);
        wrapper::vstore(out + offset_y_out + S, data_out2);
    },
    input, output);
}

template <typename T, int S>
void NEUpsampleLayerKernel::upsample_nhwc(const arm_compute::Window &window)
{
    using VectorType = typename wrapper::traits::neon_vector<T, S>::type;

    Window window_out(window);
    window_out.set(Window::DimY, Window::Dimension(0, _output->info()->dimension(1), _info.x()));
    window_out.set(Window::DimZ, Window::Dimension(0, _output->info()->dimension(2), _info.y()));

    Iterator input(_input, window);
    Iterator output(_output, window_out);

    const int offset_y_out = _output->info()->strides_in_bytes().y() / sizeof(T);
    const int offset_z_out = _output->info()->strides_in_bytes().z() / sizeof(T);
    execute_window_loop(window_out, [&](const Coordinates &)
    {
        const VectorType data = wrapper::vloadq(reinterpret_cast<const T *>(input.ptr()));
        auto              out  = reinterpret_cast<T *>(output.ptr());

        wrapper::vstore(out, data);
        wrapper::vstore(out + offset_y_out, data);
        wrapper::vstore(out + offset_z_out, data);
        wrapper::vstore(out + offset_y_out + offset_z_out, data);
    },
    input, output);
}

void NEUpsampleLayerKernel::configure(const ITensor *input, ITensor *output, const Size2D &info, const InterpolationPolicy policy)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_UNUSED(policy);

    _input  = input;
    _output = output;
    _info   = info;

    const DataLayout data_layout = input->info()->data_layout();

    TensorShape output_shape = misc::shape_calculator::compute_upsample_shape(*input->info(), info);
    auto_init_if_empty(*output->info(), output_shape, 1, input->info()->data_type());
    output->info()->set_data_layout(data_layout);

    // Perform validation step
    ARM_COMPUTE_ERROR_THROW_ON(NEUpsampleLayerKernel::validate(input->info(), output->info(), info, policy));

    _num_elems_processed_per_iteration_x = 16 / output->info()->element_size();

    switch(data_layout)
    {
        case DataLayout::NCHW:
        {
            switch(input->info()->data_type())
            {
                case DataType::QASYMM8_SIGNED:
                    _func = &NEUpsampleLayerKernel::upsample_nchw<int8_t, 16>;
                    break;
                case DataType::QASYMM8:
                    _func = &NEUpsampleLayerKernel::upsample_nchw<uint8_t, 16>;
                    break;
                case DataType::F32:
                    _func = &NEUpsampleLayerKernel::upsample_nchw<float, 4>;
                    break;
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                case DataType::F16:
                    _func = &NEUpsampleLayerKernel::upsample_nchw<float16_t, 8>;
                    ;
                    break;
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
                default:
                    ARM_COMPUTE_ERROR("Not implemented");
            }
            break;
        }
        case DataLayout::NHWC:
        {
            switch(input->info()->data_type())
            {
                case DataType::QASYMM8_SIGNED:
                    _func = &NEUpsampleLayerKernel::upsample_nhwc<int8_t, 16>;
                    break;
                case DataType::QASYMM8:
                    _func = &NEUpsampleLayerKernel::upsample_nhwc<uint8_t, 16>;
                    break;
                case DataType::F32:
                    _func = &NEUpsampleLayerKernel::upsample_nhwc<float, 4>;
                    break;
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                case DataType::F16:
                    _func = &NEUpsampleLayerKernel::upsample_nhwc<float16_t, 8>;
                    break;
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
                default:
                    ARM_COMPUTE_ERROR("Not implemented");
            }
            break;
        }
        default:
            ARM_COMPUTE_ERROR("Not implemented");
    }

    // Configure window
    std::pair<Status, Window> win_config = validate_and_configure_window(input->info(),
                                                                         output->info(),
                                                                         _num_elems_processed_per_iteration_x,
                                                                         info);
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    INEKernel::configure(win_config.second);
}

void NEUpsampleLayerKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(_func == nullptr);

    (this->*_func)(window);
}
} // namespace arm_compute
