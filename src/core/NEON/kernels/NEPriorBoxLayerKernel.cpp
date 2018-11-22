/*
 * Copyright (c) 2018 ARM Limited.
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
#include "arm_compute/core/NEON/kernels/NEPriorBoxLayerKernel.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"

#include <arm_neon.h>
#include <cstdint>

namespace arm_compute
{
namespace
{
Status validate_arguments(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, const PriorBoxLayerInfo &info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input1, input2, output);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input1, 1, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_LAYOUT(input1, input2);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input1, input2);

    // Check variances
    const int var_size = info.variances().size();
    if(var_size > 1)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(var_size != 4, "Must provide 4 variance values");
        for(int i = 0; i < var_size; ++i)
        {
            ARM_COMPUTE_RETURN_ERROR_ON_MSG(var_size <= 0, "Must be greater than 0");
        }
    }
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(info.steps()[0] < 0.f, "Step x should be greater or equal to 0");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(info.steps()[1] < 0.f, "Step y should be greater or equal to 0");

    if(!info.max_sizes().empty())
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(info.max_sizes().size() != info.min_sizes().size(), "Max and min sizes dimensions should match");
    }

    for(unsigned int i = 0; i < info.max_sizes().size(); ++i)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(info.max_sizes()[i] < info.min_sizes()[i], "Max size should be greater than min size");
    }

    if(output != nullptr && output->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON(output->dimension(get_data_layout_dimension_index(input1->data_layout(), DataLayoutDimension::HEIGHT)) != 2);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_LAYOUT(input1, output);
    }

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window(const ITensorInfo *input1, const ITensorInfo *input2, ITensorInfo *output, const PriorBoxLayerInfo &info)
{
    ARM_COMPUTE_UNUSED(input2);

    Window win            = {};
    bool   window_changed = false;
    switch(input1->data_layout())
    {
        case DataLayout::NCHW:
        {
            const int          num_priors                        = info.aspect_ratios().size() * info.min_sizes().size() + info.max_sizes().size();
            const unsigned int num_elems_processed_per_iteration = 4 * num_priors;
            win                                                  = calculate_max_window(*output, Steps(num_elems_processed_per_iteration));
            AccessWindowHorizontal output_access(output, 0, num_elems_processed_per_iteration);
            window_changed = update_window_and_padding(win, output_access);
            break;
        }
        case DataLayout::NHWC:
        {
            win = calculate_max_window(*output, Steps());
            break;
        }
        default:
            ARM_COMPUTE_ERROR("Not implemented");
    };

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win);
}
} // namespace

NEPriorBoxLayerKernel::NEPriorBoxLayerKernel()
    : _func(nullptr), _input1(nullptr), _input2(nullptr), _output(nullptr), _info()
{
}

template <DataLayout DL>
void NEPriorBoxLayerKernel::store_coordinates(float *out, const int offset, const float center_x, const float center_y, const float box_width, const float box_height, const int width,
                                              const int height)
{
    float xmin = (center_x - box_width / 2.f) / width;
    float ymin = (center_y - box_height / 2.f) / height;
    float xmax = (center_x + box_width / 2.f) / width;
    float ymax = (center_y + box_height / 2.f) / height;

    switch(DL)
    {
        case DataLayout::NCHW:
        {
            float32x4_t vec_elements = { xmin, ymin, xmax, ymax };
            if(_info.clip())
            {
                static const float32x4_t CONST_0 = vdupq_n_f32(0.f);
                static const float32x4_t CONST_1 = vdupq_n_f32(1.f);
                vec_elements                     = vmaxq_f32(vminq_f32(vec_elements, CONST_1), CONST_0);
            }
            vst1q_f32(out + offset, vec_elements);
        }
        break;
        case DataLayout::NHWC:
        {
            const int output_offset = _output->info()->strides_in_bytes()[1] / _output->info()->element_size();
            if(_info.clip())
            {
                xmin = std::min(std::max(xmin, 0.f), 1.f);
                ymin = std::min(std::max(ymin, 0.f), 1.f);
                xmax = std::min(std::max(xmax, 0.f), 1.f);
                ymax = std::min(std::max(ymax, 0.f), 1.f);
            }

            *(out + output_offset * offset)       = xmin;
            *(out + output_offset * (offset + 1)) = ymin;
            *(out + output_offset * (offset + 2)) = xmax;
            *(out + output_offset * (offset + 3)) = ymax;
        }
        break;
        default:
            ARM_COMPUTE_ERROR("Not implemented");
    }
}

template <DataLayout DL>
void NEPriorBoxLayerKernel::calculate_prior_boxes(const Window &window)
{
    const int num_priors = _info.aspect_ratios().size() * _info.min_sizes().size() + _info.max_sizes().size();

    const int width_idx  = get_data_layout_dimension_index(DL, DataLayoutDimension::WIDTH);
    const int height_idx = get_data_layout_dimension_index(DL, DataLayoutDimension::HEIGHT);

    const int layer_width  = _input1->info()->dimension(width_idx);
    const int layer_height = _input1->info()->dimension(height_idx);

    int img_width  = _info.img_size().x;
    int img_height = _info.img_size().y;
    if(img_width == 0 || img_height == 0)
    {
        img_width  = _input2->info()->dimension(width_idx);
        img_height = _input2->info()->dimension(height_idx);
    }

    float step_x = _info.steps()[0];
    float step_y = _info.steps()[1];
    if(step_x == 0.f || step_y == 0.f)
    {
        step_x = static_cast<float>(img_width) / layer_width;
        step_y = static_cast<float>(img_height) / layer_height;
    }

    Window slice = {};

    switch(DL)
    {
        case DataLayout::NCHW:
            slice = window.first_slice_window_2D();
            slice.set(Window::DimY, Window::Dimension(0, _output->info()->dimension(1), 2));
            break;
        case DataLayout::NHWC:
            slice = window.first_slice_window_3D();
            slice.set(Window::DimY, Window::Dimension(0, _output->info()->dimension(1), 4 * num_priors));
            slice.set(Window::DimZ, Window::Dimension(0, _output->info()->dimension(2), 2));
            break;
        default:
            ARM_COMPUTE_ERROR("Not implemented");
    }

    Iterator output(_output, slice);
    execute_window_loop(slice, [&](const Coordinates & id)
    {
        float center_x = 0;
        float center_y = 0;
        int   idx      = 0;
        switch(DL)
        {
            case DataLayout::NCHW:
                idx      = id.x() / (4 * num_priors);
                center_x = (static_cast<float>(idx % layer_width) + _info.offset()) * step_x;
                center_y = (static_cast<float>(idx / layer_width) + _info.offset()) * step_y;
                break;
            case DataLayout::NHWC:
                idx      = id.y() / (4 * num_priors);
                center_x = (static_cast<float>(idx % layer_width) + _info.offset()) * step_x;
                center_y = (static_cast<float>(idx / layer_width) + _info.offset()) * step_y;
                break;
            default:
                ARM_COMPUTE_ERROR("Not implemented");
        }

        float box_width;
        float box_height;
        int   offset = 0;

        auto out = reinterpret_cast<float *>(output.ptr());
        for(unsigned int i = 0; i < _info.min_sizes().size(); ++i)
        {
            const float min_size = _info.min_sizes().at(i);
            box_width            = min_size;
            box_height           = min_size;
            store_coordinates<DL>(out, offset, center_x, center_y, box_width, box_height, img_width, img_height);
            offset += 4;

            if(!_info.max_sizes().empty())
            {
                const float max_size = _info.max_sizes().at(i);
                box_width            = std::sqrt(min_size * max_size);
                box_height           = box_width;

                store_coordinates<DL>(out, offset, center_x, center_y, box_width, box_height, img_width, img_height);
                offset += 4;
            }

            // rest of priors
            for(auto ar : _info.aspect_ratios())
            {
                if(fabs(ar - 1.) < 1e-6)
                {
                    continue;
                }

                box_width  = min_size * sqrt(ar);
                box_height = min_size / sqrt(ar);

                store_coordinates<DL>(out, offset, center_x, center_y, box_width, box_height, img_width, img_height);
                offset += 4;
            }
        }

        // set the variance
        switch(DL)
        {
            case DataLayout::NCHW:
            {
                out = reinterpret_cast<float *>(_output->ptr_to_element(Coordinates(id.x(), 1)));
                float32x4_t var;
                if(_info.variances().size() == 1)
                {
                    var = vdupq_n_f32(_info.variances().at(0));
                }
                else
                {
                    const float32x4_t vars = { _info.variances().at(0), _info.variances().at(1), _info.variances().at(2), _info.variances().at(3) };
                    var                    = vars;
                }
                for(int i = 0; i < num_priors; ++i)
                {
                    vst1q_f32(out + 4 * i, var);
                }
            }
            break;
            case DataLayout::NHWC:
            {
                for(int i = 0; i < num_priors; ++i)
                {
                    const int  prior_offset = 4 * i;
                    const bool single_var   = _info.variances().size() == 1;
                    *(reinterpret_cast<float *>(_output->ptr_to_element(Coordinates(0, id.y() + prior_offset + 0, 1)))) = _info.variances().at(0);
                    *(reinterpret_cast<float *>(_output->ptr_to_element(Coordinates(0, id.y() + prior_offset + 1, 1)))) = single_var ? _info.variances().at(0) : _info.variances().at(1);
                    *(reinterpret_cast<float *>(_output->ptr_to_element(Coordinates(0, id.y() + prior_offset + 2, 1)))) = single_var ? _info.variances().at(0) : _info.variances().at(2);
                    *(reinterpret_cast<float *>(_output->ptr_to_element(Coordinates(0, id.y() + prior_offset + 3, 1)))) = single_var ? _info.variances().at(0) : _info.variances().at(3);
                }
            }
            break;
            default:
                ARM_COMPUTE_ERROR("Not implemented");
        }

    },
    output);
}

void NEPriorBoxLayerKernel::configure(const ITensor *input1, const ITensor *input2, ITensor *output, const PriorBoxLayerInfo &info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input1, input2, output);

    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input1->info(), input2->info(), output->info(), info));

    _input1 = input1;
    _input2 = input2;
    _info   = info;
    _output = output;

    switch(input1->info()->data_layout())
    {
        case DataLayout::NCHW:
        {
            _func = &NEPriorBoxLayerKernel::calculate_prior_boxes<DataLayout::NCHW>;
            break;
        }
        case DataLayout::NHWC:
        {
            _func = &NEPriorBoxLayerKernel::calculate_prior_boxes<DataLayout::NHWC>;
            break;
        }
        default:
            ARM_COMPUTE_ERROR("Not implemented.");
    }

    // Configure kernel window
    auto win_config = validate_and_configure_window(input1->info(), input2->info(), output->info(), info);
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    INEKernel::configure(win_config.second);
}

Status NEPriorBoxLayerKernel::validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, const PriorBoxLayerInfo &info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input1, input2, output);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input1, input2, output, info));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input1->clone().get(), input2->clone().get(), output->clone().get(), info)
                                .first);

    return Status{};
}
void NEPriorBoxLayerKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(_func == nullptr);

    // Run function
    (this->*_func)(window);
}
} // namespace arm_compute