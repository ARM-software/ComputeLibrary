/*
 * Copyright (c) 2019-2021, 2023 Arm Limited.
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
#include "src/core/CL/kernels/CLPadLayerKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/utils/helpers/AdjustVecSize.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"

#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"
#include "support/StringSupport.h"

namespace arm_compute
{
namespace
{
Status validate_arguments(const ITensorInfo *input,
                          const ITensorInfo *output,
                          const PaddingList &padding,
                          PixelValue         constant_value,
                          PaddingMode        mode)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_UNUSED(constant_value);
    ARM_COMPUTE_RETURN_ERROR_ON(input->data_type() == DataType::UNKNOWN);
    ARM_COMPUTE_RETURN_ERROR_ON((padding.size() < 1) || (padding.size() > input->num_dimensions()));
    if (mode == PaddingMode::REFLECT || mode == PaddingMode::SYMMETRIC)
    {
        ARM_COMPUTE_RETURN_ERROR_ON(padding.size() > 3);

        const auto is_reflect = static_cast<unsigned int>(mode == PaddingMode::REFLECT);
        for (size_t i = 0; i < padding.size(); ++i)
        {
            ARM_COMPUTE_RETURN_ERROR_ON(padding.at(i).first > (input->dimension(i) - is_reflect));
            ARM_COMPUTE_RETURN_ERROR_ON(padding.at(i).second > (input->dimension(i) - is_reflect));
        }
    }

    if (output->total_size() > 0)
    {
        TensorShape padded_shape = misc::shape_calculator::compute_padded_shape(input->tensor_shape(), padding);

        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(output, input);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(output->tensor_shape(), padded_shape);
    }

    return Status{};
}
} // namespace

CLPadLayerKernel::CLPadLayerKernel() : _input(nullptr), _output(nullptr), _4d_enabled(false)
{
    _type = CLKernelType::ELEMENTWISE;
}

void CLPadLayerKernel::configure(
    const ICLTensor *input, ICLTensor *output, const PaddingList &padding, PixelValue constant_value, PaddingMode mode)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, output, padding, constant_value, mode);
}

void CLPadLayerKernel::configure(const CLCompileContext &compile_context,
                                 const ICLTensor        *input,
                                 ICLTensor              *output,
                                 const PaddingList      &padding,
                                 PixelValue              constant_value,
                                 PaddingMode             mode)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    auto_init_if_empty(*output->info(),
                       input->info()->clone()->set_tensor_shape(
                           misc::shape_calculator::compute_padded_shape(input->info()->tensor_shape(), padding)));
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), output->info(), padding, constant_value, mode));

    auto padding_info = get_padding_info({input, output});

    _input      = input;
    _output     = output;
    _4d_enabled = (mode == PaddingMode::CONSTANT) && (padding.size() > 3);

    // Set build options
    const DataType    &data_type    = input->info()->data_type();
    const unsigned int input_width  = input->info()->dimension(0);
    const unsigned int input_height = input->info()->dimension(1);
    const unsigned int input_depth  = input->info()->dimension(2);
    const unsigned int pad_x_before = padding.at(0).first;
    const unsigned int pad_y_before = padding.size() > 1 ? padding.at(1).first : 0;
    const unsigned int pad_z_before = padding.size() > 2 ? padding.at(2).first : 0;
    const unsigned int vec_size     = adjust_vec_size(
            std::min(16U, 32U / static_cast<unsigned int>(element_size_from_data_type(input->info()->data_type()))),
            input_width);
    const unsigned int pad_right_start        = input_width + pad_x_before;
    const unsigned int pad_x_before_remainder = pad_x_before % vec_size;
    const unsigned int vec_size_leftover_write =
        vec_size - (ceil_to_multiple(output->info()->dimension(0), vec_size) - output->info()->dimension(0));

    CLBuildOptions build_opts;
    build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(data_type));
    build_opts.add_option("-DVEC_SIZE=" + support::cpp11::to_string(vec_size));
    build_opts.add_option("-DPAD_X_BEFORE=" + support::cpp11::to_string(pad_x_before));
    build_opts.add_option("-DSRC_WIDTH=" + support::cpp11::to_string(input_width));
    build_opts.add_option("-DPAD_X_BEFORE_REMAINDER=" + support::cpp11::to_string(pad_x_before_remainder));
    build_opts.add_option("-DVEC_SIZE_LEFTOVER_WRITE=" + support::cpp11::to_string(vec_size_leftover_write));
    if (padding.size() > 1)
    {
        build_opts.add_option("-DPAD_Y_BEFORE=" + support::cpp11::to_string(pad_y_before));
        build_opts.add_option("-DSRC_HEIGHT=" + support::cpp11::to_string(input_height));

        if (padding.size() > 2)
        {
            build_opts.add_option("-DPAD_Z_BEFORE=" + support::cpp11::to_string(pad_z_before));
            build_opts.add_option("-DSRC_DEPTH=" + support::cpp11::to_string(input_depth));
        }
    }

    std::string kernel_name = "pad_layer_";
    switch (mode)
    {
        case PaddingMode::CONSTANT:
        {
            kernel_name += "constant";

            const unsigned int vec_size_leftover_read =
                vec_size - (ceil_to_multiple(pad_right_start, vec_size) - pad_right_start);

            build_opts.add_option("-DCONST_VAL=" + string_from_pixel_value(constant_value, data_type));
            build_opts.add_option("-DVEC_SIZE_LEFTOVER_READ=" + support::cpp11::to_string(vec_size_leftover_read));

            if (pad_x_before >= vec_size)
            {
                build_opts.add_option("-DTHREADS_TO_SKIP_BEFORE=" + support::cpp11::to_string(pad_x_before / vec_size));
                build_opts.add_option("-DTHREADS_TO_SKIP_AFTER=" +
                                      support::cpp11::to_string(pad_right_start / vec_size));
            }
            if (_4d_enabled)
            {
                build_opts.add_option("-DPAD_W_BEFORE=" + support::cpp11::to_string(padding.at(3).first));
                build_opts.add_option("-DSRC_BATCH=" + support::cpp11::to_string(input->info()->dimension(3)));
            }

            break;
        }
        case PaddingMode::SYMMETRIC:
        case PaddingMode::REFLECT:
        {
            kernel_name += "symmetric_reflect";

            const auto is_reflect = static_cast<unsigned int>(mode == PaddingMode::REFLECT);

            const unsigned int pad_x_after_remainder = pad_right_start % vec_size;
            const unsigned int after_pad_fact_x      = (2 * input_width + pad_x_before) - is_reflect;
            const unsigned int output_last_x = ceil_to_multiple(pad_right_start + padding.at(0).second, vec_size);

            build_opts.add_option("-DIS_REFLECT=" + support::cpp11::to_string(is_reflect));
            build_opts.add_option("-DPAD_X_AFTER_REMAINDER=" + support::cpp11::to_string(pad_x_after_remainder));
            build_opts.add_option("-DPAD_X_BEFORE_REMAINDER_REFL=" +
                                  support::cpp11::to_string((pad_x_before_remainder + is_reflect) % vec_size));
            build_opts.add_option("-DPAD_X_AFTER_REMAINDER_REFL=" +
                                  support::cpp11::to_string((pad_x_after_remainder - is_reflect) % vec_size));
            build_opts.add_option("-DAFTER_PAD_FACT_X=" + support::cpp11::to_string(after_pad_fact_x));
            build_opts.add_option_if(after_pad_fact_x < output_last_x,
                                     "-DAFTER_PAD_REM=" + support::cpp11::to_string(after_pad_fact_x % vec_size));

            break;
        }
        default:
            ARM_COMPUTE_ERROR("Padding mode not supported.");
    }

    // Create kernel
    _kernel = create_kernel(compile_context, kernel_name, build_opts.options());

    // Configure window
    Window win = calculate_max_window(*output->info(), Steps(vec_size));
    ICLKernel::configure_internal(win);

    ARM_COMPUTE_ERROR_ON(has_padding_changed(padding_info));
}

Status CLPadLayerKernel::validate(const ITensorInfo *input,
                                  const ITensorInfo *output,
                                  const PaddingList &padding,
                                  PixelValue         constant_value,
                                  PaddingMode        mode)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, padding, constant_value, mode));
    return Status{};
}

void CLPadLayerKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    Window       slice = window.first_slice_window_3D();
    unsigned int batch = 0;
    do
    {
        unsigned int idx = 0;
        add_3D_tensor_argument(idx, _input, slice);
        add_3D_tensor_argument(idx, _output, slice);
        if (_4d_enabled)
        {
            add_argument<unsigned int>(idx, batch++);
        }

        enqueue(queue, *this, slice, lws_hint());
    } while (window.slide_window_slice_3D(slice));
}
} // namespace arm_compute
