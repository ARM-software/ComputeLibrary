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
#include "arm_compute/core/CL/kernels/CLPriorBoxLayerKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/CLValidate.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Window.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"

#include "support/ToolchainSupport.h"

using namespace arm_compute::misc::shape_calculator;

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
        ARM_COMPUTE_RETURN_ERROR_ON(output->dimension(1) != 2);
    }

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window(const ITensorInfo *input1, const ITensorInfo *input2, ITensorInfo *output, const PriorBoxLayerInfo &info, int num_priors)
{
    ARM_COMPUTE_UNUSED(input2);
    // Output tensor auto initialization if not yet initialized
    TensorShape output_shape = compute_prior_box_shape(*input1, info);
    auto_init_if_empty(*output, output_shape, 1, input1->data_type());

    const unsigned int     num_elems_processed_per_iteration = 4 * num_priors;
    Window                 win                               = calculate_max_window(*output, Steps(num_elems_processed_per_iteration));
    AccessWindowHorizontal output_access(output, 0, num_elems_processed_per_iteration);
    bool                   window_changed = update_window_and_padding(win, output_access);
    Status                 err            = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win);
}
} // namespace

CLPriorBoxLayerKernel::CLPriorBoxLayerKernel()
    : _input1(nullptr), _input2(nullptr), _output(nullptr), _info(), _num_priors(), _min(), _max(), _aspect_ratios()
{
}

void CLPriorBoxLayerKernel::configure(const ICLTensor *input1, const ICLTensor *input2, ICLTensor *output, const PriorBoxLayerInfo &info, cl::Buffer *min, cl::Buffer *max, cl::Buffer *aspect_ratios)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input1, input2, output);

    _input1        = input1;
    _input2        = input2;
    _output        = output;
    _info          = info;
    _min           = min;
    _max           = max;
    _aspect_ratios = aspect_ratios;

    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input1->info(), input2->info(), output->info(), info));

    // Calculate number of aspect ratios
    _num_priors = info.aspect_ratios().size() * info.min_sizes().size() + info.max_sizes().size();

    const DataLayout data_layout = input1->info()->data_layout();

    const int width_idx  = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const int height_idx = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);

    const int layer_width  = input1->info()->dimension(width_idx);
    const int layer_height = input1->info()->dimension(height_idx);

    int img_width  = info.img_size().x;
    int img_height = info.img_size().y;
    if(img_width == 0 || img_height == 0)
    {
        img_width  = input2->info()->dimension(width_idx);
        img_height = input2->info()->dimension(height_idx);
    }

    float step_x = info.steps()[0];
    float step_y = info.steps()[0];
    if(step_x == 0.f || step_y == 0.f)
    {
        step_x = static_cast<float>(img_width) / layer_width;
        step_y = static_cast<float>(img_height) / layer_height;
    }

    // Set build options
    CLBuildOptions build_opts;
    build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(input1->info()->data_type()));
    build_opts.add_option("-DWIDTH=" + support::cpp11::to_string(img_width));
    build_opts.add_option("-DHEIGHT=" + support::cpp11::to_string(img_height));
    build_opts.add_option("-DLAYER_WIDTH=" + support::cpp11::to_string(layer_width));
    build_opts.add_option("-DLAYER_HEIGHT=" + support::cpp11::to_string(layer_height));
    build_opts.add_option("-DSTEP_X=" + support::cpp11::to_string(step_x));
    build_opts.add_option("-DSTEP_Y=" + support::cpp11::to_string(step_y));
    build_opts.add_option("-DNUM_PRIORS=" + support::cpp11::to_string(_num_priors));
    build_opts.add_option("-DOFFSET=" + support::cpp11::to_string(info.offset()));
    build_opts.add_option_if(info.clip(), "-DIN_PLACE");

    if(info.variances().size() > 1)
    {
        for(unsigned int i = 0; i < info.variances().size(); ++i)
        {
            build_opts.add_option("-DVARIANCE_" + support::cpp11::to_string(i) + "=" + support::cpp11::to_string(info.variances().at(i)));
        }
    }
    else
    {
        for(unsigned int i = 0; i < 4; ++i)
        {
            build_opts.add_option("-DVARIANCE_" + support::cpp11::to_string(i) + "=" + support::cpp11::to_string(info.variances().at(0)));
        }
    }

    unsigned int idx = num_arguments_per_2D_tensor();
    _kernel          = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel("prior_box_layer_nchw", build_opts.options()));

    _kernel.setArg(idx++, *_min);
    _kernel.setArg(idx++, *_max);
    _kernel.setArg(idx++, *_aspect_ratios);
    _kernel.setArg<unsigned int>(idx++, info.min_sizes().size());
    _kernel.setArg<unsigned int>(idx++, info.max_sizes().size());
    _kernel.setArg<unsigned int>(idx++, info.aspect_ratios().size());

    // Configure kernel window
    auto win_config = validate_and_configure_window(input1->info(), input2->info(), output->info(), info, _num_priors);

    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    ICLKernel::configure_internal(win_config.second);
}

Status CLPriorBoxLayerKernel::validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, const PriorBoxLayerInfo &info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input1, input2, output);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input1, input2, output, info));
    const int num_priors = info.aspect_ratios().size() * info.min_sizes().size() + info.max_sizes().size();
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input1->clone().get(), input2->clone().get(), output->clone().get(), info, num_priors)
                                .first);

    return Status{};
}

void CLPriorBoxLayerKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);

    queue.enqueueWriteBuffer(*_min, CL_TRUE, 0, _info.min_sizes().size() * sizeof(float), _info.min_sizes().data());
    queue.enqueueWriteBuffer(*_aspect_ratios, CL_TRUE, 0, _info.aspect_ratios().size() * sizeof(float), _info.aspect_ratios().data());
    if(!_info.max_sizes().empty())
    {
        queue.enqueueWriteBuffer(*_max, CL_TRUE, 0, _info.max_sizes().size() * sizeof(float), _info.max_sizes().data());
    }

    Window slice = window.first_slice_window_2D();
    slice.set(Window::DimY, Window::Dimension(0, _output->info()->dimension(1), 2));

    unsigned int idx = 0;
    add_2D_tensor_argument(idx, _output, slice);
    enqueue(queue, *this, slice);
}
} // namespace arm_compute
