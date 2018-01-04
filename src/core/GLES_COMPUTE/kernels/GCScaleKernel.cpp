/*
 * Copyright (c) 2016, 2018 ARM Limited.
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
#include "arm_compute/core/GLES_COMPUTE/kernels/GCScaleKernel.h"

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/GLES_COMPUTE/GCHelpers.h"
#include "arm_compute/core/GLES_COMPUTE/GCKernelLibrary.h"
#include "arm_compute/core/GLES_COMPUTE/IGCKernel.h"
#include "arm_compute/core/GLES_COMPUTE/IGCTensor.h"
#include "arm_compute/core/GLES_COMPUTE/OpenGLES.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Validate.h"

#include <set>
#include <string>

using namespace arm_compute;

BorderSize GCScaleKernel::border_size() const
{
    return BorderSize(1);
}

void GCScaleKernel::configure(const IGCTensor *input, IGCTensor *output, InterpolationPolicy policy, bool border_undefined, SamplingPolicy sampling_policy)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F16);
    ARM_COMPUTE_ERROR_ON_NULLPTR(output);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    ARM_COMPUTE_ERROR_ON(output == input);
    ARM_COMPUTE_ERROR_ON(policy != InterpolationPolicy::NEAREST_NEIGHBOR);
    ARM_COMPUTE_UNUSED(sampling_policy);

    _input  = input;
    _output = output;

    // Compute the ratio between source width/height and destination width/height
    const auto wr = static_cast<float>(input->info()->dimension(0)) / static_cast<float>(output->info()->dimension(0));
    const auto hr = static_cast<float>(input->info()->dimension(1)) / static_cast<float>(output->info()->dimension(1));

    // Compute actual border size
    BorderSize border = border_undefined ? BorderSize(0) : border_size();

    // Area interpolation behaves as Nearest Neighbour in case of up-sampling
    if(policy == InterpolationPolicy::AREA && wr <= 1.f && hr <= 1.f)
    {
        policy = InterpolationPolicy::NEAREST_NEIGHBOR;
    }
    else
    {
        ARM_COMPUTE_ERROR_ON(policy == InterpolationPolicy::AREA);
    }

    // Create kernel
    std::set<std::string> build_opts;
    build_opts.emplace("#define LOCAL_SIZE_X " + support::cpp11::to_string(1));
    build_opts.emplace("#define LOCAL_SIZE_Y " + support::cpp11::to_string(1));
    build_opts.emplace("#define LOCAL_SIZE_Z " + support::cpp11::to_string(1));

    build_opts.emplace("#define DATA_TYPE_FP16");
    build_opts.emplace("#define BORDER_SIZE " + support::cpp11::to_string(border.right));

    // Configure kernel window
    unsigned int num_elems_processed_per_iteration = 4;
    unsigned int input_width_alignment             = 2;

    // performance optimization for 2x upscaling with no border
    if((fabs(wr - 0.5) < 1e-6) && (fabs(hr - 0.5) < 1e-6) && border_undefined)
    {
        num_elems_processed_per_iteration = 8;
        input_width_alignment             = 4;
        build_opts.emplace("#define SCALE_NEAREST_8X");
    }
    else
    {
        build_opts.emplace("#define SCALE_NEAREST_GENERIC");
    }

    std::string interpolation_name = string_from_interpolation_policy(policy); // NOLINT
    std::transform(interpolation_name.begin(), interpolation_name.end(), interpolation_name.begin(), ::tolower);
    std::string kernel_name = "scale_" + interpolation_name;
    _kernel                 = GCKernelLibrary::get().create_kernel(kernel_name, build_opts);

    Window win = calculate_max_window(*output->info(), Steps(num_elems_processed_per_iteration));

    const ValidRegion &input_valid_region = input->info()->valid_region();

    const int total_width   = border.left + input_valid_region.anchor[0] + input_valid_region.shape[0] + border.right;
    const int padding_right = ceil_to_multiple(total_width, input_width_alignment) - border.left - input_valid_region.anchor[0] - input_valid_region.shape[0];

    // Reads can occur within the valid region of the input
    AccessWindowStatic input_access(input->info(),
                                    input_valid_region.anchor[0] - border.left, input_valid_region.anchor[1] - border.top,
                                    input_valid_region.anchor[0] + input_valid_region.shape[0] + padding_right,
                                    input_valid_region.anchor[1] + input_valid_region.shape[1] + border.bottom);

    AccessWindowHorizontal output_access(output->info(), 0, num_elems_processed_per_iteration);

    update_window_and_padding(win, input_access, output_access);

    output_access.set_valid_region(win, calculate_valid_region_scale(*(input->info()),
                                                                     output->info()->tensor_shape(),
                                                                     policy,
                                                                     border,
                                                                     border_undefined));

    IGCKernel::configure(win);

    unsigned int idx = 2 * num_arguments_per_2D_tensor(); //Skip the tensor parameters
    _kernel.set_argument<float>(idx++, static_cast<float>(input->info()->dimension(0)));
    _kernel.set_argument<float>(idx++, static_cast<float>(input->info()->dimension(1)));
    _kernel.set_argument<float>(idx++, wr);
    _kernel.set_argument<float>(idx++, hr);
}
