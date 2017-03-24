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
#include "arm_compute/core/NEON/kernels/NENormalizationLayerKernel.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/NEON/NEMath.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

using namespace arm_compute;

NENormalizationLayerKernel::NENormalizationLayerKernel()
    : _func(nullptr), _input(nullptr), _input_squared(nullptr), _output(nullptr), _norm_info(NormType::IN_MAP), _border_size()
{
}

BorderSize NENormalizationLayerKernel::border_size() const
{
    return _border_size;
}

void NENormalizationLayerKernel::configure(const ITensor *input, const ITensor *input_squared, ITensor *output, NormalizationLayerInfo norm_info)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F32);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::F32);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    ARM_COMPUTE_ERROR_ON_MSG(!(norm_info.norm_size() % 2), "Normalization size should be odd");

    const unsigned int border_width = (_norm_info.type() == NormType::IN_MAP) ? 3 : 0;

    _input         = input;
    _input_squared = input_squared;
    _output        = output;
    _norm_info     = norm_info;
    _func          = (norm_info.type() == NormType::IN_MAP) ? &NENormalizationLayerKernel::normalize<0> : &NENormalizationLayerKernel::normalize<2>;
    _border_size   = BorderSize(0, border_width);

    const unsigned int num_elems_processed_per_iteration = 4;
    const unsigned int num_elems_read_per_iteration      = num_elems_processed_per_iteration + 2 * (norm_info.norm_size() / 2);

    Window win = calculate_max_window(*input->info(), Steps(num_elems_processed_per_iteration));

    AccessWindowHorizontal input_access(input->info(), -_border_size.left, num_elems_read_per_iteration);
    AccessWindowHorizontal input_squared_access(input_squared->info(), -_border_size.left, num_elems_read_per_iteration);
    AccessWindowHorizontal output_access(output->info(), 0, num_elems_processed_per_iteration);

    update_window_and_padding(win, input_access, input_squared_access, output_access);

    output_access.set_valid_region(win, input->info()->valid_region());

    INEKernel::configure(win);
}

template <unsigned int dim>
void NENormalizationLayerKernel::normalize(const Window &window)
{
    Iterator input(_input, window);
    Iterator input_squared(_input_squared, window);
    Iterator output(_output, window);

    const int radius               = _norm_info.norm_size() / 2;
    const int total_size           = _input->info()->dimension(dim) - 1;
    const int input_squared_stride = _input_squared->info()->strides_in_bytes()[dim];
    // We account padding when we normalize across X
    const int min_left  = (dim == 0) ? -static_cast<int>(border_size().left) : 0;
    const int max_right = (dim == 0) ? total_size + border_size().left : total_size;

    const float32x4_t coeff_vec = vdupq_n_f32(_norm_info.scale_coeff());
    const float32x4_t beta_vec  = vdupq_n_f32(_norm_info.beta());
    const float32x4_t kappa_vec = vdupq_n_f32(_norm_info.kappa());

    execute_window_loop(window, [&](const Coordinates & id)
    {
        // Get range to normalize
        const int current_slice = id[dim];
        const int first_slice   = std::max(current_slice - radius, min_left);
        const int last_slice    = std::min(current_slice + radius, max_right);

        // Accumulate cross map values
        float32x4_t accu = vdupq_n_f32(0.f);
        for(int i = first_slice; i <= last_slice; ++i)
        {
            accu = vaddq_f32(accu, vld1q_f32(reinterpret_cast<float *>(input_squared.ptr() + (i - current_slice) * input_squared_stride)));
        }

        // Normalize
        const float32x4_t normalized       = vpowq_f32(vmlaq_f32(kappa_vec, coeff_vec, accu), beta_vec);
        const float32x4_t normalized_pixel = vmulq_f32(vld1q_f32(reinterpret_cast<float *>(input.ptr())), vinv_f32(normalized));
        vst1q_f32(reinterpret_cast<float *>(output.ptr()), normalized_pixel);
    },
    input, input_squared, output);
}

void NENormalizationLayerKernel::run(const Window &window)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(_func == nullptr);

    // Run function
    (this->*_func)(window);
}
