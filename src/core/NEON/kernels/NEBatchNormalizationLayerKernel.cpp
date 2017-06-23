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
#include "arm_compute/core/NEON/kernels/NEBatchNormalizationLayerKernel.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/NEON/NEFixedPoint.h"
#include "arm_compute/core/NEON/NEMath.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

using namespace arm_compute;

NEBatchNormalizationLayerKernel::NEBatchNormalizationLayerKernel()
    : _func(nullptr), _input(nullptr), _output(nullptr), _mean(nullptr), _var(nullptr), _gamma(nullptr), _beta(nullptr), _epsilon()
{
}

void batch_normalization_q8(const ITensor *in, ITensor *out, const ITensor *mean, const ITensor *var, const ITensor *beta, const ITensor *gamma, float epsilon, const Window &window)
{
    Iterator input(in, window);
    Iterator output(out, window);

    // Hold information about the current feature map we are iterating.
    // Only compute denominator and NEON vectors once per feature map.
    int slice = -1;

    int        fixed_point_position = in->info()->fixed_point_position();
    const auto input_mean           = reinterpret_cast<const qint8_t *>(mean->ptr_to_element(Coordinates(0, 0)));
    const auto input_var            = reinterpret_cast<const qint8_t *>(var->ptr_to_element(Coordinates(0, 0)));
    const auto input_gamma          = reinterpret_cast<const qint8_t *>(gamma->ptr_to_element(Coordinates(0, 0)));
    const auto input_beta           = reinterpret_cast<const qint8_t *>(beta->ptr_to_element(Coordinates(0, 0)));

    qint8x16_t       mean_vec    = vdupq_n_qs8(0);
    qint8x16_t       var_vec     = vdupq_n_qs8(0);
    qint8x16_t       gamma_vec   = vdupq_n_qs8(0);
    qint8x16_t       beta_vec    = vdupq_n_qs8(0);
    qint8x16_t       denominator = vdupq_n_qs8(0);
    const qint8x16_t epsilon_vec = vdupq_n_qs8(scvt_qs8_f32(epsilon, fixed_point_position));
    execute_window_loop(window, [&](const Coordinates & id)
    {
        if(slice != id.z())
        {
            // Conctruct vectors
            mean_vec  = vdupq_n_qs8(*(input_mean + id.z()));
            var_vec   = vdupq_n_qs8(*(input_var + id.z()));
            gamma_vec = vdupq_n_qs8(*(input_gamma + id.z()));
            beta_vec  = vdupq_n_qs8(*(input_beta + id.z()));

            // Calculate denominator
            denominator = vqinvsqrtq_qs8(vqaddq_qs8(var_vec, epsilon_vec), fixed_point_position);
            slice       = id.z();
        }

        // Calculate x bar and store results
        const qint8x16_t numerator = vqsubq_qs8(vld1q_qs8(reinterpret_cast<const qint8_t *>(input.ptr())), mean_vec);
        const qint8x16_t x_bar     = vqmulq_qs8(numerator, denominator, fixed_point_position);
        vst1q_qs8(reinterpret_cast<qint8_t *>(output.ptr()), vqmlaq_qs8(beta_vec, x_bar, gamma_vec, fixed_point_position));
    },
    input, output);
}

void batch_normalization_fp32(const ITensor *in, ITensor *out, const ITensor *mean, const ITensor *var, const ITensor *beta, const ITensor *gamma, float epsilon, const Window &window)
{
    Iterator input(in, window);
    Iterator output(out, window);

    // Hold information about the current feature map we are iterating.
    // Only compute denominator and NEON vectors once per feature map.
    int slice = -1;

    const auto input_mean  = reinterpret_cast<const float *>(mean->ptr_to_element(Coordinates(0, 0)));
    const auto input_var   = reinterpret_cast<const float *>(var->ptr_to_element(Coordinates(0, 0)));
    const auto input_gamma = reinterpret_cast<const float *>(gamma->ptr_to_element(Coordinates(0, 0)));
    const auto input_beta  = reinterpret_cast<const float *>(beta->ptr_to_element(Coordinates(0, 0)));

    float32x4_t       mean_vec    = vdupq_n_f32(0.0);
    float32x4_t       var_vec     = vdupq_n_f32(0.0);
    float32x4_t       gamma_vec   = vdupq_n_f32(0.0);
    float32x4_t       beta_vec    = vdupq_n_f32(0.0);
    float32x4_t       denominator = vdupq_n_f32(0.0);
    const float32x4_t epsilon_vec = vdupq_n_f32(epsilon);
    execute_window_loop(window, [&](const Coordinates & id)
    {
        if(slice != id.z())
        {
            // Conctruct vectors
            mean_vec  = vdupq_n_f32(*(input_mean + id.z()));
            var_vec   = vdupq_n_f32(*(input_var + id.z()));
            gamma_vec = vdupq_n_f32(*(input_gamma + id.z()));
            beta_vec  = vdupq_n_f32(*(input_beta + id.z()));

            // Calculate denominator
            denominator = vinvsqrtq_f32(vaddq_f32(var_vec, epsilon_vec));
            slice       = id.z();
        }

        // Calculate x bar and store results
        const float32x4_t numerator = vsubq_f32(vld1q_f32(reinterpret_cast<const float *>(input.ptr())), mean_vec);
        const float32x4_t x_bar     = vmulq_f32(numerator, denominator);
        vst1q_f32(reinterpret_cast<float *>(output.ptr()), vmlaq_f32(beta_vec, x_bar, gamma_vec));
    },
    input, output);
}

void NEBatchNormalizationLayerKernel::configure(const ITensor *input, ITensor *output, const ITensor *mean, const ITensor *var, const ITensor *beta, const ITensor *gamma, float epsilon)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QS8, DataType::F32);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::QS8, DataType::F32);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(mean, 1, DataType::QS8, DataType::F32);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(var, 1, DataType::QS8, DataType::F32);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(gamma, 1, DataType::QS8, DataType::F32);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(beta, 1, DataType::QS8, DataType::F32);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_SHAPES(input, output);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_SHAPES(mean, var);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_SHAPES(mean, beta);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_SHAPES(mean, gamma);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_FIXED_POINT(input, output);

    _input   = input;
    _output  = output;
    _mean    = mean;
    _var     = var;
    _gamma   = gamma;
    _beta    = beta;
    _epsilon = epsilon;

    unsigned int num_elems_processed_per_iteration = 0;

    switch(input->info()->data_type())
    {
        case DataType::QS8:
            _func                             = &batch_normalization_q8;
            num_elems_processed_per_iteration = 16;
            break;
        case DataType::F32:
            _func                             = &batch_normalization_fp32;
            num_elems_processed_per_iteration = 4;
            break;
        default:
            ARM_COMPUTE_ERROR("Element size not supported");
            break;
    }

    Window win = calculate_max_window(*input->info(), Steps(num_elems_processed_per_iteration));

    AccessWindowHorizontal input_access(input->info(), 0, num_elems_processed_per_iteration);
    AccessWindowHorizontal output_access(output->info(), 0, num_elems_processed_per_iteration);

    update_window_and_padding(win, input_access, output_access);

    output_access.set_valid_region(win, input->info()->valid_region());

    INEKernel::configure(win);
}

void NEBatchNormalizationLayerKernel::run(const Window &window)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(_func == nullptr);

    (*_func)(_input, _output, _mean, _var, _beta, _gamma, _epsilon, window);
}
