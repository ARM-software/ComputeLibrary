/*
 * Copyright (c) 2019-2020 Arm Limited.
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
#ifndef ARM_COMPUTE_MISC_INFO_HELPERS_H
#define ARM_COMPUTE_MISC_INFO_HELPERS_H

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/common/LSTMParams.h"

namespace arm_compute
{
namespace utils
{
namespace info_helpers
{
/** Checks if activation information correspond to a relu activation function
 *
 * @param[in] activation_info Activation metadata
 *
 * @return True if activation metadata correspond to a relu activation else false
 */
inline bool is_relu(ActivationLayerInfo activation_info)
{
    return activation_info.enabled() && activation_info.activation() == ActivationLayerInfo::ActivationFunction::RELU;
}

/** Checks if activation information correspond to a relu6 activation function
 *
 * @param[in] activation_info Activation metadata
 *
 * @return True if activation metadata correspond to a relu6 activation else false
 */
inline bool is_relu6(ActivationLayerInfo activation_info)
{
    const bool is_lu_bounded_relu = activation_info.activation() == ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU
                                    && activation_info.a() == 6.f && activation_info.b() == 0.f;
    const bool is_bounded_relu = activation_info.activation() == ActivationLayerInfo::ActivationFunction::BOUNDED_RELU
                                 && activation_info.a() == 6.f;
    return activation_info.enabled() && (is_lu_bounded_relu || is_bounded_relu);
}

/** Build LSTMParams<ITensorInfo> object by extracting the metadata from each
 * tensor.
 *
 * @param[in]  lstm_params      The LSTMParams<T> object containing the tensors.
 * @param[out] lstm_params_info The LSTMParams<ITensorInfo> to be constructed.
 *
 */
template <typename T>
inline void build_lstm_params_tensor_info(const LSTMParams<T>     &lstm_params,
                                          LSTMParams<ITensorInfo> *lstm_params_info)
{
    if(lstm_params.has_peephole_opt())
    {
        ARM_COMPUTE_ERROR_ON_NULLPTR(lstm_params.cell_to_forget_weights(), lstm_params.cell_to_output_weights());
        lstm_params_info->set_peephole_params(lstm_params.cell_to_forget_weights()->info(), lstm_params.cell_to_output_weights()->info());
    }
    if(lstm_params.has_projection())
    {
        ARM_COMPUTE_ERROR_ON_NULLPTR(lstm_params.projection_weights());
        lstm_params_info->set_projection_params(lstm_params.projection_weights()->info(),
                                                lstm_params.projection_bias() != nullptr ? lstm_params.projection_bias()->info() : nullptr);
    }
    if(!lstm_params.has_cifg_opt())
    {
        ARM_COMPUTE_ERROR_ON_NULLPTR(lstm_params.input_to_input_weights(), lstm_params.recurrent_to_input_weights(), lstm_params.input_gate_bias());

        ITensorInfo *cell_to_input_weights_info = (lstm_params.has_peephole_opt()) ? lstm_params.cell_to_input_weights()->info() : nullptr;
        lstm_params_info->set_cifg_params(lstm_params.input_to_input_weights()->info(), lstm_params.recurrent_to_input_weights()->info(),
                                          cell_to_input_weights_info, lstm_params.input_gate_bias()->info());
    }
    if(lstm_params.use_layer_norm())
    {
        ARM_COMPUTE_ERROR_ON_NULLPTR(lstm_params.forget_layer_norm_weights(),
                                     lstm_params.output_layer_norm_weights(),
                                     lstm_params.cell_layer_norm_weights());
        if(!lstm_params.has_cifg_opt())
        {
            ARM_COMPUTE_ERROR_ON_NULLPTR(lstm_params.input_layer_norm_weights());
        }

        ITensorInfo *forget_info = lstm_params.forget_layer_norm_weights()->info();
        ITensorInfo *cell_info   = lstm_params.cell_layer_norm_weights()->info();
        ITensorInfo *output_info = lstm_params.output_layer_norm_weights()->info();
        ITensorInfo *input_info  = lstm_params.has_cifg_opt() ? nullptr : lstm_params.input_layer_norm_weights()->info();

        lstm_params_info->set_layer_normalization_params(input_info, forget_info, cell_info, output_info);
    }

    lstm_params_info->set_matmul_scale_params(lstm_params.input_intermediate_scale(),
                                              lstm_params.forget_intermediate_scale(),
                                              lstm_params.cell_intermediate_scale(),
                                              lstm_params.output_intermediate_scale());

    lstm_params_info->set_hidden_state_params(lstm_params.hidden_state_zero(), lstm_params.hidden_state_scale());
}
} // namespace info_helpers
} // namespace utils
} // namespace arm_compute
#endif /* ARM_COMPUTE_MISC_INFO_HELPERS_H */
