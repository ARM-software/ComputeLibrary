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
#ifndef ARM_COMPUTE_LSTMPARAMS_H
#define ARM_COMPUTE_LSTMPARAMS_H

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/Tensor.h"

#include <cstddef>
#include <memory>

namespace arm_compute
{
template <typename T>
class LSTMParams
{
public:
    /** Constructor */
    LSTMParams()
        : _input_to_input_weights(nullptr),
          _recurrent_to_input_weights(nullptr),
          _cell_to_input_weights(nullptr),
          _input_gate_bias(nullptr),
          _cell_to_forget_weights(nullptr),
          _cell_to_output_weights(nullptr),
          _projection_weights(nullptr),
          _projection_bias(nullptr),
          _input_layer_norm_weights(nullptr),
          _forget_layer_norm_weights(nullptr),
          _cell_layer_norm_weights(nullptr),
          _output_layer_norm_weights(nullptr),
          _cell_clip(0.f),
          _projection_clip(0.0f),
          _input_intermediate_scale(0.0f),
          _forget_intermediate_scale(0.0f),
          _cell_intermediate_scale(0.0f),
          _output_intermediate_scale(0.0f),
          _hidden_state_zero(0),
          _hidden_state_scale(0.0f),
          _has_peephole_opt(false),
          _has_projection(false),
          _has_cifg_opt(true),
          _use_layer_norm(false)
    {
    }
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    LSTMParams(const LSTMParams &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    LSTMParams &operator=(const LSTMParams &) = delete;
    /** Default destructor */
    ~LSTMParams() = default;
    /** Set CIFG tensor parameters.
     *
     * @param[in] input_to_input_weights     2D weights tensor with dimensions [input_size, num_units]. Data types supported: QSYMM8/F16/F32.
     * @param[in] recurrent_to_input_weights 2D weights tensor with dimensions [output_size, num_units]. Data type supported: Same as @p input_to_input_weights.
     * @param[in] cell_to_input_weights      1D weights tensor with dimensions [num_units]. Can be nullptr. Data type supported: Same as @p input_to_input_weights.
     * @param[in] input_gate_bias            1D weights tensor with dimensions [num_units]. Data type supported: Same as @p input_to_input_weights, S32 when @p input_to_input_weights is QSYMM8
     *
     * @return Reference to this LSTMParams object
     */
    LSTMParams &set_cifg_params(const T *input_to_input_weights, const T *recurrent_to_input_weights, T *cell_to_input_weights, const T *input_gate_bias)
    {
        _input_to_input_weights     = input_to_input_weights;
        _recurrent_to_input_weights = recurrent_to_input_weights;
        _cell_to_input_weights      = cell_to_input_weights;
        _input_gate_bias            = input_gate_bias;
        _has_cifg_opt               = false;
        return *this;
    }
    /** Set projection tensor parameters.
     *
     * @param[in] projection_weights 2D weights tensor with dimensions [output_size, num_units]. Data type supported: Data types supported: QSYMM8/F16/F32.
     * @param[in] projection_bias    1D weights tensor with dimensions [output_size]. Data type supported: Same as @p projection_weights, S32 when @p input_to_input_weights is QSYMM8.
     *
     * @return Reference to this LSTMParams object
     */
    LSTMParams &set_projection_params(const T *projection_weights, const T *projection_bias)
    {
        _projection_weights = projection_weights;
        _projection_bias    = projection_bias;
        _has_projection     = true;
        return *this;
    }
    /** Set peephole tensor parameters.
     *
     * @param[in] cell_to_forget_weights 1D weights tensor with dimensions [num_units]. Data type supported: Data types supported: QSYMM16/F16/F32.
     * @param[in] cell_to_output_weights 1D weights tensor with dimensions [num_units]. Data type supported: Same as @p cell_to_forget_weights.
     *
     * @return Reference to this LSTMParams object
     */
    LSTMParams &set_peephole_params(T *cell_to_forget_weights, T *cell_to_output_weights)
    {
        _cell_to_forget_weights = cell_to_forget_weights;
        _cell_to_output_weights = cell_to_output_weights;
        _has_peephole_opt       = true;
        return *this;
    }
    /** Set layer normalization tensor parameters.
     *
     * @param[in] input_layer_norm_weights  1D weights tensor with dimensions [num_units]. Data type supported: Data types supported: QSYMM16/F16/F32.
     * @param[in] forget_layer_norm_weights 1D weights tensor with dimensions [num_units]. Data type supported: Same as @p input_layer_norm_weights.
     * @param[in] cell_layer_norm_weights   1D weights tensor with dimensions [num_units]. Data type supported: Same as @p input_layer_norm_weights.
     * @param[in] output_layer_norm_weights 1D weights tensor with dimensions [num_units]. Data type supported: Same as @p input_layer_norm_weights.
     *
     * @return Reference to this LSTMParams object
     */
    LSTMParams &set_layer_normalization_params(T *input_layer_norm_weights, T *forget_layer_norm_weights,
                                               T *cell_layer_norm_weights, T *output_layer_norm_weights)
    {
        _input_layer_norm_weights  = input_layer_norm_weights;
        _forget_layer_norm_weights = forget_layer_norm_weights;
        _cell_layer_norm_weights   = cell_layer_norm_weights;
        _output_layer_norm_weights = output_layer_norm_weights;
        _use_layer_norm            = true;
        return *this;
    }

    /** Set cell clip value.
     *
     * @param[in] cell_clip Value to be used to clip the cell state prior to the cell output activation.
     *
     * @return Reference to this LSTMParams object
     */
    LSTMParams &set_cell_clip_params(float cell_clip)
    {
        _cell_clip = cell_clip;
        return *this;
    }

    /** Set projection clip value.
     *
     * @param[in] projection_clip Value to be used to clip the projection, in case projection is enabled.
     *
     * @return Reference to this LSTMParams object
     */
    LSTMParams &set_projection_clip_params(float projection_clip)
    {
        _projection_clip = projection_clip;
        return *this;
    }

    /** Set scale of the intermediate results of matmul of each layer parameters.
     *
     * @param[in] input_intermediate_scale  Scale of the intermediate result of matmul, i.e. input to layer normalization, at input gate.
     * @param[in] forget_intermediate_scale Scale of the intermediate result of matmul, i.e. input to layer normalization, at forget gate.
     * @param[in] cell_intermediate_scale   Scale of the intermediate result of matmul, i.e. input to layer normalization, at cell gate.
     * @param[in] output_intermediate_scale Scale of the intermediate result of matmul, i.e. input to layer normalization, at output gate.
     *
     * @return Reference to this LSTMParams object
     */
    LSTMParams &set_matmul_scale_params(float input_intermediate_scale, float forget_intermediate_scale, float cell_intermediate_scale, float output_intermediate_scale)
    {
        _input_intermediate_scale  = input_intermediate_scale;
        _forget_intermediate_scale = forget_intermediate_scale;
        _cell_intermediate_scale   = cell_intermediate_scale;
        _output_intermediate_scale = output_intermediate_scale;
        return *this;
    }

    /** Set hidden state zero and scale parameters.
     *
     * @param[in] hidden_state_zero  The zero point of the hidden state.
     * @param[in] hidden_state_scale The scale of the hidden state.
     *
     * @return Reference to this LSTMParams object
     */
    LSTMParams &set_hidden_state_params(int32_t hidden_state_zero, float hidden_state_scale)
    {
        _hidden_state_zero  = hidden_state_zero;
        _hidden_state_scale = hidden_state_scale;
        return *this;
    }

    const T *input_to_input_weights() const
    {
        return _input_to_input_weights;
    }

    const T *recurrent_to_input_weights() const
    {
        return _recurrent_to_input_weights;
    }

    T *cell_to_input_weights() const
    {
        return _cell_to_input_weights;
    }

    const T *input_gate_bias() const
    {
        return _input_gate_bias;
    }

    T *cell_to_forget_weights() const
    {
        return _cell_to_forget_weights;
    }

    T *cell_to_output_weights() const
    {
        return _cell_to_output_weights;
    }

    const T *projection_weights() const
    {
        return _projection_weights;
    }

    const T *projection_bias() const
    {
        return _projection_bias;
    }

    T *input_layer_norm_weights() const
    {
        return _input_layer_norm_weights;
    }

    T *forget_layer_norm_weights() const
    {
        return _forget_layer_norm_weights;
    }

    T *cell_layer_norm_weights() const
    {
        return _cell_layer_norm_weights;
    }

    T *output_layer_norm_weights() const
    {
        return _output_layer_norm_weights;
    }

    float cell_clip() const
    {
        return _cell_clip;
    }

    float projection_clip() const
    {
        return _projection_clip;
    }

    float input_intermediate_scale() const
    {
        return _input_intermediate_scale;
    }

    float forget_intermediate_scale() const
    {
        return _forget_intermediate_scale;
    }

    float cell_intermediate_scale() const
    {
        return _cell_intermediate_scale;
    }

    float output_intermediate_scale() const
    {
        return _output_intermediate_scale;
    }

    int32_t hidden_state_zero() const
    {
        return _hidden_state_zero;
    }

    float hidden_state_scale() const
    {
        return _hidden_state_scale;
    }

    bool has_peephole_opt() const
    {
        return _has_peephole_opt;
    }

    bool has_projection() const
    {
        return _has_projection;
    }

    bool has_cifg_opt() const
    {
        return _has_cifg_opt;
    }

    bool use_layer_norm() const
    {
        return _use_layer_norm;
    }

private:
    const T *_input_to_input_weights;
    const T *_recurrent_to_input_weights;
    T       *_cell_to_input_weights;
    const T *_input_gate_bias;
    T       *_cell_to_forget_weights;
    T       *_cell_to_output_weights;
    const T *_projection_weights;
    const T *_projection_bias;
    T       *_input_layer_norm_weights;
    T       *_forget_layer_norm_weights;
    T       *_cell_layer_norm_weights;
    T       *_output_layer_norm_weights;
    float    _cell_clip;
    float    _projection_clip;
    float    _input_intermediate_scale;
    float    _forget_intermediate_scale;
    float    _cell_intermediate_scale;
    float    _output_intermediate_scale;
    int32_t  _hidden_state_zero;
    float    _hidden_state_scale;
    bool     _has_peephole_opt;
    bool     _has_projection;
    bool     _has_cifg_opt;
    bool     _use_layer_norm;
};
}
#endif /*ARM_COMPUTE_LSTMPARAMS_H */
