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
#ifndef __ARM_COMPUTE_CLLSTMLAYER_H__
#define __ARM_COMPUTE_CLLSTMLAYER_H__

#include "arm_compute/runtime/IFunction.h"

#include "arm_compute/core/CL/kernels/CLActivationLayerKernel.h"
#include "arm_compute/core/CL/kernels/CLArithmeticAdditionKernel.h"
#include "arm_compute/core/CL/kernels/CLArithmeticSubtractionKernel.h"
#include "arm_compute/core/CL/kernels/CLCopyKernel.h"
#include "arm_compute/core/CL/kernels/CLPixelWiseMultiplicationKernel.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/CLMemoryGroup.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/functions/CLArithmeticAddition.h"
#include "arm_compute/runtime/CL/functions/CLFullyConnectedLayer.h"
#include "arm_compute/runtime/CL/functions/CLGEMM.h"
#include "arm_compute/runtime/CL/functions/CLWidthConcatenateLayer.h"
#include "arm_compute/runtime/IMemoryManager.h"

#include <memory>

namespace arm_compute
{
class ICLTensor;

template <typename T>
class LSTMParams
{
public:
    /** Constructor */
    LSTMParams()
        : _input_to_input_weights(nullptr), _recurrent_to_input_weights(nullptr), _cell_to_input_weights(nullptr), _input_gate_bias(nullptr), _cell_to_forget_weights(nullptr),
          _cell_to_output_weights(nullptr), _projection_weights(nullptr), _projection_bias(nullptr), _has_peephole_opt(false), _has_projection(false), _has_cifg_opt(true)
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
     * @param[in] input_to_input_weights     2D weights tensor with dimensions [input_size, num_units]. Data types supported: F16/F32.
     * @param[in] recurrent_to_input_weights 2D weights tensor with dimensions [output_size, num_units]. Data type supported: Same as @p input_to_input_weights.
     * @param[in] cell_to_input_weights      1D weights tensor with dimensions [num_units]. Can be nullptr. Data type supported: Same as @p input_to_input_weights.
     * @param[in] input_gate_bias            1D weights tensor with dimensions [num_units]. Data type supported: Same as @p input_to_input_weights
     *
     * @return Reference to this LSTMParams object
     */
    LSTMParams &set_cifg_params(const T *input_to_input_weights, const T *recurrent_to_input_weights, const T *cell_to_input_weights, const T *input_gate_bias)
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
     * @param[in] projection_weights 2D weights tensor with dimensions [output_size, num_units]. Data type supported: Data types supported: F16/F32.
     * @param[in] projection_bias    1D weights tensor with dimensions [output_size]. Data type supported: Same as @p projection_weights.
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
     * @param[in] cell_to_forget_weights 1D weights tensor with dimensions [num_units]. Data type supported: Data types supported: F16/F32.
     * @param[in] cell_to_output_weights 1D weights tensor with dimensions [num_units]. Data type supported: Same as @p cell_to_input_weights.
     *
     * @return Reference to this LSTMParams object
     */
    LSTMParams &set_peephole_params(const T *cell_to_forget_weights, const T *cell_to_output_weights)
    {
        _cell_to_forget_weights = cell_to_forget_weights;
        _cell_to_output_weights = cell_to_output_weights;
        _has_peephole_opt       = true;
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

    const T *cell_to_input_weights() const
    {
        return _cell_to_input_weights;
    }

    const T *input_gate_bias() const
    {
        return _input_gate_bias;
    }

    const T *cell_to_forget_weights() const
    {
        return _cell_to_forget_weights;
    }

    const T *cell_to_output_weights() const
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

private:
    const T *_input_to_input_weights;
    const T *_recurrent_to_input_weights;
    const T *_cell_to_input_weights;
    const T *_input_gate_bias;
    const T *_cell_to_forget_weights;
    const T *_cell_to_output_weights;
    const T *_projection_weights;
    const T *_projection_bias;
    bool     _has_peephole_opt;
    bool     _has_projection;
    bool     _has_cifg_opt;
};

/** This function performs a single time step in a Long Short-Term Memory (LSTM) layer.
 *
 */
class CLLSTMLayer : public IFunction
{
public:
    /** Default constructor */
    CLLSTMLayer(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
    /** Initialize function's tensors.
     *
     * @param[in]  input                       Source tensor. Input is a 2D tensor with dimensions [input_size, batch_size]. Data types supported: F16/F32.
     * @param[in]  input_to_forget_weights     2D weights tensor with dimensions [input_size, num_units]. Data type supported: Same as @p input.
     * @param[in]  input_to_cell_weights       2D weights tensor with dimensions [input_size, num_units]. Data type supported: Same as @p input.
     * @param[in]  input_to_output_weights     2D weights tensor with dimensions [input_size, num_units]. Data type supported: Same as @p input.
     * @param[in]  recurrent_to_forget_weights 2D weights tensor with dimensions [output_size, num_units]. Data type supported: Same as @p input.
     * @param[in]  recurrent_to_cell_weights   2D weights tensor with dimensions [output_size, num_units]. Data type supported: Same as @p input.
     * @param[in]  recurrent_to_output_weights 2D weights tensor with dimensions [output_size, num_units]. Data type supported: Same as @p input.
     * @param[in]  forget_gate_bias            1D weights tensor with dimensions [num_units]. Data type supported: Same as @p input.
     * @param[in]  cell_bias                   1D weights tensor with dimensions [num_units]. Data type supported: Same as @p input.
     * @param[in]  output_gate_bias            1D weights tensor with dimensions [num_units]. Data type supported: Same as @p input.
     * @param[in]  output_state_in             2D weights tensor with dimensions [output_size, batch_size]. Data type supported: Same as @p input.
     * @param[in]  cell_state_in               2D tensor with dimensions [num_units, batch_size]. Data type supported: Same as @p input.
     * @param[out] scratch_buffer              2D tensor with dimensions [num_units * 4, batch_size] with CIFG or [num_units * 3, batch_size] without CIGF. Data type supported: Same as @p input.
     * @param[out] output_state_out            2D weights tensor with dimensions [output_size, batch_size]. Data type supported: Same as @p input.
     * @param[out] cell_state_out              2D tensor with dimensions [num_units, batch_size]. Data type supported: Same as @p input.
     * @param[out] output                      Destination tensor. Output is a 2D tensor with dimensions [output_size, batch_size].
     *                                         Data types supported: Same as @p input.
     * @param[in]  lstm_params                 (Optional) Weights tensors used in peephole optimization:
     *                                         input_to_input_weights       2D weights tensor with dimensions [input_size, num_units]. Data type supported: Same as @p input.
     *                                         recurrent_to_input_weights   2D weights tensor with dimensions [output_size, num_units]. Data type supported: Same as @p input.
     *                                         cell_to_input_weights        1D weights tensor with dimensions [num_units]. Can be nullptr. Data type supported: Same as @p input.
     *                                         cell_to_forget_weights       1D weights tensor with dimensions [num_units]. Data type supported: Same as @p input.
     *                                         cell_to_output_weights       1D weights tensor with dimensions [num_units]. Data type supported: Same as @p input.
     *                                         input_gate_bias              1D weights tensor with dimensions [num_units]. Data type supported: Same as @p input
     *                                         projection_weights           2D weights tensor with dimensions [output_size, num_units]. Data type supported: Same as @p input.
     *                                         projection_bias              1D weights tensor with dimensions [output_size]. Data type supported: Same as @p input.
     * @param[in]  activation_info             Contains activation information described in @ref ActivationLayerInfo.
     * @param[in]  cell_threshold              The clipping threshold for the cell state, such that values are bound within [-cell_clip, cell_clip]. If set to 0.0 then clipping is disabled.
     * @param[in]  projection_threshold        The clipping threshold for the output from the projection layer, such that values are bound within [-proj_clip, proj_clip]. If set to 0.0 then clipping is disabled.
     */
    void configure(const ICLTensor *input,
                   const ICLTensor *input_to_forget_weights, const ICLTensor *input_to_cell_weights, const ICLTensor *input_to_output_weights,
                   const ICLTensor *recurrent_to_forget_weights, const ICLTensor *recurrent_to_cell_weights, const ICLTensor *recurrent_to_output_weights,
                   const ICLTensor *forget_gate_bias, const ICLTensor *cell_bias, const ICLTensor *output_gate_bias,
                   const ICLTensor *output_state_in, const ICLTensor *cell_state_in,
                   ICLTensor *scratch_buffer, ICLTensor *output_state_out, ICLTensor *cell_state_out, ICLTensor *output,
                   const LSTMParams<ICLTensor> &lstm_params, const ActivationLayerInfo &activation_info, float cell_threshold = 0.f, float projection_threshold = 0.f);

    /** Static function to check if given info will lead to a valid configuration of @ref CLLSTMLayer
     *
     * @param[in] input                       Source tensor. Input is a 2D tensor with dimensions [input_size, batch_size]. Data types supported: F16/F32.
     * @param[in] input_to_forget_weights     2D weights tensor with dimensions [input_size, num_units]. Data type supported: Same as @p input.
     * @param[in] input_to_cell_weights       2D weights tensor with dimensions [input_size, num_units]. Data type supported: Same as @p input.
     * @param[in] input_to_output_weights     2D weights tensor with dimensions [input_size, num_units]. Data type supported: Same as @p input.
     * @param[in] recurrent_to_forget_weights 2D weights tensor with dimensions [output_size, num_units]. Data type supported: Same as @p input.
     * @param[in] recurrent_to_cell_weights   2D weights tensor with dimensions [output_size, num_units]. Data type supported: Same as @p input.
     * @param[in] recurrent_to_output_weights 2D weights tensor with dimensions [output_size, num_units]. Data type supported: Same as @p input.
     * @param[in] forget_gate_bias            1D weights tensor with dimensions [num_units]. Data type supported: Same as @p input.
     * @param[in] cell_bias                   1D weights tensor with dimensions [num_units]. Data type supported: Same as @p input.
     * @param[in] output_gate_bias            1D weights tensor with dimensions [num_units]. Data type supported: Same as @p input.
     * @param[in] output_state_in             2D weights tensor with dimensions [output_size, batch_size]. Data type supported: Same as @p input.
     * @param[in] cell_state_in               2D tensor with dimensions [num_units, batch_size]. Data type supported: Same as @p input.
     * @param[in] scratch_buffer              2D tensor with dimensions [num_units * 4, batch_size] with CIFG or [num_units * 3, batch_size] without CIGF. Data type supported: Same as @p input.
     * @param[in] output_state_out            2D weights tensor with dimensions [output_size, batch_size]. Data type supported: Same as @p input.
     * @param[in] cell_state_out              2D tensor with dimensions [num_units, batch_size]. Data type supported: Same as @p input.
     * @param[in] output                      Destination tensor. Output is a 2D tensor with dimensions [output_size, batch_size].
     *                                        Data types supported: Same as @p input.
     * @param[in] lstm_params                 (Optional) Weights tensors used in peephole optimization:
     *                                        input_to_input_weights       2D weights tensor with dimensions [input_size, num_units]. Data type supported: Same as @p input.
     *                                        recurrent_to_input_weights   2D weights tensor with dimensions [output_size, num_units]. Data type supported: Same as @p input.
     *                                        cell_to_input_weights        1D weights tensor with dimensions [num_units]. Can be nullptr. Data type supported: Same as @p input.
     *                                        cell_to_forget_weights       1D weights tensor with dimensions [num_units]. Data type supported: Same as @p input.
     *                                        cell_to_output_weights       1D weights tensor with dimensions [num_units]. Data type supported: Same as @p input.
     *                                        input_gate_bias              1D weights tensor with dimensions [num_units]. Data type supported: Same as @p input
     *                                        projection_weights           2D weights tensor with dimensions [output_size, num_units]. Data type supported: Same as @p input.
     *                                        projection_bias              1D weights tensor with dimensions [output_size]. Data type supported: Same as @p input.
     * @param[in] activation_info             Contains activation information described in @ref ActivationLayerInfo.
     * @param[in] cell_threshold              The clipping threshold for the cell state, such that values are bound within [-cell_clip, cell_clip]. If set to 0.0 then clipping is disabled.
     * @param[in] projection_threshold        The clipping threshold for the output from the projection layer, such that values are bound within [-proj_clip, proj_clip]. If set to 0.0 then clipping is disabled.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input,
                           const ITensorInfo *input_to_forget_weights, const ITensorInfo *input_to_cell_weights, const ITensorInfo *input_to_output_weights,
                           const ITensorInfo *recurrent_to_forget_weights, const ITensorInfo *recurrent_to_cell_weights, const ITensorInfo *recurrent_to_output_weights,
                           const ITensorInfo *forget_gate_bias, const ITensorInfo *cell_bias, const ITensorInfo *output_gate_bias,
                           const ITensorInfo *output_state_in, const ITensorInfo *cell_state_in,
                           const ITensorInfo *scratch_buffer, const ITensorInfo *output_state_out, const ITensorInfo *cell_state_out, const ITensorInfo *output,
                           const LSTMParams<ITensorInfo> &lstm_params, const ActivationLayerInfo &activation_info, float cell_threshold = 0.f, float projection_threshold = 0.f);

    // Inherited methods overridden:
    void run() override;

private:
    CLMemoryGroup                   _memory_group;
    CLFullyConnectedLayer           _fully_connected_input_gate;
    CLGEMM                          _gemm_input_gate;
    CLTransposeKernel               _transpose_input_gate;
    CLArithmeticAdditionKernel      _accum_input_gate1;
    CLArithmeticAddition            _accum_input_gate2;
    CLArithmeticSubtractionKernel   _subtract_input_gate;
    CLPixelWiseMultiplicationKernel _pixelwise_mul_input_gate;
    CLActivationLayerKernel         _activation_input_gate;
    CLFullyConnectedLayer           _fully_connected_forget_gate;
    CLGEMM                          _gemm_forget_gate;
    CLTransposeKernel               _transpose_forget_gate;
    CLArithmeticAdditionKernel      _accum_forget_gate1;
    CLArithmeticAddition            _accum_forget_gate2;
    CLPixelWiseMultiplicationKernel _pixelwise_mul_forget_gate;
    CLActivationLayerKernel         _activation_forget_gate;
    CLFullyConnectedLayer           _fully_connected_cell_state;
    CLGEMM                          _gemm_cell_state1;
    CLGEMM                          _gemm_cell_state2;
    CLTransposeKernel               _transpose_cell_state;
    CLArithmeticAdditionKernel      _accum_cell_state1;
    CLArithmeticAdditionKernel      _accum_cell_state2;
    CLPixelWiseMultiplicationKernel _pixelwise_mul_cell_state1;
    CLActivationLayerKernel         _activation_cell_state;
    CLActivationLayerKernel         _cell_clip;
    CLPixelWiseMultiplicationKernel _pixelwise_mul_cell_state2;
    CLFullyConnectedLayer           _fully_connected_output;
    CLGEMM                          _gemm_output;
    CLPixelWiseMultiplicationKernel _pixelwise_mul_output_state1;
    CLTransposeKernel               _transpose_output;
    CLArithmeticAdditionKernel      _accum_output1;
    CLArithmeticAddition            _accum_output2;
    CLActivationLayerKernel         _activation_output;
    CLActivationLayerKernel         _activation_output_state;
    CLPixelWiseMultiplicationKernel _pixelwise_mul_output_state2;
    CLFullyConnectedLayer           _fully_connected_output_state;
    CLGEMM                          _gemm_output_state;
    CLArithmeticAdditionKernel      _accum_output_state;
    CLActivationLayerKernel         _projection_clip;
    CLCopyKernel                    _copy_cell_state;
    CLCopyKernel                    _copy_output;
    CLWidthConcatenateLayer         _concat_scratch_buffer;
    CLTensor                        _input_gate_out1;
    CLTensor                        _input_gate_out2;
    CLTensor                        _input_gate_out3;
    CLTensor                        _input_gate_out4;
    CLTensor                        _input_gate_out5;
    CLTensor                        _forget_gate_out1;
    CLTensor                        _forget_gate_out2;
    CLTensor                        _forget_gate_out3;
    CLTensor                        _forget_gate_out4;
    CLTensor                        _forget_gate_out5;
    CLTensor                        _cell_state_out1;
    CLTensor                        _cell_state_out2;
    CLTensor                        _cell_state_out3;
    CLTensor                        _cell_state_out4;
    CLTensor                        _cell_state_out5;
    CLTensor                        _output1;
    CLTensor                        _output2;
    CLTensor                        _output3;
    CLTensor                        _output4;
    CLTensor                        _output5;
    CLTensor                        _cell_state_activation;
    CLTensor                        _output_state1;
    CLTensor                        _ones;
    bool                            _run_peephole_opt;
    bool                            _run_cifg_opt;
    bool                            _perform_cell_clipping;
    bool                            _has_projection_weights;
    bool                            _perform_projection_clipping;
};
}
#endif /* __ARM_COMPUTE_CLLSTMLAYER_H__ */
