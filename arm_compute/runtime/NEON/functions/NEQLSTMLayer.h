/*
 * Copyright (c) 2020-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_NEQLSTMLAYER_H
#define ARM_COMPUTE_NEQLSTMLAYER_H

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/NEON/functions/NEActivationLayer.h"
#include "arm_compute/runtime/NEON/functions/NEArithmeticAddition.h"
#include "arm_compute/runtime/NEON/functions/NEArithmeticSubtraction.h"
#include "arm_compute/runtime/NEON/functions/NECopy.h"
#include "arm_compute/runtime/NEON/functions/NEGEMMLowpMatrixMultiplyCore.h"
#include "arm_compute/runtime/NEON/functions/NEGEMMLowpOutputStage.h"
#include "arm_compute/runtime/NEON/functions/NEPixelWiseMultiplication.h"
#include "arm_compute/runtime/NEON/functions/NETranspose.h"
#include "arm_compute/runtime/common/LSTMParams.h"

#include <memory>

namespace arm_compute
{
// Forward declarations
class ITensor;
class ITensorInfo;
class NEQLSTMLayerNormalizationKernel;
class NEGEMMLowpMatrixAReductionKernel;

/** Basic function to run @ref NEQLSTMLayer
 *
 * This function calls the following Neon functions/kernels:
 *
 * -# @ref NEActivationLayer                                     Activation functions (tanh and logistic)
 * -# @ref NEArithmeticAddition                                  Elementwise addition
 * -# @ref NEArithmeticSubtraction                               Elementwise subtraction
 * -# @ref NECopy                                                Copy kernel for copying output_state_out to output
 * -# @ref NEGEMMLowpMatrixMultiplyCore                          Quantized matrix multiplication core. Accumulators are 32-bit integers
 * -# @ref NEGEMMLowpQuantizeDownInt32ToInt16ScaleByFixedPoint   Convert 32-bit integers into QSYMM16
 * -# @ref NEGEMMLowpMatrixAReductionKernel                      For precomputing effective biases to use
 * -# @ref NEPixelWiseMultiplication                             Elementwise multiplication
 * -# @ref NETranspose                                           Transpose function for reshaping the weights
 * */
class NEQLSTMLayer : public IFunction
{
public:
    /** Default constructor */
    NEQLSTMLayer(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEQLSTMLayer(const NEQLSTMLayer &) = delete;
    /** Prevent instances of this class from being moved (As this class contains pointers) */
    NEQLSTMLayer(NEQLSTMLayer &&) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEQLSTMLayer &operator=(const NEQLSTMLayer &) = delete;
    /** Prevent instances of this class from being moved (As this class contains pointers) */
    NEQLSTMLayer &operator=(NEQLSTMLayer &&) = delete;
    /** Default destructor */
    ~NEQLSTMLayer();
    /** Initialize function's tensors.
     *
     * @param[in]  input                       Source tensor. Input is a 2D tensor with dimensions [input_size, batch_size]. Data types supported: QASYMM8_SIGNED.
     * @param[in]  input_to_forget_weights     2D weights tensor with dimensions [input_size, num_units]. Data type supported: QSYMM8.
     * @param[in]  input_to_cell_weights       2D weights tensor with dimensions [input_size, num_units]. Data type supported: QSYMM8.
     * @param[in]  input_to_output_weights     2D weights tensor with dimensions [input_size, num_units]. Data type supported: QSYMM8.
     * @param[in]  recurrent_to_forget_weights 2D weights tensor with dimensions [output_size, num_units]. Data type supported: QSYMM8.
     * @param[in]  recurrent_to_cell_weights   2D weights tensor with dimensions [output_size, num_units]. Data type supported: QSYMM8.
     * @param[in]  recurrent_to_output_weights 2D weights tensor with dimensions [output_size, num_units]. Data type supported: QSYMM8.
     * @param[in]  forget_gate_bias            1D weights tensor with dimensions [num_units]. Data type supported: S32.
     * @param[in]  cell_bias                   1D weights tensor with dimensions [num_units]. Data type supported: S32.
     * @param[in]  output_gate_bias            1D weights tensor with dimensions [num_units]. Data type supported: S32.
     * @param[in]  cell_state_in               2D tensor with dimensions [num_units, batch_size]. Data type supported:  QSYMM16.
     * @param[in]  output_state_in             2D tensor with dimensions [output_size, batch_size]. Data type supported: Same as @p input.
     * @param[out] cell_state_out              Destination tensor. Output is a 2D tensor with dimensions [num_units, batch_size]. Data type supported:  QSYMM16.
     * @param[out] output_state_out            Destination tensor. Output is a 2D tensor with dimensions [output_size, batch_size].Data types supported: Same as @p input.
     * @param[out] output                      Destination tensor. Output is a 2D tensor with dimensions [output_size, batch_size].Data types supported: Same as @p input.
     * @param[in]  lstm_params                 Weights tensors used in peephole, CIFG and layer normalization optimizations:
     *                                         input_intermediate_scale   Scale of the intermediate result of matmul, i.e. input to layer normalization, at input gate.
     *                                         forget_intermediate_scale  Scale of the intermediate result of matmul, i.e. input to layer normalization, at forget gate.
     *                                         cell_intermediate_scale    Scale of the intermediate result of matmul, i.e. input to layer normalization, at cell gate.
     *                                         output_intermediate_scale  Scale of the intermediate result of matmul, i.e. input to layer normalization, at output gate.
     *                                         hidden_state_zero          The zero point of the hidden state.
     *                                         hidden_state_scale         The scale of the hidden state.
     *                                         input_to_input_weights     (Optional) 2D weights tensor with dimensions [input_size, num_units]. Data type supported: QSYMM8.
     *                                         recurrent_to_input_weights (Optional) 2D weights tensor with dimensions [output_size, num_units]. Data type supported: QSYMM8.
     *                                         cell_to_input_weights      (Optional) 1D weights tensor with dimensions [num_units]. Can be nullptr. Data type supported: QSYMM16.
     *                                         cell_to_forget_weights     (Optional) 1D weights tensor with dimensions [num_units]. Data type supported: QSYMM16.
     *                                         cell_to_output_weights     (Optional) 1D weights tensor with dimensions [num_units]. Data type supported: QSYMM16.
     *                                         input_gate_bias            (Optional) 1D weights tensor with dimensions [num_units]. Data type supported: S32.
     *                                         projection_weights         (Optional) 2D weights tensor with dimensions [output_size, num_units]. Data type supported: QSYMM8.
     *                                         projection_bias            (Optional) 1D weights tensor with dimensions [output_size]. S32.
     *                                         input_layer_norm_weights   (Optional) 1D weights tensor with dimensions [num_units]. Data type supported: QSYMM16.
     *                                         forget_layer_norm_weights  (Optional) 1D weights tensor with dimensions [num_units]. Data type supported: QSYMM16.
     *                                         cell_layer_norm_weights    (Optional) 1D weights tensor with dimensions [num_units]. Data type supported: QSYMM16.
     *                                         output_layer_norm_weights  (Optional) 1D weights tensor with dimensions [num_units]. Data type supported: QSYMM16.
     *                                         cell_threshold             (Optional) The clipping threshold for the cell state, such that values are bound within [-cell_clip, cell_clip].
     *                                                                               If set to 0.0 then clipping is disabled.
     *                                         projection_threshold       (Optional) The clipping threshold for the output from the projection layer, such that values are bound within
     *                                                                               [-proj_clip, proj_clip]. If set to 0.0 then clipping is disabled.
     */
    void configure(const ITensor *input,
                   const ITensor *input_to_forget_weights, const ITensor *input_to_cell_weights, const ITensor *input_to_output_weights,
                   const ITensor *recurrent_to_forget_weights, const ITensor *recurrent_to_cell_weights, const ITensor *recurrent_to_output_weights,
                   const ITensor *forget_gate_bias, const ITensor *cell_bias, const ITensor *output_gate_bias,
                   const ITensor *cell_state_in, ITensor *output_state_in,
                   ITensor *cell_state_out, ITensor *output_state_out, ITensor *output,
                   const LSTMParams<ITensor> &lstm_params);

    /** Static function to check if given info will lead to a valid configuration of @ref NEQLSTMLayer
     *
     * @param[in] input                       Source tensor info. Input is a 2D tensor info with dimensions [input_size, batch_size]. Data types supported: QASYMM8_SIGNED.
     * @param[in] input_to_forget_weights     2D weights tensor info with dimensions [input_size, num_units]. Data type supported: QSYMM8.
     * @param[in] input_to_cell_weights       2D weights tensor info with dimensions [input_size, num_units]. Data type supported: QSYMM8.
     * @param[in] input_to_output_weights     2D weights tensor info with dimensions [input_size, num_units]. Data type supported: QSYMM8.
     * @param[in] recurrent_to_forget_weights 2D weights tensor info with dimensions [output_size, num_units]. Data type supported: QSYMM8.
     * @param[in] recurrent_to_cell_weights   2D weights tensor info with dimensions [output_size, num_units]. Data type supported: QSYMM8.
     * @param[in] recurrent_to_output_weights 2D weights tensor info with dimensions [output_size, num_units]. Data type supported: QSYMM8.
     * @param[in] forget_gate_bias            1D weights tensor info with dimensions [num_units]. Data type supported: S32.
     * @param[in] cell_bias                   1D weights tensor info with dimensions [num_units]. Data type supported: S32.
     * @param[in] output_gate_bias            1D weights tensor info with dimensions [num_units]. Data type supported: S32.
     * @param[in] cell_state_in               2D tensor info with dimensions [num_units, batch_size]. Data type supported:  QSYMM16.
     * @param[in] output_state_in             2D tensor info with dimensions [output_size, batch_size]. Data type supported: Same as @p input.
     * @param[in] cell_state_out              Destination tensor info. Output is a 2D tensor info with dimensions [num_units, batch_size]. Data type supported:  QSYMM16.
     * @param[in] output_state_out            Destination tensor info. Output is a 2D tensor info with dimensions [output_size, batch_size].Data types supported: Same as @p input.
     * @param[in] output                      Destination tensor info. Output is a 2D tensor info with dimensions [output_size, batch_size].Data types supported: Same as @p input.
     * @param[in] lstm_params                 Weights tensors info used in peephole, CIFG and layer normalization optimizations:
     *                                        input_intermediate_scale   Scale of the intermediate result of matmul, i.e. input to layer normalization, at input gate.
     *                                        forget_intermediate_scale  Scale of the intermediate result of matmul, i.e. input to layer normalization, at forget gate.
     *                                        cell_intermediate_scale    Scale of the intermediate result of matmul, i.e. input to layer normalization, at cell gate.
     *                                        output_intermediate_scale  Scale of the intermediate result of matmul, i.e. input to layer normalization, at output gate.
     *                                        hidden_state_zero          The zero point of the hidden state.
     *                                        hidden_state_scale         The scale of the hidden state.
     *                                        input_to_input_weights     (Optional) 2D weights tensor with dimensions [input_size, num_units]. Data type supported: QSYMM8.
     *                                        recurrent_to_input_weights (Optional) 2D weights tensor with dimensions [output_size, num_units]. Data type supported: QSYMM8.
     *                                        cell_to_input_weights      (Optional) 1D weights tensor with dimensions [num_units]. Can be nullptr. Data type supported: QSYMM16.
     *                                        cell_to_forget_weights     (Optional) 1D weights tensor with dimensions [num_units]. Data type supported: QSYMM16.
     *                                        cell_to_output_weights     (Optional) 1D weights tensor with dimensions [num_units]. Data type supported: QSYMM16.
     *                                        input_gate_bias            (Optional) 1D weights tensor with dimensions [num_units]. Data type supported: S32.
     *                                        projection_weights         (Optional) 2D weights tensor with dimensions [output_size, num_units]. Data type supported: QSYMM8.
     *                                        projection_bias            (Optional) 1D weights tensor with dimensions [output_size]. S32.
     *                                        input_layer_norm_weights   (Optional) 1D weights tensor with dimensions [num_units]. Data type supported: QSYMM16.
     *                                        forget_layer_norm_weights  (Optional) 1D weights tensor with dimensions [num_units]. Data type supported: QSYMM16.
     *                                        cell_layer_norm_weights    (Optional) 1D weights tensor with dimensions [num_units]. Data type supported: QSYMM16.
     *                                        output_layer_norm_weights  (Optional) 1D weights tensor with dimensions [num_units]. Data type supported: QSYMM16.
     *                                        cell_threshold             (Optional) The clipping threshold for the cell state, such that values are bound within [-cell_clip, cell_clip].
     *                                                                              If set to 0.0 then clipping is disabled.
     *                                        projection_threshold       (Optional) The clipping threshold for the output from the projection layer, such that values are bound within
     *                                                                              [-proj_clip, proj_clip]. If set to 0.0 then clipping is disabled.
     * @return a status
     */
    static Status validate(const ITensorInfo *input,
                           const ITensorInfo *input_to_forget_weights, const ITensorInfo *input_to_cell_weights, const ITensorInfo *input_to_output_weights,
                           const ITensorInfo *recurrent_to_forget_weights, const ITensorInfo *recurrent_to_cell_weights, const ITensorInfo *recurrent_to_output_weights,
                           const ITensorInfo *forget_gate_bias, const ITensorInfo *cell_bias, const ITensorInfo *output_gate_bias,
                           const ITensorInfo *cell_state_in, const ITensorInfo *output_state_in,
                           const ITensorInfo *cell_state_out, const ITensorInfo *output_state_out, const ITensorInfo *output,
                           const LSTMParams<ITensorInfo> &lstm_params);

    // Inherited methods overridden:
    void run() override;
    void prepare() override;

private:
    enum class LayerNormGate : uint8_t
    {
        Forget,
        Cell,
        Input,
        Output,
        Count
    };
    static constexpr uint8_t  _layer_norm_count                    = static_cast<uint8_t>(LayerNormGate::Count);
    static constexpr uint32_t _out_state_output_size_dimension_idx = 0;

    /** Internal method to configure matrix multiplication plus output stage of each gate.
     *
     * @param[in] mm             Matrix multiplication function to use.
     * @param[in] outstage       Output stage function to use.
     * @param[in] gemmlowp_info  GEMMLowp metadata to be used by the output stage.
     * @param[in] mm_input       Input tensor to matrix multiplication function.
     * @param[in] mm_weights     Weights tensor to matrix multiplication function.
     * @param[in] bias           Bias tensor to matrix multiplication function.
     * @param[in] outstage_res   Tensor to be used for storing the result of the output stage.
     * @param[in] gemmlowp_scale Real multiplier to be used computing multiplier and shift for requantization.
     * @param[in] mm_res_info    Tensor info to be used to initialize matrix multiplication result tensor.
     * @param[in] mm_res_info    Tensor info to be used to initialize output stage result tensor.
     *
     */
    void configure_mm(NEGEMMLowpMatrixMultiplyCore &mm, NEGEMMLowpOutputStage &outstage, GEMMLowpOutputStageInfo &gemmlowp_info,
                      const ITensor *mm_input, const ITensor *mm_weights, const ITensor *bias, Tensor *mm_res,
                      Tensor *outstage_res, float gemmlowp_scale,
                      const TensorInfo &mm_res_info, const TensorInfo &outstage_tensor_info);

    MemoryGroup _memory_group;

    /** A small internel kernel do the copy between two tensors */
    class TensorCopyKernel
    {
        static constexpr uint32_t max_dimension_supported = 2;

        ITensor *_src{ nullptr };
        ITensor *_dst{ nullptr };
        size_t   _row_size{};
        Window   _window{};

    public:
        /** Destructor */
        ~TensorCopyKernel();
        /** Static function to check if given info will lead to a valid configuration of @ref NEQLSTMLayer::TensorCopyKernel
         *
         * @param[in] src Source tensor info.
         * @param[in] dst Destination tensor info
         *
         * @return a status
         */
        static Status validate(const ITensorInfo &src, const ITensorInfo &dst);
        /** Set the input and output tensors.
         *
         * @param[in]  src Source tensor
         * @param[out] dst Destination tensor
         */
        void configure(ITensor &src, ITensor &dst);
        /** run the kernel */
        void run();
    };

    // Functions used
    NETranspose                                       _transpose_input_to_forget_weights;
    NETranspose                                       _transpose_input_to_cell_weights;
    NETranspose                                       _transpose_input_to_output_weights;
    NETranspose                                       _transpose_input_to_input_weights;
    NETranspose                                       _transpose_recurrent_to_forget_weights;
    NETranspose                                       _transpose_recurrent_to_cell_weights;
    NETranspose                                       _transpose_recurrent_to_output_weights;
    NETranspose                                       _transpose_recurrent_to_input_weights;
    NETranspose                                       _transpose_projection_weights;
    std::unique_ptr<NEGEMMLowpMatrixAReductionKernel> _input_to_input_reduction;
    std::unique_ptr<NEGEMMLowpMatrixAReductionKernel> _recurrent_to_input_reduction;
    std::unique_ptr<NEGEMMLowpMatrixAReductionKernel> _input_to_forget_reduction;
    std::unique_ptr<NEGEMMLowpMatrixAReductionKernel> _recurrent_to_forget_reduction;
    std::unique_ptr<NEGEMMLowpMatrixAReductionKernel> _input_to_cell_reduction;
    std::unique_ptr<NEGEMMLowpMatrixAReductionKernel> _recurrent_to_cell_reduction;
    std::unique_ptr<NEGEMMLowpMatrixAReductionKernel> _input_to_output_reduction;
    std::unique_ptr<NEGEMMLowpMatrixAReductionKernel> _recurrent_to_output_reduction;
    std::unique_ptr<NEGEMMLowpMatrixAReductionKernel> _projection_reduction;
    NEArithmeticAddition                              _projection_bias_add;
    NEGEMMLowpMatrixMultiplyCore                      _mm_input_to_forget;
    NEGEMMLowpMatrixMultiplyCore                      _mm_recurrent_to_forget;
    NEPixelWiseMultiplication                         _pixelwise_mul_cell_to_forget;
    NEGEMMLowpOutputStage                             _input_to_forget_outstage;
    NEGEMMLowpOutputStage                             _recurrent_to_forget_outstage;
    NEGEMMLowpOutputStage                             _cell_to_forget_outstage;
    NEArithmeticAddition                              _accumulate_input_recurrent_forget;
    NEArithmeticAddition                              _accumulate_cell_forget;
    NEActivationLayer                                 _forget_gate_sigmoid;
    NEGEMMLowpMatrixMultiplyCore                      _mm_input_to_cell;
    NEGEMMLowpOutputStage                             _input_to_cell_outstage;
    NEGEMMLowpMatrixMultiplyCore                      _mm_recurrent_to_cell;
    NEGEMMLowpOutputStage                             _recurrent_to_cell_outstage;
    NEArithmeticAddition                              _accumulate_input_recurrent_modulation;
    NEActivationLayer                                 _cell_gate_tanh;
    NEArithmeticSubtraction                           _input_gate_sub;
    NEGEMMLowpMatrixMultiplyCore                      _mm_input_to_input;
    NEGEMMLowpOutputStage                             _input_to_input_outstage;
    NEGEMMLowpMatrixMultiplyCore                      _mm_recurrent_to_input;
    NEGEMMLowpOutputStage                             _recurrent_to_input_outstage;
    NEArithmeticAddition                              _accumulate_input_recurrent_input;
    NEPixelWiseMultiplication                         _pixelwise_mul_cell_to_input;
    NEGEMMLowpOutputStage                             _cell_to_input_outstage;
    NEArithmeticAddition                              _accumulate_cell_input;
    NEActivationLayer                                 _input_gate_sigmoid;
    NEPixelWiseMultiplication                         _pixelwise_mul_forget_cell;
    NEPixelWiseMultiplication                         _pixelwise_mul_input_cell;
    NEArithmeticAddition                              _add_forget_cell;
    NEActivationLayer                                 _cell_clip;
    NEGEMMLowpMatrixMultiplyCore                      _mm_input_to_output;
    NEGEMMLowpOutputStage                             _input_to_output_outstage;
    NEGEMMLowpMatrixMultiplyCore                      _mm_recurrent_to_output;
    NEGEMMLowpOutputStage                             _recurrent_to_output_outstage;
    NEArithmeticAddition                              _accumulate_input_recurrent_output;
    NEPixelWiseMultiplication                         _pixelwise_mul_cell_to_output;
    NEGEMMLowpOutputStage                             _cell_to_output_outstage;
    NEArithmeticAddition                              _accumulate_cell_to_output;
    NEActivationLayer                                 _output_gate_sigmoid;
    NEActivationLayer                                 _hidden_tanh;
    NEPixelWiseMultiplication                         _pixelwise_mul_hidden;
    NEGEMMLowpOutputStage                             _hidden_outstage;
    NEGEMMLowpMatrixMultiplyCore                      _mm_projection;
    NEGEMMLowpOutputStage                             _projection_outstage;
    NEArithmeticAddition                              _accumulate_projection;
    NEActivationLayer                                 _projection_clip;

    TensorCopyKernel _projection_bias_copy;
    TensorCopyKernel _projection_output_to_accumulate_copy;
    TensorCopyKernel _projection_accumulate_to_output_copy;
    TensorCopyKernel _hidden_to_output_copy;

    std::array<std::unique_ptr<NEQLSTMLayerNormalizationKernel>, _layer_norm_count> _layer_norms;

    NECopy _copy_output;

    // Tensor pointers
    const ITensor *_input_to_input_weights
    {
        nullptr
    };
    const ITensor *_recurrent_to_input_weights{ nullptr };
    const ITensor *_projection_bias{ nullptr };
    const ITensor *_input_to_forget_weights{ nullptr };
    const ITensor *_input_to_cell_weights{ nullptr };
    const ITensor *_input_to_output_weights{ nullptr };
    const ITensor *_recurrent_to_forget_weights{ nullptr };
    const ITensor *_recurrent_to_cell_weights{ nullptr };
    const ITensor *_recurrent_to_output_weights{ nullptr };
    const ITensor *_projection_weights{ nullptr };
    std::array<const ITensor *, _layer_norm_count> _layer_norm_weights{};
    std::array<const ITensor *, _layer_norm_count> _layer_norm_bias{};

    using LayerNormIndexType = typename std::underlying_type<LayerNormGate>::type;
    inline LayerNormIndexType getGateIndex(LayerNormGate g)
    {
        return static_cast<LayerNormIndexType>(g);
    }

    inline void set_layer_norm_weight(const ITensor *t, LayerNormGate g)
    {
        _layer_norm_weights[getGateIndex(g)] = t;
    }

    inline void set_layer_norm_bias(const ITensor *t, LayerNormGate g)
    {
        _layer_norm_bias[getGateIndex(g)] = t;
    }

    inline const ITensor *get_layer_norm_weight(LayerNormGate g)
    {
        return _layer_norm_weights[getGateIndex(g)];
    }

    inline const ITensor *get_layer_norm_bias(LayerNormGate g)
    {
        return _layer_norm_bias[getGateIndex(g)];
    }

    inline std::unique_ptr<NEQLSTMLayerNormalizationKernel> &get_layer_norm(LayerNormGate g)
    {
        return _layer_norms[getGateIndex(g)];
    }

    void configure_layer_norm(LayerNormGate g, const ITensor *in);
    static Status validate_layer_norm(const ITensorInfo &in, const ITensorInfo &weight, const ITensorInfo &bias);

    // Temporary tensors
    Tensor _input_to_forget_weights_transposed{ nullptr };
    Tensor _input_to_cell_weights_transposed{ nullptr };
    Tensor _input_to_output_weights_transposed{ nullptr };
    Tensor _input_to_input_weights_transposed{ nullptr };
    Tensor _recurrent_to_forget_weights_transposed{ nullptr };
    Tensor _recurrent_to_cell_weights_transposed{ nullptr };
    Tensor _recurrent_to_output_weights_transposed{ nullptr };
    Tensor _recurrent_to_input_weights_transposed{ nullptr };
    Tensor _projection_weights_transposed{ nullptr };
    Tensor _input_to_input_eff_bias{ nullptr };
    Tensor _recurrent_to_input_eff_bias{ nullptr };
    Tensor _input_to_forget_eff_bias{ nullptr };
    Tensor _recurrent_to_forget_eff_bias{ nullptr };
    Tensor _input_to_cell_eff_bias{ nullptr };
    Tensor _recurrent_to_cell_eff_bias{ nullptr };
    Tensor _input_to_output_eff_bias{ nullptr };
    Tensor _recurrent_to_output_eff_bias{ nullptr };
    Tensor _projection_reduction_res{ nullptr };
    Tensor _projection_eff_bias{ nullptr };
    Tensor _mm_input_to_forget_res{ nullptr };
    Tensor _mm_recurrent_to_forget_res{ nullptr };
    Tensor _mul_cell_to_forget_res{ nullptr };
    Tensor _input_to_forget_outstage_res{ nullptr };
    Tensor _cell_to_forget_outstage_res{ nullptr };
    Tensor _recurrent_to_forget_outstage_res{ nullptr };
    Tensor _forget_gate{ nullptr };
    Tensor _mm_input_to_cell_res{ nullptr };
    Tensor _input_to_cell_outstage_res{ nullptr };
    Tensor _mm_recurrent_to_cell_res{ nullptr };
    Tensor _recurrent_to_cell_outstage_res{ nullptr };
    Tensor _cell_gate{ nullptr };
    Tensor _mul_input_cell_res{ nullptr };
    Tensor _mm_input_to_input_res{ nullptr };
    Tensor _input_to_input_outstage_res{ nullptr };
    Tensor _mm_recurrent_to_input_res{ nullptr };
    Tensor _mul_cell_to_input_res{ nullptr };
    Tensor _cell_to_input_outstage_res{ nullptr };
    Tensor _recurrent_to_input_outstage_res{ nullptr };
    Tensor _input_gate{ nullptr };
    Tensor _mm_input_to_output_res{ nullptr };
    Tensor _input_to_output_outstage_res{ nullptr };
    Tensor _mm_recurrent_to_output_res{ nullptr };
    Tensor _mul_cell_to_output_res{ nullptr };
    Tensor _cell_to_output_outstage_res{ nullptr };
    Tensor _recurrent_to_output_outstage_res{ nullptr };
    Tensor _output_gate{ nullptr };
    Tensor _hidden_mul_res{ nullptr };
    Tensor _hidden_gate{ nullptr };
    Tensor _mm_projection_res{ nullptr };
    Tensor _projection_outstage_res{ nullptr };
    Tensor _projection_out_res{ nullptr };
    Tensor _projection_accumulate_res{ nullptr };
    Tensor _ones{ nullptr };
    std::array<Tensor, _layer_norm_count> _layer_norm_output{};

    inline Tensor &get_layer_norm_output(LayerNormGate g)
    {
        return _layer_norm_output[getGateIndex(g)];
    }

    bool _is_prepared{ false };
    bool _has_cifg{ false };
    bool _has_cell_clipping{ false };
    bool _has_projection{ false };
    bool _has_projection_clipping{ false };
    bool _has_peephole{ false };
    bool _has_layer_norm{ false };
    bool _projection_tensor_copy_required{ false };
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_NEQLSTMLAYER_H */
