/*
 * Copyright (c) 2018-2022 Arm Limited.
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
#ifndef ARM_COMPUTE_TEST_LSTM_LAYER_FIXTURE
#define ARM_COMPUTE_TEST_LSTM_LAYER_FIXTURE

#include "tests/Globals.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/reference/ActivationLayer.h"
#include "tests/validation/reference/ArithmeticOperations.h"
#include "tests/validation/reference/ConcatenateLayer.h"
#include "tests/validation/reference/FullyConnectedLayer.h"
#include "tests/validation/reference/GEMM.h"
#include "tests/validation/reference/MeanStdDevNormalizationLayer.h"
#include "tests/validation/reference/PixelWiseMultiplication.h"
#include "tests/validation/reference/Transpose.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename FunctionType, typename FunctionParams, typename T>
class LSTMLayerValidationFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape input_shape, TensorShape input_weights_shape, TensorShape recurrent_weights_shape, TensorShape cell_bias_shape, TensorShape output_cell_shape, TensorShape output_shape,
               TensorShape scratch_shape, ActivationLayerInfo info, float cell_threshold, float projection_threshold, DataType data_type, bool projection_opt, bool peephole_opt,
               bool use_layer_norm)
    {
        _target = compute_target(input_shape, input_weights_shape, recurrent_weights_shape, cell_bias_shape, output_cell_shape, output_shape, scratch_shape, info, cell_threshold, projection_threshold,
                                 data_type, projection_opt, peephole_opt, use_layer_norm);
        _reference = compute_reference(input_shape, input_weights_shape, recurrent_weights_shape, cell_bias_shape, output_cell_shape, output_shape, scratch_shape, info, cell_threshold, projection_threshold,
                                       data_type, projection_opt, peephole_opt, use_layer_norm);
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i)
    {
        static_assert(std::is_floating_point<T>::value || std::is_same<T, half>::value, "Only floating point data types supported.");
        using DistributionType = typename std::conditional<std::is_same<T, half>::value, arm_compute::utils::uniform_real_distribution_16bit<T>, std::uniform_real_distribution<T>>::type;

        DistributionType distribution{ T(-1.0f), T(1.0f) };
        library->fill(tensor, distribution, i);
    }
    template <typename U>
    void fill_custom_val(U &&tensor, float num, int i)
    {
        static_assert(std::is_floating_point<T>::value || std::is_same<T, half>::value, "Only floating point data types supported.");
        using DistributionType = typename std::conditional<std::is_same<T, half>::value, arm_compute::utils::uniform_real_distribution_16bit<T>, std::uniform_real_distribution<T>>::type;

        DistributionType distribution{ T(num), T(num) };
        library->fill(tensor, distribution, i);
    }
    TensorType compute_target(const TensorShape &input_shape, const TensorShape &input_weights_shape, const TensorShape &recurrent_weights_shape, const TensorShape &cell_bias_shape,
                              const TensorShape &output_cell_shape, const TensorShape &output_shape, const TensorShape &scratch_shape, ActivationLayerInfo info, float cell_threshold,
                              float projection_threshold, DataType data_type, bool projection_opt, bool peephole_opt, bool use_layer_norm)
    {
        const unsigned int num_cells   = input_weights_shape.y();
        const unsigned int num_outputs = recurrent_weights_shape.x();

        // Create tensors
        TensorType input                 = create_tensor<TensorType>(input_shape, data_type);
        TensorType input_to_forget_w     = create_tensor<TensorType>(input_weights_shape, data_type);
        TensorType input_to_cell_w       = create_tensor<TensorType>(input_weights_shape, data_type);
        TensorType input_to_output_w     = create_tensor<TensorType>(input_weights_shape, data_type);
        TensorType recurrent_to_forget_w = create_tensor<TensorType>(recurrent_weights_shape, data_type);
        TensorType recurrent_to_cell_w   = create_tensor<TensorType>(recurrent_weights_shape, data_type);
        TensorType recurrent_to_output_w = create_tensor<TensorType>(recurrent_weights_shape, data_type);
        TensorType forget_gate_bias      = create_tensor<TensorType>(cell_bias_shape, data_type);
        TensorType cell_bias             = create_tensor<TensorType>(cell_bias_shape, data_type);
        TensorType output_gate_bias      = create_tensor<TensorType>(cell_bias_shape, data_type);
        TensorType output_state_in       = create_tensor<TensorType>(output_shape, data_type);
        TensorType cell_state_in         = create_tensor<TensorType>(output_cell_shape, data_type);
        TensorType scratch               = create_tensor<TensorType>(scratch_shape, data_type);
        TensorType output_state_out      = create_tensor<TensorType>(output_shape, data_type);
        TensorType cell_state_out        = create_tensor<TensorType>(output_cell_shape, data_type);
        TensorType output                = create_tensor<TensorType>(output_shape, data_type);
        TensorType input_to_input_w;
        TensorType recurrent_to_input_w;
        TensorType cell_to_input_w;
        TensorType cell_to_forget_w;
        TensorType input_gate_bias;
        TensorType cell_to_output_w;
        TensorType projection_w;
        TensorType projection_bias;
        TensorType input_layer_norm_w;
        TensorType forget_layer_norm_w;
        TensorType cell_layer_norm_w;
        TensorType output_layer_norm_w;

        bool cifg_opt = scratch_shape.x() == cell_bias_shape.x() * 4 ? false : true;

        FunctionParams lstm_params;

        if(!cifg_opt)
        {
            input_to_input_w     = create_tensor<TensorType>(input_weights_shape, data_type);
            recurrent_to_input_w = create_tensor<TensorType>(recurrent_weights_shape, data_type);
            if(peephole_opt)
            {
                cell_to_input_w = create_tensor<TensorType>(cell_bias_shape, data_type);
            }
            input_gate_bias = create_tensor<TensorType>(cell_bias_shape, data_type);
            lstm_params.set_cifg_params(&input_to_input_w, &recurrent_to_input_w, &cell_to_input_w, &input_gate_bias);
        }

        if(peephole_opt)
        {
            cell_to_forget_w = create_tensor<TensorType>(cell_bias_shape, data_type);
            cell_to_output_w = create_tensor<TensorType>(cell_bias_shape, data_type);
            lstm_params.set_peephole_params(&cell_to_forget_w, &cell_to_output_w);
        }

        if(projection_opt)
        {
            projection_w    = create_tensor<TensorType>(TensorShape(num_cells, num_outputs), data_type);
            projection_bias = create_tensor<TensorType>(TensorShape(num_outputs), data_type);
            lstm_params.set_projection_params(&projection_w, &projection_bias);
        }

        if(use_layer_norm)
        {
            forget_layer_norm_w = create_tensor<TensorType>(TensorShape(num_cells), data_type);
            cell_layer_norm_w   = create_tensor<TensorType>(TensorShape(num_cells), data_type);
            output_layer_norm_w = create_tensor<TensorType>(TensorShape(num_cells), data_type);
            if(!cifg_opt)
            {
                input_layer_norm_w = create_tensor<TensorType>(TensorShape(num_cells), data_type);
                lstm_params.set_layer_normalization_params(&input_layer_norm_w, &forget_layer_norm_w, &cell_layer_norm_w, &output_layer_norm_w);
            }
            else
            {
                lstm_params.set_layer_normalization_params(nullptr, &forget_layer_norm_w, &cell_layer_norm_w, &output_layer_norm_w);
            }
        }

        // Create and configure function
        FunctionType lstm;
        lstm.configure(&input, &input_to_forget_w, &input_to_cell_w, &input_to_output_w, &recurrent_to_forget_w,
                       &recurrent_to_cell_w, &recurrent_to_output_w, &forget_gate_bias, &cell_bias, &output_gate_bias,
                       &output_state_in, &cell_state_in,
                       &scratch, &output_state_out, &cell_state_out, &output,
                       lstm_params, info, cell_threshold, projection_threshold);

        ARM_COMPUTE_ASSERT(input.info()->is_resizable());
        ARM_COMPUTE_ASSERT(input_to_forget_w.info()->is_resizable());
        ARM_COMPUTE_ASSERT(input_to_cell_w.info()->is_resizable());
        ARM_COMPUTE_ASSERT(input_to_output_w.info()->is_resizable());
        ARM_COMPUTE_ASSERT(recurrent_to_forget_w.info()->is_resizable());
        ARM_COMPUTE_ASSERT(recurrent_to_cell_w.info()->is_resizable());
        ARM_COMPUTE_ASSERT(recurrent_to_output_w.info()->is_resizable());
        ARM_COMPUTE_ASSERT(forget_gate_bias.info()->is_resizable());
        ARM_COMPUTE_ASSERT(cell_bias.info()->is_resizable());
        ARM_COMPUTE_ASSERT(output_gate_bias.info()->is_resizable());
        ARM_COMPUTE_ASSERT(output_state_in.info()->is_resizable());
        ARM_COMPUTE_ASSERT(cell_state_in.info()->is_resizable());
        ARM_COMPUTE_ASSERT(scratch.info()->is_resizable());
        ARM_COMPUTE_ASSERT(output_state_out.info()->is_resizable());
        ARM_COMPUTE_ASSERT(cell_state_out.info()->is_resizable());
        ARM_COMPUTE_ASSERT(output.info()->is_resizable());

        // Allocate tensors
        input.allocator()->allocate();
        input_to_forget_w.allocator()->allocate();
        input_to_cell_w.allocator()->allocate();
        input_to_output_w.allocator()->allocate();
        recurrent_to_forget_w.allocator()->allocate();
        recurrent_to_cell_w.allocator()->allocate();
        recurrent_to_output_w.allocator()->allocate();
        forget_gate_bias.allocator()->allocate();
        cell_bias.allocator()->allocate();
        output_gate_bias.allocator()->allocate();
        output_state_in.allocator()->allocate();
        cell_state_in.allocator()->allocate();
        scratch.allocator()->allocate();
        output_state_out.allocator()->allocate();
        cell_state_out.allocator()->allocate();
        output.allocator()->allocate();

        ARM_COMPUTE_ASSERT(!input.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!input_to_forget_w.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!input_to_cell_w.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!input_to_output_w.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!recurrent_to_forget_w.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!recurrent_to_cell_w.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!recurrent_to_output_w.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!forget_gate_bias.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!cell_bias.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!output_gate_bias.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!output_state_in.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!cell_state_in.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!scratch.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!output_state_out.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!cell_state_out.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!output.info()->is_resizable());

        // Fill tensors
        fill(AccessorType(input), 0);
        fill(AccessorType(input_to_forget_w), 1);
        fill(AccessorType(input_to_cell_w), 2);
        fill(AccessorType(input_to_output_w), 3);
        fill(AccessorType(recurrent_to_forget_w), 4);
        fill(AccessorType(recurrent_to_cell_w), 5);
        fill(AccessorType(recurrent_to_output_w), 6);
        fill(AccessorType(forget_gate_bias), 7);
        fill(AccessorType(cell_bias), 8);
        fill(AccessorType(output_gate_bias), 9);
        fill(AccessorType(output_state_in), 10);
        fill(AccessorType(cell_state_in), 11);
        fill(AccessorType(scratch), 12);

        if(!cifg_opt)
        {
            ARM_COMPUTE_ASSERT(input_to_input_w.info()->is_resizable());
            ARM_COMPUTE_ASSERT(recurrent_to_input_w.info()->is_resizable());
            ARM_COMPUTE_ASSERT(cell_to_input_w.info()->is_resizable());
            ARM_COMPUTE_ASSERT(input_gate_bias.info()->is_resizable());
            input_to_input_w.allocator()->allocate();
            recurrent_to_input_w.allocator()->allocate();
            cell_to_input_w.allocator()->allocate();
            input_gate_bias.allocator()->allocate();
            ARM_COMPUTE_ASSERT(!input_to_input_w.info()->is_resizable());
            ARM_COMPUTE_ASSERT(!recurrent_to_input_w.info()->is_resizable());
            ARM_COMPUTE_ASSERT(!cell_to_input_w.info()->is_resizable());
            ARM_COMPUTE_ASSERT(!input_gate_bias.info()->is_resizable());
            fill(AccessorType(input_to_input_w), 13);
            fill(AccessorType(recurrent_to_input_w), 14);
            if(peephole_opt)
            {
                fill(AccessorType(cell_to_input_w), 15);
            }
            fill(AccessorType(recurrent_to_input_w), 16);
            fill(AccessorType(input_gate_bias), 17);
        }

        if(peephole_opt)
        {
            ARM_COMPUTE_ASSERT(cell_to_forget_w.info()->is_resizable());
            ARM_COMPUTE_ASSERT(cell_to_output_w.info()->is_resizable());
            cell_to_forget_w.allocator()->allocate();
            cell_to_output_w.allocator()->allocate();
            ARM_COMPUTE_ASSERT(!cell_to_forget_w.info()->is_resizable());
            ARM_COMPUTE_ASSERT(!cell_to_output_w.info()->is_resizable());
            fill(AccessorType(cell_to_forget_w), 18);
            fill(AccessorType(cell_to_output_w), 19);
        }

        if(projection_opt)
        {
            ARM_COMPUTE_ASSERT(projection_w.info()->is_resizable());
            ARM_COMPUTE_ASSERT(projection_bias.info()->is_resizable());

            projection_w.allocator()->allocate();
            projection_bias.allocator()->allocate();

            ARM_COMPUTE_ASSERT(!projection_w.info()->is_resizable());
            ARM_COMPUTE_ASSERT(!projection_bias.info()->is_resizable());

            fill(AccessorType(projection_w), 20);
            fill(AccessorType(projection_bias), 21);
        }

        if(use_layer_norm)
        {
            if(!cifg_opt)
            {
                ARM_COMPUTE_ASSERT(input_layer_norm_w.info()->is_resizable());

                input_layer_norm_w.allocator()->allocate();

                ARM_COMPUTE_ASSERT(!input_layer_norm_w.info()->is_resizable());

                fill(AccessorType(input_layer_norm_w), 22);
            }
            ARM_COMPUTE_ASSERT(forget_layer_norm_w.info()->is_resizable());
            ARM_COMPUTE_ASSERT(cell_layer_norm_w.info()->is_resizable());
            ARM_COMPUTE_ASSERT(output_layer_norm_w.info()->is_resizable());

            forget_layer_norm_w.allocator()->allocate();
            cell_layer_norm_w.allocator()->allocate();
            output_layer_norm_w.allocator()->allocate();

            ARM_COMPUTE_ASSERT(!forget_layer_norm_w.info()->is_resizable());
            ARM_COMPUTE_ASSERT(!cell_layer_norm_w.info()->is_resizable());
            ARM_COMPUTE_ASSERT(!output_layer_norm_w.info()->is_resizable());

            fill(AccessorType(forget_layer_norm_w), 23);
            fill(AccessorType(cell_layer_norm_w), 24);
            fill(AccessorType(output_layer_norm_w), 25);
        }

        // Compute function
        lstm.run();

        _target_scratch = std::move(scratch);
        return output;
    }

    SimpleTensor<T> compute_reference(const TensorShape &input_shape, const TensorShape &input_weights_shape, const TensorShape &recurrent_weights_shape, const TensorShape &cell_bias_shape,
                                      const TensorShape &output_cell_shape, const TensorShape &output_shape, const TensorShape &scratch_shape, ActivationLayerInfo info, float cell_threshold,
                                      float projection_threshold, DataType data_type, bool projection_opt, bool peephole_opt, bool use_layer_norm)
    {
        const unsigned int num_cells   = input_weights_shape.y();
        const unsigned int num_outputs = recurrent_weights_shape.x();

        // Create projection weights shape
        TensorShape projection_weights_shape(num_cells, num_outputs);

        // Create projection bias shape
        TensorShape projection_bias_shape(num_outputs);

        TensorShape     gemm_shape{ 1, output_shape.y() };
        SimpleTensor<T> gemm_out{ gemm_shape, data_type };

        // Create reference
        SimpleTensor<T> input{ input_shape, data_type };
        SimpleTensor<T> input_to_input_w{ input_weights_shape, data_type };
        SimpleTensor<T> input_to_forget_w{ input_weights_shape, data_type };
        SimpleTensor<T> input_to_cell_w{ input_weights_shape, data_type };
        SimpleTensor<T> input_to_output_w{ input_weights_shape, data_type };
        SimpleTensor<T> recurrent_to_input_w{ recurrent_weights_shape, data_type };
        SimpleTensor<T> recurrent_to_forget_w{ recurrent_weights_shape, data_type };
        SimpleTensor<T> recurrent_to_cell_w{ recurrent_weights_shape, data_type };
        SimpleTensor<T> recurrent_to_output_w{ recurrent_weights_shape, data_type };
        SimpleTensor<T> cell_to_input_w{ cell_bias_shape, data_type };
        SimpleTensor<T> cell_to_forget_w{ cell_bias_shape, data_type };
        SimpleTensor<T> cell_to_output_w{ cell_bias_shape, data_type };
        SimpleTensor<T> input_gate_bias{ cell_bias_shape, data_type };
        SimpleTensor<T> forget_gate_bias{ cell_bias_shape, data_type };
        SimpleTensor<T> cell_bias{ cell_bias_shape, data_type };
        SimpleTensor<T> output_gate_bias{ cell_bias_shape, data_type };
        SimpleTensor<T> projection_w{ projection_weights_shape, data_type };
        SimpleTensor<T> projection_bias{ projection_bias_shape, data_type };
        SimpleTensor<T> output_state_in{ output_shape, data_type };
        SimpleTensor<T> cell_state_in{ output_cell_shape, data_type };
        SimpleTensor<T> scratch{ scratch_shape, data_type };
        SimpleTensor<T> output_state_out{ output_shape, data_type };
        SimpleTensor<T> cell_state_out{ output_cell_shape, data_type };
        SimpleTensor<T> output{ output_shape, data_type };

        bool cifg_opt = scratch_shape.x() == cell_bias_shape.x() * 4 ? false : true;

        // Fill reference
        fill(input, 0);
        fill(input_to_forget_w, 1);
        fill(input_to_cell_w, 2);
        fill(input_to_output_w, 3);
        fill(recurrent_to_forget_w, 4);
        fill(recurrent_to_cell_w, 5);
        fill(recurrent_to_output_w, 6);
        if(use_layer_norm)
        {
            fill_custom_val(forget_gate_bias, 0.f, 7);
            fill_custom_val(cell_bias, 0.f, 8);
            fill_custom_val(output_gate_bias, 0.f, 9);
        }
        else
        {
            fill(forget_gate_bias, 7);
            fill(cell_bias, 8);
            fill(output_gate_bias, 9);
        }
        fill(output_state_in, 10);
        fill(cell_state_in, 11);
        fill(scratch, 12);
        fill(input_to_input_w, 13);
        fill(recurrent_to_input_w, 14);
        fill(cell_to_input_w, 15);
        fill(recurrent_to_input_w, 16);
        if(!cifg_opt && use_layer_norm)
        {
            fill_custom_val(input_gate_bias, 0.f, 17);
        }
        else
        {
            fill(input_gate_bias, 17);
        }
        fill(cell_to_forget_w, 18);
        fill(cell_to_output_w, 19);
        fill(projection_w, 20);
        fill(projection_bias, 21);

        // Compute forget_gate
        SimpleTensor<T> fully_connected_forget = reference::fully_connected_layer(input, input_to_forget_w, forget_gate_bias, output_cell_shape);
        SimpleTensor<T> transposed_weights     = reference::transpose(recurrent_to_forget_w);
        SimpleTensor<T> gemm                   = reference::gemm(output_state_in, transposed_weights, cell_state_in, 1.f, 0.f);
        SimpleTensor<T> forget_gate            = reference::arithmetic_operation(reference::ArithmeticOperation::ADD, fully_connected_forget, gemm, data_type, ConvertPolicy::SATURATE);

        if(peephole_opt)
        {
            SimpleTensor<T> pixelwise_mul_forget_gate = reference::pixel_wise_multiplication<T, T, T>(cell_state_in, cell_to_forget_w, 1, ConvertPolicy::SATURATE, RoundingPolicy::TO_ZERO, data_type);
            forget_gate                               = reference::arithmetic_operation(reference::ArithmeticOperation::ADD, forget_gate, pixelwise_mul_forget_gate, data_type, ConvertPolicy::SATURATE);
        }

        if(use_layer_norm)
        {
            SimpleTensor<T> forget_layer_norm_w{ cell_bias_shape, data_type };
            fill(forget_layer_norm_w, 23);
            forget_gate = reference::mean_std_normalization_layer(forget_gate);
            forget_gate = reference::pixel_wise_multiplication<T, T, T>(forget_gate, forget_layer_norm_w, 1, ConvertPolicy::SATURATE, RoundingPolicy::TO_NEAREST_EVEN, data_type);
            fill(forget_gate_bias, 7);
            forget_gate = reference::arithmetic_operation(reference::ArithmeticOperation::ADD, forget_gate, forget_gate_bias, data_type, ConvertPolicy::SATURATE);
        }
        forget_gate = reference::activation_layer(forget_gate, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LOGISTIC));

        // Compute input_gate
        SimpleTensor<T> input_gate;
        if(cifg_opt)
        {
            SimpleTensor<T> ones{ cell_bias_shape, data_type };
            fill_custom_val(ones, 1.f, 0);
            input_gate = reference::arithmetic_operation<T>(reference::ArithmeticOperation::SUB, ones, forget_gate, data_type, ConvertPolicy::SATURATE);
        }
        else
        {
            SimpleTensor<T> fully_connected_input = reference::fully_connected_layer(input, input_to_input_w, input_gate_bias, output_cell_shape);
            transposed_weights                    = reference::transpose(recurrent_to_input_w);
            gemm                                  = reference::gemm(output_state_in, transposed_weights, cell_state_in, 1.f, 0.f);
            input_gate                            = reference::arithmetic_operation(reference::ArithmeticOperation::ADD, fully_connected_input, gemm, data_type, ConvertPolicy::SATURATE);
            if(peephole_opt)
            {
                SimpleTensor<T> pixelwise_mul_input_gate = reference::pixel_wise_multiplication<T, T, T>(cell_state_in, cell_to_input_w, 1, ConvertPolicy::SATURATE, RoundingPolicy::TO_NEAREST_EVEN, data_type);
                input_gate                               = reference::arithmetic_operation(reference::ArithmeticOperation::ADD, input_gate, pixelwise_mul_input_gate, data_type, ConvertPolicy::SATURATE);
            }
            if(use_layer_norm)
            {
                SimpleTensor<T> input_layer_norm_w{ cell_bias_shape, data_type };
                fill(input_layer_norm_w, 22);
                input_gate = reference::mean_std_normalization_layer(input_gate);
                input_gate = reference::pixel_wise_multiplication<T, T, T>(input_gate, input_layer_norm_w, 1, ConvertPolicy::SATURATE, RoundingPolicy::TO_NEAREST_EVEN, data_type);
                fill(input_gate_bias, 17);
                input_gate = reference::arithmetic_operation(reference::ArithmeticOperation::ADD, input_gate, input_gate_bias, data_type, ConvertPolicy::SATURATE);
            }
            input_gate = reference::activation_layer(input_gate, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LOGISTIC));
        }
        // Compute cell_state
        SimpleTensor<T> fully_connected_cell_state = reference::fully_connected_layer(input, input_to_cell_w, cell_bias, output_cell_shape);
        transposed_weights                         = reference::transpose(recurrent_to_cell_w);
        gemm                                       = reference::gemm(output_state_in, transposed_weights, cell_state_out, 1.f, 0.f);
        SimpleTensor<T> pixelwise_mul              = reference::pixel_wise_multiplication<T, T, T>(cell_state_in, forget_gate, 1, ConvertPolicy::SATURATE, RoundingPolicy::TO_NEAREST_EVEN, data_type);
        cell_state_out                             = reference::arithmetic_operation(reference::ArithmeticOperation::ADD, fully_connected_cell_state, gemm, data_type, ConvertPolicy::SATURATE);
        if(use_layer_norm)
        {
            SimpleTensor<T> cell_layer_norm_w{ cell_bias_shape, data_type };
            fill(cell_layer_norm_w, 24);
            cell_state_out = reference::mean_std_normalization_layer(cell_state_out);
            cell_state_out = reference::pixel_wise_multiplication<T, T, T>(cell_state_out, cell_layer_norm_w, 1, ConvertPolicy::SATURATE, RoundingPolicy::TO_NEAREST_EVEN, data_type);
            fill(cell_bias, 8);
            cell_state_out = reference::arithmetic_operation(reference::ArithmeticOperation::ADD, cell_state_out, cell_bias, data_type, ConvertPolicy::SATURATE);
        }
        cell_state_out = reference::activation_layer(cell_state_out, info);
        cell_state_out = reference::pixel_wise_multiplication<T, T, T>(cell_state_out, input_gate, 1, ConvertPolicy::SATURATE, RoundingPolicy::TO_NEAREST_EVEN, data_type);
        cell_state_out = reference::arithmetic_operation(reference::ArithmeticOperation::ADD, cell_state_out, pixelwise_mul, data_type, ConvertPolicy::SATURATE);

        if(cell_threshold != 0.f)
        {
            cell_state_out = reference::activation_layer(cell_state_out, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, cell_threshold, -cell_threshold));
        }

        // Compute output
        SimpleTensor<T> fully_connected_output = reference::fully_connected_layer(input, input_to_output_w, output_gate_bias, output_cell_shape);
        transposed_weights                     = reference::transpose(recurrent_to_output_w);
        gemm                                   = reference::gemm(output_state_in, transposed_weights, cell_state_out, 1.f, 0.f);
        output                                 = reference::arithmetic_operation(reference::ArithmeticOperation::ADD, fully_connected_output, gemm, data_type, ConvertPolicy::SATURATE);
        if(peephole_opt)
        {
            pixelwise_mul = reference::pixel_wise_multiplication<T, T, T>(cell_state_out, cell_to_output_w, 1, ConvertPolicy::SATURATE, RoundingPolicy::TO_NEAREST_EVEN, data_type);
            output        = reference::arithmetic_operation(reference::ArithmeticOperation::ADD, output, pixelwise_mul, data_type, ConvertPolicy::SATURATE);
        }
        if(use_layer_norm)
        {
            SimpleTensor<T> output_layer_norm_w{ cell_bias_shape, data_type };
            fill(output_layer_norm_w, 25);
            output = reference::mean_std_normalization_layer(output);
            output = reference::pixel_wise_multiplication<T, T, T>(output, output_layer_norm_w, 1, ConvertPolicy::SATURATE, RoundingPolicy::TO_NEAREST_EVEN, data_type);
            fill(output_gate_bias, 9);
            output = reference::arithmetic_operation(reference::ArithmeticOperation::ADD, output, output_gate_bias, data_type, ConvertPolicy::SATURATE);
        }
        output = reference::activation_layer(output, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LOGISTIC));

        // Compute output state
        SimpleTensor<T> cell_state_activation = reference::activation_layer(cell_state_out, info);
        output_state_out                      = reference::pixel_wise_multiplication<T, T, T>(output, cell_state_activation, 1, ConvertPolicy::SATURATE, RoundingPolicy::TO_NEAREST_EVEN, data_type);

        if(projection_opt)
        {
            SimpleTensor<T> fully_connected_projection = reference::fully_connected_layer(output_state_out, projection_w, projection_bias, output_cell_shape);
            if(projection_threshold != 0.f)
            {
                output_state_out = reference::activation_layer(fully_connected_projection, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, -projection_threshold, projection_threshold));
            }
        }
        std::vector<SimpleTensor<T>> scratch_inputs;
        if(!cifg_opt)
        {
            scratch_inputs.emplace_back(std::move(input_gate));
        }
        scratch_inputs.emplace_back(std::move(cell_state_out));
        scratch_inputs.emplace_back(std::move(forget_gate));
        scratch_inputs.emplace_back(std::move(output));
        scratch            = reference::concatenate_layer(scratch_inputs, scratch, Window::DimX);
        _reference_scratch = std::move(scratch);
        return output_state_out;
    }

    TensorType      _target{};
    TensorType      _target_scratch{};
    SimpleTensor<T> _reference{};
    SimpleTensor<T> _reference_scratch{};
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_LSTM_LAYER_FIXTURE */
