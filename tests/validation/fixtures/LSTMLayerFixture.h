/*
 * Copyright (c) 2018-2019 ARM Limited.
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
#include "tests/validation/reference/FullyConnectedLayer.h"
#include "tests/validation/reference/GEMM.h"
#include "tests/validation/reference/PixelWiseMultiplication.h"
#include "tests/validation/reference/Transpose.h"
#include "tests/validation/reference/WidthConcatenateLayer.h"

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
               TensorShape scratch_shape, ActivationLayerInfo info, float cell_threshold, float projection_threshold, DataType data_type, bool projection_opt, bool peephole_opt)
    {
        _target = compute_target(input_shape, input_weights_shape, recurrent_weights_shape, cell_bias_shape, output_cell_shape, output_shape, scratch_shape, info, cell_threshold, projection_threshold,
                                 data_type, projection_opt, peephole_opt);
        _reference = compute_reference(input_shape, input_weights_shape, recurrent_weights_shape, cell_bias_shape, output_cell_shape, output_shape, scratch_shape, info, cell_threshold, projection_threshold,
                                       data_type, projection_opt, peephole_opt);
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i)
    {
        std::uniform_real_distribution<> distribution(-1.0f, 1.0f);
        library->fill(tensor, distribution, i);
    }
    template <typename U>
    void fill_custom_val(U &&tensor, float num, int i)
    {
        std::uniform_real_distribution<> distribution(num, num);
        library->fill(tensor, distribution, i);
    }
    TensorType compute_target(const TensorShape &input_shape, const TensorShape &input_weights_shape, const TensorShape &recurrent_weights_shape, const TensorShape &cell_bias_shape,
                              const TensorShape &output_cell_shape, const TensorShape &output_shape, const TensorShape &scratch_shape, ActivationLayerInfo info, float cell_threshold,
                              float projection_threshold, DataType data_type, bool projection_opt, bool peephole_opt)
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

        // Create and configure function
        FunctionType lstm;
        lstm.configure(&input, &input_to_forget_w, &input_to_cell_w, &input_to_output_w, &recurrent_to_forget_w,
                       &recurrent_to_cell_w, &recurrent_to_output_w, &forget_gate_bias, &cell_bias, &output_gate_bias,
                       &output_state_in, &cell_state_in,
                       &scratch, &output_state_out, &cell_state_out, &output,
                       lstm_params, info, cell_threshold, projection_threshold);

        ARM_COMPUTE_EXPECT(input.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(input_to_forget_w.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(input_to_cell_w.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(input_to_output_w.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(recurrent_to_forget_w.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(recurrent_to_cell_w.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(recurrent_to_output_w.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(forget_gate_bias.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(cell_bias.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(output_gate_bias.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(output_state_in.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(cell_state_in.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(scratch.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(output_state_out.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(cell_state_out.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(output.info()->is_resizable(), framework::LogLevel::ERRORS);

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

        ARM_COMPUTE_EXPECT(!input.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!input_to_forget_w.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!input_to_cell_w.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!input_to_output_w.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!recurrent_to_forget_w.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!recurrent_to_cell_w.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!recurrent_to_output_w.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!forget_gate_bias.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!cell_bias.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!output_gate_bias.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!output_state_in.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!cell_state_in.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!scratch.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!output_state_out.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!cell_state_out.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!output.info()->is_resizable(), framework::LogLevel::ERRORS);

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
            ARM_COMPUTE_EXPECT(input_to_input_w.info()->is_resizable(), framework::LogLevel::ERRORS);
            ARM_COMPUTE_EXPECT(recurrent_to_input_w.info()->is_resizable(), framework::LogLevel::ERRORS);
            ARM_COMPUTE_EXPECT(cell_to_input_w.info()->is_resizable(), framework::LogLevel::ERRORS);
            ARM_COMPUTE_EXPECT(input_gate_bias.info()->is_resizable(), framework::LogLevel::ERRORS);
            input_to_input_w.allocator()->allocate();
            recurrent_to_input_w.allocator()->allocate();
            cell_to_input_w.allocator()->allocate();
            input_gate_bias.allocator()->allocate();
            ARM_COMPUTE_EXPECT(!input_to_input_w.info()->is_resizable(), framework::LogLevel::ERRORS);
            ARM_COMPUTE_EXPECT(!recurrent_to_input_w.info()->is_resizable(), framework::LogLevel::ERRORS);
            ARM_COMPUTE_EXPECT(!cell_to_input_w.info()->is_resizable(), framework::LogLevel::ERRORS);
            ARM_COMPUTE_EXPECT(!input_gate_bias.info()->is_resizable(), framework::LogLevel::ERRORS);
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
            ARM_COMPUTE_EXPECT(cell_to_forget_w.info()->is_resizable(), framework::LogLevel::ERRORS);
            ARM_COMPUTE_EXPECT(cell_to_output_w.info()->is_resizable(), framework::LogLevel::ERRORS);
            cell_to_forget_w.allocator()->allocate();
            cell_to_output_w.allocator()->allocate();
            ARM_COMPUTE_EXPECT(!cell_to_forget_w.info()->is_resizable(), framework::LogLevel::ERRORS);
            ARM_COMPUTE_EXPECT(!cell_to_output_w.info()->is_resizable(), framework::LogLevel::ERRORS);
            fill(AccessorType(cell_to_forget_w), 18);
            fill(AccessorType(cell_to_output_w), 19);
        }

        if(projection_opt)
        {
            ARM_COMPUTE_EXPECT(projection_w.info()->is_resizable(), framework::LogLevel::ERRORS);
            ARM_COMPUTE_EXPECT(projection_bias.info()->is_resizable(), framework::LogLevel::ERRORS);

            projection_w.allocator()->allocate();
            projection_bias.allocator()->allocate();

            ARM_COMPUTE_EXPECT(!projection_w.info()->is_resizable(), framework::LogLevel::ERRORS);
            ARM_COMPUTE_EXPECT(!projection_bias.info()->is_resizable(), framework::LogLevel::ERRORS);

            fill(AccessorType(projection_w), 20);
            fill(AccessorType(projection_bias), 21);
        }

        // Compute function
        lstm.run();

        _target_scratch = std::move(scratch);
        return output;
    }

    SimpleTensor<T> compute_reference(const TensorShape &input_shape, const TensorShape &input_weights_shape, const TensorShape &recurrent_weights_shape, const TensorShape &cell_bias_shape,
                                      const TensorShape &output_cell_shape, const TensorShape &output_shape, const TensorShape &scratch_shape, ActivationLayerInfo info, float cell_threshold,
                                      float projection_threshold, DataType data_type, bool projection_opt, bool peephole_opt)
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

        // Fill reference
        fill(input, 0);
        fill(input_to_forget_w, 1);
        fill(input_to_cell_w, 2);
        fill(input_to_output_w, 3);
        fill(recurrent_to_forget_w, 4);
        fill(recurrent_to_cell_w, 5);
        fill(recurrent_to_output_w, 6);
        fill(forget_gate_bias, 7);
        fill(cell_bias, 8);
        fill(output_gate_bias, 9);
        fill(output_state_in, 10);
        fill(cell_state_in, 11);
        fill(scratch, 12);
        fill(input_to_input_w, 13);
        fill(recurrent_to_input_w, 14);
        fill(cell_to_input_w, 15);
        fill(recurrent_to_input_w, 16);
        fill(input_gate_bias, 17);
        fill(cell_to_forget_w, 18);
        fill(cell_to_output_w, 19);
        fill(projection_w, 20);
        fill(projection_bias, 21);

        bool cifg_opt = scratch_shape.x() == cell_bias_shape.x() * 4 ? false : true;

        // Compute forget_gate
        SimpleTensor<T> fully_connected_forget = reference::fully_connected_layer(input, input_to_forget_w, forget_gate_bias, output_cell_shape);
        SimpleTensor<T> transposed_weights     = reference::transpose(recurrent_to_forget_w);
        SimpleTensor<T> gemm                   = reference::gemm(output_state_in, transposed_weights, cell_state_in, 1.f, 0.f);
        SimpleTensor<T> forget_gate            = reference::arithmetic_operation(reference::ArithmeticOperation::ADD, fully_connected_forget, gemm, data_type, ConvertPolicy::SATURATE);

        if(peephole_opt)
        {
            SimpleTensor<T> pixelwise_mul_forget_gate = reference::pixel_wise_multiplication(cell_state_in, cell_to_forget_w, 1, ConvertPolicy::SATURATE, RoundingPolicy::TO_ZERO);
            forget_gate                               = reference::arithmetic_operation(reference::ArithmeticOperation::ADD, forget_gate, pixelwise_mul_forget_gate, data_type, ConvertPolicy::SATURATE);
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
                SimpleTensor<T> pixelwise_mul_input_gate = reference::pixel_wise_multiplication(cell_state_in, cell_to_input_w, 1, ConvertPolicy::SATURATE, RoundingPolicy::TO_NEAREST_EVEN);
                input_gate                               = reference::arithmetic_operation(reference::ArithmeticOperation::ADD, input_gate, pixelwise_mul_input_gate, data_type, ConvertPolicy::SATURATE);
            }
            input_gate = reference::activation_layer(input_gate, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LOGISTIC));
        }

        // Compute cell_state
        SimpleTensor<T> fully_connected_cell_state = reference::fully_connected_layer(input, input_to_cell_w, cell_bias, output_cell_shape);
        transposed_weights                         = reference::transpose(recurrent_to_cell_w);
        gemm                                       = reference::gemm(output_state_in, transposed_weights, cell_state_out, 1.f, 0.f);
        SimpleTensor<T> pixelwise_mul              = reference::pixel_wise_multiplication(cell_state_in, forget_gate, 1, ConvertPolicy::SATURATE, RoundingPolicy::TO_NEAREST_EVEN);
        cell_state_out                             = reference::arithmetic_operation(reference::ArithmeticOperation::ADD, fully_connected_cell_state, gemm, data_type, ConvertPolicy::SATURATE);
        cell_state_out                             = reference::activation_layer(cell_state_out, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LOGISTIC));
        cell_state_out                             = reference::pixel_wise_multiplication(cell_state_out, input_gate, 1, ConvertPolicy::SATURATE, RoundingPolicy::TO_NEAREST_EVEN);
        cell_state_out                             = reference::arithmetic_operation(reference::ArithmeticOperation::ADD, cell_state_out, pixelwise_mul, data_type, ConvertPolicy::SATURATE);
        if(cell_threshold != 0.f)
        {
            cell_state_out = reference::activation_layer(cell_state_out, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, -cell_threshold, cell_threshold));
        }

        // Compute output
        SimpleTensor<T> fully_connected_output = reference::fully_connected_layer(input, input_to_output_w, output_gate_bias, output_cell_shape);
        transposed_weights                     = reference::transpose(recurrent_to_output_w);
        gemm                                   = reference::gemm(output_state_in, transposed_weights, cell_state_out, 1.f, 0.f);
        output                                 = reference::arithmetic_operation(reference::ArithmeticOperation::ADD, fully_connected_output, gemm, data_type, ConvertPolicy::SATURATE);
        if(peephole_opt)
        {
            pixelwise_mul = reference::pixel_wise_multiplication(cell_state_out, cell_to_output_w, 1, ConvertPolicy::SATURATE, RoundingPolicy::TO_NEAREST_EVEN);
            output        = reference::arithmetic_operation(reference::ArithmeticOperation::ADD, output, pixelwise_mul, data_type, ConvertPolicy::SATURATE);
        }
        output = reference::activation_layer(output, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LOGISTIC));

        // Compute output state
        SimpleTensor<T> cell_state_activation = reference::activation_layer(cell_state_out, info);
        output_state_out                      = reference::pixel_wise_multiplication(output, cell_state_activation, 1, ConvertPolicy::SATURATE, RoundingPolicy::TO_NEAREST_EVEN);

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
        scratch            = reference::widthconcatenate_layer(scratch_inputs, scratch);
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
