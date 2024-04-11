/*
 * Copyright (c) 2018-2021, 2023 Arm Limited.
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
#ifndef ARM_COMPUTE_TEST_RNN_LAYER_FIXTURE
#define ARM_COMPUTE_TEST_RNN_LAYER_FIXTURE

#include "tests/Globals.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/reference/ActivationLayer.h"
#include "tests/validation/reference/ArithmeticOperations.h"
#include "tests/validation/reference/FullyConnectedLayer.h"
#include "tests/validation/reference/GEMM.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class RNNLayerValidationFixture : public framework::Fixture
{
public:
    void setup(TensorShape input_shape, TensorShape weights_shape, TensorShape recurrent_weights_shape, TensorShape bias_shape, TensorShape output_shape, ActivationLayerInfo info,
               DataType data_type)
    {
        _target    = compute_target(input_shape, weights_shape, recurrent_weights_shape, bias_shape, output_shape, info, data_type);
        _reference = compute_reference(input_shape, weights_shape, recurrent_weights_shape, bias_shape, output_shape, info, data_type);
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

    TensorType compute_target(const TensorShape &input_shape, const TensorShape &weights_shape, const TensorShape &recurrent_weights_shape, const TensorShape &bias_shape, const TensorShape &output_shape,
                              ActivationLayerInfo info, DataType data_type)
    {
        // Create tensors
        TensorType input             = create_tensor<TensorType>(input_shape, data_type);
        TensorType weights           = create_tensor<TensorType>(weights_shape, data_type);
        TensorType recurrent_weights = create_tensor<TensorType>(recurrent_weights_shape, data_type);
        TensorType bias              = create_tensor<TensorType>(bias_shape, data_type);
        TensorType hidden_state      = create_tensor<TensorType>(output_shape, data_type);
        TensorType output            = create_tensor<TensorType>(output_shape, data_type);

        // Create and configure function
        FunctionType rnn;
        rnn.configure(&input, &weights, &recurrent_weights, &bias, &hidden_state, &output, info);

        ARM_COMPUTE_ASSERT(input.info()->is_resizable());
        ARM_COMPUTE_ASSERT(weights.info()->is_resizable());
        ARM_COMPUTE_ASSERT(recurrent_weights.info()->is_resizable());
        ARM_COMPUTE_ASSERT(bias.info()->is_resizable());
        ARM_COMPUTE_ASSERT(hidden_state.info()->is_resizable());
        ARM_COMPUTE_ASSERT(output.info()->is_resizable());

        // Allocate tensors
        input.allocator()->allocate();
        weights.allocator()->allocate();
        recurrent_weights.allocator()->allocate();
        bias.allocator()->allocate();
        hidden_state.allocator()->allocate();
        output.allocator()->allocate();

        ARM_COMPUTE_ASSERT(!input.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!weights.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!recurrent_weights.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!bias.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!hidden_state.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!output.info()->is_resizable());

        // Fill tensors
        fill(AccessorType(input), 0);
        fill(AccessorType(weights), 0);
        fill(AccessorType(recurrent_weights), 0);
        fill(AccessorType(bias), 0);
        fill(AccessorType(hidden_state), 0);

        // Compute function
        rnn.run();

        return output;
    }

    SimpleTensor<T> compute_reference(const TensorShape &input_shape, const TensorShape &weights_shape, const TensorShape &recurrent_weights_shape, const TensorShape &bias_shape,
                                      const TensorShape &output_shape, ActivationLayerInfo info, DataType data_type)
    {
        // Create reference
        SimpleTensor<T> input{ input_shape, data_type };
        SimpleTensor<T> weights{ weights_shape, data_type };
        SimpleTensor<T> recurrent_weights{ recurrent_weights_shape, data_type };
        SimpleTensor<T> bias{ bias_shape, data_type };
        SimpleTensor<T> hidden_state{ output_shape, data_type };

        // Fill reference
        fill(input, 0);
        fill(weights, 0);
        fill(recurrent_weights, 0);
        fill(bias, 0);
        fill(hidden_state, 0);

        TensorShape out_shape = recurrent_weights_shape;
        out_shape.set(1, output_shape.y());

        // Compute reference
        SimpleTensor<T> out_w{ out_shape, data_type };
        SimpleTensor<T> fully_connected = reference::fully_connected_layer(input, weights, bias, out_shape);
        SimpleTensor<T> gemm            = reference::gemm(hidden_state, recurrent_weights, out_w, 1.f, 0.f);
        SimpleTensor<T> add_res         = reference::arithmetic_operation(reference::ArithmeticOperation::ADD, fully_connected, gemm, data_type, ConvertPolicy::SATURATE);
        return reference::activation_layer(add_res, info);
    }

    TensorType      _target{};
    SimpleTensor<T> _reference{};
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_RNN_LAYER_FIXTURE */
