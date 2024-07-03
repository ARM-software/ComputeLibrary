/*
 * Copyright (c) 2018-2021, 2023-2024 Arm Limited.
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

#ifndef ACL_TESTS_VALIDATION_FIXTURES_CPUWINOGRADCONV2DFIXTURE_H
#define ACL_TESTS_VALIDATION_FIXTURES_CPUWINOGRADCONV2DFIXTURE_H

#include "tests/validation/fixtures/WinogradConvolutionLayerFixture.h"

#include <memory>

namespace arm_compute
{
namespace test
{
namespace validation
{

template <typename TensorType, typename AccessorType, typename FunctionType>
class CpuWinogradConv2dValidationFixture : public framework::Fixture
{
public:
    void setup(TensorShape         input_shape,
               TensorShape         weights_shape,
               TensorShape         bias_shape,
               TensorShape         output_shape,
               PadStrideInfo       info,
               Size2D              dilation,
               ActivationLayerInfo act_info)
    {
        ARM_COMPUTE_UNUSED(dilation);
        _act_info = act_info;

        _target    = compute_target(input_shape, weights_shape, bias_shape, output_shape, info);
        _reference = compute_reference(input_shape, weights_shape, bias_shape, info);
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i, float min, float max)
    {
        std::uniform_real_distribution<float> distribution(min, max);
        library->fill(tensor, distribution, i);
    }

    TensorType compute_target(TensorShape          input_shape,
                              TensorShape          weights_shape,
                              TensorShape          bias_shape,
                              TensorShape          output_shape,
                              const PadStrideInfo &info)
    {
        permute(input_shape, PermutationVector(2U, 0U, 1U));
        permute(weights_shape, PermutationVector(2U, 0U, 1U));
        permute(output_shape, PermutationVector(2U, 0U, 1U));

        // Create tensors
        TensorType src     = create_tensor<TensorType>(input_shape, _data_type, 1, QuantizationInfo(), _data_layout);
        TensorType weights = create_tensor<TensorType>(weights_shape, _data_type, 1, QuantizationInfo(), _data_layout);
        TensorType bias    = create_tensor<TensorType>(bias_shape, _data_type, 1, QuantizationInfo(), _data_layout);
        TensorType dst     = create_tensor<TensorType>(output_shape, _data_type, 1, QuantizationInfo(), _data_layout);

        // Create and configure function
        auto conv = std::make_unique<FunctionType>();
        ARM_COMPUTE_EXPECT(static_cast<bool>(conv->validate(src.info(), weights.info(), bias.info(), dst.info(), info,
                                                            _act_info, true)),
                           framework::LogLevel::ERRORS);
        conv->configure(src.info(), weights.info(), bias.info(), dst.info(), info, _act_info, true);

        ARM_COMPUTE_ASSERT(src.info()->is_resizable());
        ARM_COMPUTE_ASSERT(weights.info()->is_resizable());
        ARM_COMPUTE_ASSERT(bias.info()->is_resizable());
        ARM_COMPUTE_ASSERT(dst.info()->is_resizable());

        add_padding_x({&src, &weights, &bias, &dst}, _data_layout);

        // Allocate tensors
        src.allocator()->allocate();
        weights.allocator()->allocate();
        dst.allocator()->allocate();
        bias.allocator()->allocate();

        ARM_COMPUTE_ASSERT(!src.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!weights.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!bias.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!dst.info()->is_resizable());

        // Fill tensors
        fill(AccessorType(src), 0, -0.5f, 0.5f);
        fill(AccessorType(weights), 1, -0.5f, 0.5f);
        fill(AccessorType(bias), 2, -0.5f, 0.5f);

        // Compute function
        ITensorPack run_pack  = {{ACL_SRC_0, &src}, {ACL_SRC_1, &weights}, {ACL_SRC_2, &bias}, {ACL_DST, &dst}};
        ITensorPack prep_pack = {{ACL_SRC_1, &weights}, {ACL_SRC_2, &bias}};

        auto const aux_mem_req = conv->workspace();
        auto       mg          = MemoryGroup{};
        auto       ws          = manage_workspace<Tensor>(aux_mem_req, mg, run_pack, prep_pack);

        conv->prepare(prep_pack);
        conv->run(run_pack);

        src.allocator()->free();
        weights.allocator()->free();
        bias.allocator()->free();

        return dst;
    }

    SimpleTensor<float> compute_reference(const TensorShape   &input_shape,
                                          const TensorShape   &weights_shape,
                                          const TensorShape   &bias_shape,
                                          const PadStrideInfo &info)
    {
        // Create reference
        SimpleTensor<float> src_t{input_shape, _data_type, 1};
        SimpleTensor<float> weights_t{weights_shape, _data_type, 1};
        SimpleTensor<float> bias_t{bias_shape, _data_type, 1};

        // Fill reference
        fill(src_t, 0, -0.5f, 0.5f);
        SimpleTensor<float> src_t1(copy_tensor<float, float>(src_t));

        fill(weights_t, 1, -0.5f, 0.5f);
        SimpleTensor<float> weights_t1(copy_tensor<float, float>(weights_t));
        fill(bias_t, 2, -0.5f, 0.5f);
        SimpleTensor<float> bias_t1(copy_tensor<float, float>(bias_t));

        // Set output tile
        Size2D output_tile(4U, 4U);
        if (weights_shape[0] == 7 && weights_shape[1] == 1)
        {
            output_tile.width  = 2;
            output_tile.height = 1;
        }
        else if (weights_shape[0] == 1 && weights_shape[1] == 7)
        {
            output_tile.width  = 1;
            output_tile.height = 2;
        }
        else if (weights_shape[0] == 1)
        {
            output_tile.width = 1;
        }
        else if (weights_shape[1] == 1)
        {
            output_tile.height = 1;
        }

        WinogradInfo winograd_info(output_tile, Size2D(weights_shape[0], weights_shape[1]),
                                   Size2D(input_shape[0], input_shape[1]), info, src_t1.data_layout());

        // Compute tensor shapes for input, filter and output transforms
        TensorShape input_transform_shape =
            compute_winograd_input_transform_shape(TensorInfo(input_shape, 1, _data_type), winograd_info);
        TensorShape filter_transform_shape =
            compute_winograd_filter_transform_shape(TensorInfo(weights_shape, 1, _data_type), winograd_info);
        TensorShape batched_gemm_shape = input_transform_shape;
        batched_gemm_shape[0]          = filter_transform_shape[0];
        TensorShape output_transform_shape =
            compute_winograd_output_transform_shape(TensorInfo(batched_gemm_shape, 1, _data_type), winograd_info);

        // Dummy matrix C to perform matrix multiplication
        SimpleTensor<float> dummy_c{batched_gemm_shape, _data_type, 1};

        // Compute Winograd-based convolution
        SimpleTensor<float> input_transform_out =
            reference::winograd_input_transform<float>(src_t1, input_transform_shape, winograd_info);

        SimpleTensor<float> filter_transform_out =
            reference::winograd_filter_transform<float>(weights_t1, filter_transform_shape, winograd_info);
        SimpleTensor<float> batched_gemm =
            reference::gemm<float>(input_transform_out, filter_transform_out, dummy_c, 1.0f, 0.0f);
        SimpleTensor<float> conv_out =
            reference::winograd_output_transform<float>(batched_gemm, bias_t1, output_transform_shape, winograd_info);
        SimpleTensor<float> conv_out_t(copy_tensor<float, float>(conv_out));
        return (_act_info.enabled()) ? reference::activation_layer<float>(conv_out_t, _act_info) : conv_out_t;
    }

    TensorType          _target{};
    SimpleTensor<float> _reference{};
    ActivationLayerInfo _act_info{};
    DataType            _data_type{DataType::F32};
    DataLayout          _data_layout{DataLayout::NHWC};
};

} // namespace validation
} // namespace test
} // namespace arm_compute

#endif // ACL_TESTS_VALIDATION_FIXTURES_CPUWINOGRADCONV2DFIXTURE_H
