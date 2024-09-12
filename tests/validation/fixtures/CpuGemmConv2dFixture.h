/*
 * Copyright (c) 2024 Arm Limited.
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

#ifndef ACL_TESTS_VALIDATION_FIXTURES_CPUGEMMCONV2DFIXTURE_H
#define ACL_TESTS_VALIDATION_FIXTURES_CPUGEMMCONV2DFIXTURE_H

#include "arm_compute/core/experimental/Types.h"
#include "arm_compute/core/QuantizationInfo.h"
#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/function_info/ActivationLayerInfo.h"
#include "arm_compute/graph/Utils.h"

#include "src/core/helpers/MemoryHelpers.h"
#include "tests/AssetsLibrary.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/validation/Helpers.h"
#include "tests/validation/reference/ConvolutionLayer.h"
#include "tests/validation/reference/Utils.h"

namespace arm_compute
{
namespace test
{
namespace validation
{

template <typename TensorType, typename AccessorType, typename FunctionType>
class CpuGemmConv2dValidationFixture : public framework::Fixture
{
public:
    void setup(TensorShape   input_shape,
               TensorShape   weights_shape,
               TensorShape   bias_shape,
               TensorShape   output_shape,
               PadStrideInfo info,
               Size2D        dilation)
    {
        _dilation = dilation;
        _hash     = input_shape[0] + input_shape[1] + input_shape[2] + input_shape[3] + weights_shape[0] +
                weights_shape[1] + weights_shape[2] + weights_shape[3];
        _target    = compute_target(input_shape, weights_shape, bias_shape, output_shape, info);
        _reference = compute_reference(input_shape, weights_shape, bias_shape, output_shape, info);
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i)
    {
        std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
        library->fill(tensor, distribution, i);
    }

    TensorType compute_target(TensorShape          input_shape,
                              TensorShape          weights_shape,
                              const TensorShape   &bias_shape,
                              TensorShape          output_shape,
                              const PadStrideInfo &info)
    {
        // We need to permute to the same layout that the reference impl needs.
        permute(input_shape, PermutationVector(2U, 0U, 1U));
        permute(weights_shape, PermutationVector(2U, 0U, 1U));
        permute(output_shape, PermutationVector(2U, 0U, 1U));

        const auto src_info     = TensorInfo(input_shape, 1, DataType::F32, _data_layout);
        const auto weights_info = TensorInfo(weights_shape, 1, DataType::F32, _data_layout);
        const auto biases_info  = TensorInfo(bias_shape, 1, DataType::F32, _data_layout);
        auto       dst_info     = TensorInfo(output_shape, 1, DataType::F32, _data_layout);

        auto conv = std::make_unique<FunctionType>();
        conv->configure(&src_info, &weights_info, &biases_info, &dst_info, info);
        ARM_COMPUTE_ASSERT(conv->validate(&src_info, &weights_info, &biases_info, &dst_info, info));

        // Create tensors
        auto src     = create_tensor<Tensor>(src_info);
        auto weights = create_tensor<Tensor>(weights_info);
        auto biases  = create_tensor<Tensor>(biases_info);
        auto dst     = create_tensor<Tensor>(dst_info);

        // Allocate tensors
        src.allocator()->allocate();
        weights.allocator()->allocate();
        biases.allocator()->allocate();
        dst.allocator()->allocate();

        ITensorPack run_pack{{arm_compute::TensorType::ACL_SRC_0, &src},
                             {arm_compute::TensorType::ACL_SRC_1, &weights},
                             {arm_compute::TensorType::ACL_SRC_2, &biases},
                             {arm_compute::TensorType::ACL_DST, &dst}};
        ITensorPack prep_pack{{arm_compute::TensorType::ACL_SRC_1, &weights},
                              {arm_compute::TensorType::ACL_SRC_2, &biases}};

        auto const aux_mem_req = conv->workspace();
        auto       mg          = MemoryGroup{};
        auto       ws          = manage_workspace<Tensor>(aux_mem_req, mg, run_pack, prep_pack);

        // Fill tensors
        fill(AccessorType(src), 0 + _hash);
        fill(AccessorType(weights), 1 + _hash);
        fill(AccessorType(biases), 2 + _hash);

        conv->prepare(prep_pack);
        conv->run(run_pack);

        src.allocator()->free();
        weights.allocator()->free();
        biases.allocator()->free();

        return dst;
    }

    SimpleTensor<float> compute_reference(const TensorShape   &input_shape,
                                          const TensorShape   &weights_shape,
                                          const TensorShape   &bias_shape,
                                          const TensorShape   &output_shape,
                                          const PadStrideInfo &info)
    {
        // Create reference
        SimpleTensor<float> src{input_shape, DataType::F32};
        SimpleTensor<float> weights{weights_shape, DataType::F32};
        SimpleTensor<float> bias{bias_shape, DataType::F32};

        fill(src, 0 + _hash);
        fill(weights, 1 + _hash);
        fill(bias, 2 + _hash);

        return reference::convolution_layer<float>(src, weights, bias, output_shape, info, _dilation);
    }

    TensorType          _target{};
    SimpleTensor<float> _reference{};
    Size2D              _dilation{};
    int32_t             _hash{0};
    DataLayout          _data_layout{DataLayout::NHWC};
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T, typename TW>
class CpuGemmConv2dStaticQuantValidationFixture : public ConvolutionValidationGenericFixture<TensorType, AccessorType, FunctionType, T, T>
{
public:
    void setup(TensorShape input_shape, TensorShape weights_shape, TensorShape bias_shape, TensorShape output_shape, PadStrideInfo info, Size2D dilation, bool reshape_weights,
               DataType data_type, DataType weights_data_type, DataLayout data_layout, QuantizationInfo quantization_info, QuantizationInfo weight_quantization_info, ActivationLayerInfo act_info)
    {
        ARM_COMPUTE_ASSERT(data_type == DataType::QASYMM8_SIGNED || data_type == DataType::QASYMM8);

        // This hash is used by random generators. There may be hash collisions but
        // this is intentional as it's a very easy way to make the the current
        // random generation process almost different for many test configurations,
        // which were using the same set of values before.
        this->_hash = input_shape[0] + input_shape[1] + input_shape[2] + input_shape[3] +
            + weights_shape[0] + weights_shape[1] + weights_shape[2] + weights_shape[3] +
              (data_type == DataType::QASYMM8_SIGNED) + (data_layout == DataLayout::NHWC);

        this->_data_type                = data_type;
        this->_weights_data_type        = weights_data_type;
        this->_bias_data_type           = DataType::S32;
        this->_output_data_type         = data_type;
        this->_quantization_info        = quantization_info;
        this->_weight_quantization_info = weight_quantization_info;
        this->_data_layout              = data_layout;
        this->_dst_q_info               = quantization_info;

        if(!is_data_type_quantized_symmetric(weights_data_type) && (!act_info.enabled() || act_info.activation() == ActivationFunction::IDENTITY))
        {
            this->setup_quantization(input_shape, weights_shape, this->_quantization_info, this->_weight_quantization_info, data_type);
            this->_use_dynamic_output_quant = true;
        }

        this->_target = compute_target(input_shape, weights_shape, bias_shape, output_shape, info, reshape_weights, dilation, act_info);

        this->_reference = this->compute_reference(input_shape, weights_shape, bias_shape, output_shape, info, dilation, act_info);
    }

protected:

    // Compute the target when updating static quantization information after configuration for the stateless api.
    TensorType compute_target(TensorShape input_shape, TensorShape weights_shape, const TensorShape &bias_shape, TensorShape output_shape, const PadStrideInfo &info,
                              bool reshape_weights, const Size2D &dilation, const ActivationLayerInfo act_info, PaddingList pre_pad_layer = PaddingList({}), bool padded_weights = false)
    {
        ARM_COMPUTE_ASSERT((std::is_same<FunctionType, experimental::op::CpuGemmConv2d>::value == true));

        ARM_COMPUTE_ERROR_ON((input_shape[2] % weights_shape[2]) != 0);

        const unsigned int num_groups = input_shape[2] / weights_shape[2];

        if(this->_data_layout == DataLayout::NHWC)
        {
            permute(input_shape, PermutationVector(2U, 0U, 1U));
            permute(weights_shape, PermutationVector(2U, 0U, 1U));
            permute(output_shape, PermutationVector(2U, 0U, 1U));

            if(pre_pad_layer.size() > 0)
            {
                // make sure paddings exist for each c,h,w dimensions
                for(unsigned int i = 0; i < 3 - pre_pad_layer.size(); ++i)
                {
                    pre_pad_layer.push_back({ 0, 0 });
                }

                // rotate padding info from nchw to nhwc
                std::rotate(pre_pad_layer.begin(), pre_pad_layer.begin() + 2, pre_pad_layer.begin() + 3);
            }
        }

        const int idx_width  = get_data_layout_dimension_index(this->_data_layout, DataLayoutDimension::WIDTH);
        const int idx_height = get_data_layout_dimension_index(this->_data_layout, DataLayoutDimension::HEIGHT);

        WeightsInfo weights_info(!reshape_weights, weights_shape[idx_width], weights_shape[idx_height], weights_shape[3]);
        TensorShape reshaped_weights_shape(weights_shape);

        // Create tensors with fake quantization info and defer to pass the correct ones to a later stage.
        auto qi = QuantizationInfo(0.550721, 37, true);
        TensorType src     = create_tensor<TensorType>(input_shape, this->_data_type, 1, qi, this->_data_layout);
        TensorType weights = create_tensor<TensorType>(reshaped_weights_shape, this->_weights_data_type, 1, qi, this->_data_layout);
        TensorType dst     = create_tensor<TensorType>(output_shape, this->_output_data_type, 1, qi, this->_data_layout);
        TensorType bias    = create_tensor<TensorType>(bias_shape, this->_bias_data_type, 1, QuantizationInfo() /*bias is not a quantized type*/, this->_data_layout);

        // Create and configure function
        FunctionType conv;

        const unsigned int height_index = arm_compute::graph::get_dimension_idx(this->_data_layout, DataLayoutDimension::HEIGHT);
        const unsigned int width_index  = arm_compute::graph::get_dimension_idx(this->_data_layout, DataLayoutDimension::WIDTH);

        const PaddingInfo pad_w = width_index < pre_pad_layer.size() ? pre_pad_layer[width_index] : PaddingInfo(0, 0);
        const PaddingInfo pad_h = height_index < pre_pad_layer.size() ? pre_pad_layer[height_index] : PaddingInfo(0, 0);

        if(pre_pad_layer.size() > 0 && arm_compute::graph::is_padding_in_height_or_width(this->_data_layout, pre_pad_layer))
        {
            // this is the logic implemented in NodeFusionMutator -> fuse_pad_with_convolution
            const PadStrideInfo new_conv_info(
                info.stride().first,
                info.stride().second,
                info.pad_left() + pad_w.first,
                info.pad_right() + pad_w.second,
                info.pad_top() + pad_h.first,
                info.pad_bottom() + pad_h.second,
                info.round());

            conv.configure(src.info(), weights.info(), bias.info(), dst.info(), new_conv_info, weights_info, dilation, act_info, false /* enable_fast_math */, num_groups);
            auto const status = conv.validate(src.info(), weights.info(), bias.info(), dst.info(), new_conv_info);
            ARM_COMPUTE_ASSERT(status);
        }
        else
        {
            conv.configure(src.info(), weights.info(), bias.info(), dst.info(), info, weights_info, dilation, act_info, false /* enable_fast_math */, num_groups);
            auto const status = conv.validate(src.info(), weights.info(), bias.info(), dst.info(), info);
            ARM_COMPUTE_ASSERT(status);
        }

        // After calling configure, we appropriately set the correct quantization info and update ACL.
        src.info()->set_quantization_info(QuantizationInfo(this->_quantization_info.scale(), this->_quantization_info.offset(), true));
        weights.info()->set_quantization_info(QuantizationInfo(this->_weight_quantization_info.scale(), this->_weight_quantization_info.offset(), true));
        dst.info()->set_quantization_info(QuantizationInfo(this->_dst_q_info.scale(), this->_dst_q_info.offset(), true));

        ARM_COMPUTE_ASSERT(src.info()->is_resizable());
        ARM_COMPUTE_ASSERT(weights.info()->is_resizable());
        ARM_COMPUTE_ASSERT(bias.info()->is_resizable());
        ARM_COMPUTE_ASSERT(dst.info()->is_resizable());

        // Test "add padding after configure" behavior. This behavior should not affect the correctness
        add_padding_x({ &src, &bias, &dst }, this->_data_layout);
        // Padding weights may affect code path in some backends
        if (padded_weights)
        {
            add_padding_x({ &weights }, this->_data_layout);
        }

        // // Allocate tensors
        src.allocator()->allocate();
        weights.allocator()->allocate();
        bias.allocator()->allocate();
        dst.allocator()->allocate();

        ITensorPack run_pack{
            {ACL_SRC_0, &src}, {ACL_SRC_1, &weights}, {ACL_SRC_2, &bias}, {ACL_DST, &dst}};
        ITensorPack prep_pack{{ACL_SRC_1, &weights}, {ACL_SRC_2, &bias}};

        // propagate trough ACL the correct quantization info
        conv.update_quantization_parameters(run_pack);

        auto mg = MemoryGroup{};
        auto ws = manage_workspace<Tensor>(conv.workspace(), mg, run_pack, prep_pack);

        ARM_COMPUTE_ASSERT(!src.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!weights.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!bias.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!dst.info()->is_resizable());

        // Fill tensors
        this->fill(AccessorType(src), 0 + this->_hash);
        this->fill(AccessorType(weights), 1 + this->_hash);
        this->fill(AccessorType(bias), 2 + this->_hash);

        // Compute Convolution function
        conv.prepare(prep_pack);
        conv.run(run_pack);

        return dst;
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class CpuGemmConv2dForUpdatedStaticQuantInfoAfterConfigureFixture : public CpuGemmConv2dStaticQuantValidationFixture<TensorType, AccessorType, FunctionType, T, T>
{
public:
    void setup(TensorShape input_shape, TensorShape weights_shape, TensorShape bias_shape, TensorShape output_shape, PadStrideInfo info, Size2D dilation, bool reshape_weights, DataType data_type,
               DataLayout data_layout, QuantizationInfo quantization_info, ActivationLayerInfo act_info)
    {
        CpuGemmConv2dStaticQuantValidationFixture<TensorType, AccessorType, FunctionType, T, T>::setup(input_shape, weights_shape, bias_shape, output_shape, info, dilation, reshape_weights,
                                                                                                 data_type, data_type, data_layout, quantization_info, quantization_info, act_info);
    }
};

} // namespace validation
} // namespace test
} // namespace arm_compute

#endif // ACL_TESTS_VALIDATION_FIXTURES_CPUGEMMCONV2DFIXTURE_H
