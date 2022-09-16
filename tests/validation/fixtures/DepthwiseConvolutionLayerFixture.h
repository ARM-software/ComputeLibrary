/*
 * Copyright (c) 2017-2022 Arm Limited.
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
#ifndef ARM_COMPUTE_TEST_DEPTHWISE_CONVOLUTION_FIXTURE
#define ARM_COMPUTE_TEST_DEPTHWISE_CONVOLUTION_FIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/Helpers.h"
#include "tests/validation/reference/ActivationLayer.h"
#include "tests/validation/reference/DepthwiseConvolutionLayer.h"

#include "utils/Utils.h"

#include <random>

namespace arm_compute
{
namespace test
{
namespace validation
{
using namespace arm_compute::misc::shape_calculator;

template <typename TensorType, typename AccessorType, typename FunctionType, typename T, typename TW>
class DepthwiseConvolutionLayerValidationGenericFixture : public framework::Fixture
{
public:
    using TBias = typename std::conditional < std::is_same<T, uint8_t>::value || std::is_same<T, int8_t>::value, int32_t, T >::type;

public:
    template <typename...>
    void setup(TensorShape in_shape, Size2D kernel_size, PadStrideInfo pad_stride_info, Size2D dilation,
               unsigned int depth_multiplier, DataType input_data_type, DataType weights_data_type,
               QuantizationInfo input_quantization_info, QuantizationInfo weights_quantization_info, QuantizationInfo output_quantization_info,
               DataLayout data_layout, ActivationLayerInfo act_info, bool mixed_layout = false, bool in_place = false)
    {
        ARM_COMPUTE_ERROR_ON(mixed_layout && in_place);
        _mixed_layout              = mixed_layout;
        _input_shape               = in_shape;
        _input_data_type           = input_data_type;
        _weights_data_type         = weights_data_type;
        _input_quantization_info   = input_quantization_info;
        _weights_quantization_info = weights_quantization_info;
        _output_quantization_info  = output_quantization_info;
        _data_layout               = data_layout;
        _pad_stride_info           = pad_stride_info;
        _act_info                  = act_info;
        _depth_multiplier          = depth_multiplier;
        _dilation                  = dilation;
        _in_place                  = in_place;

        _bias_data_type = is_data_type_quantized(_input_data_type) ? DataType::S32 : _input_data_type;

        _weights_shape = TensorShape(kernel_size.width, kernel_size.height);

        const TensorInfo      in_info(_input_shape, 1, _input_data_type);
        const TensorInfo      we_info(_weights_shape, 1, _weights_data_type);
        const ConvolutionInfo info{ _pad_stride_info, _depth_multiplier, _act_info, _dilation };
        _output_shape = compute_depthwise_convolution_shape(in_info, we_info, info);

        _weights_shape.set(2, _output_shape.z());
        _biases_shape = TensorShape(_weights_shape[2]);
    }

    void configure_target()
    {
        TensorShape input_shape   = _input_shape;
        TensorShape weights_shape = _weights_shape;
        TensorShape output_shape  = _output_shape;

        if(_data_layout == DataLayout::NHWC)
        {
            permute(input_shape, PermutationVector(2U, 0U, 1U));
            permute(weights_shape, PermutationVector(2U, 0U, 1U));
            permute(output_shape, PermutationVector(2U, 0U, 1U));
        }

        // Create tensors
        _src                      = create_tensor<TensorType>(input_shape, _input_data_type, 1, _input_quantization_info, _data_layout);
        _weights                  = create_tensor<TensorType>(weights_shape, _weights_data_type, 1, _weights_quantization_info, _data_layout);
        _biases                   = create_tensor<TensorType>(_biases_shape, _bias_data_type, 1, _input_quantization_info, _data_layout);
        TensorType *target_to_use = nullptr;
        if(!_in_place)
        {
            _target       = create_tensor<TensorType>(output_shape, _input_data_type, 1, _output_quantization_info, _data_layout);
            target_to_use = &_target;
        }

        add_padding_x({ &_src, &_biases }, _data_layout);
        add_padding_x({ &_weights }, _data_layout, true);
        if(!_in_place)
        {
            add_padding_x({ &_target }, _data_layout);
        }

        // Create Depthwise Convolution configure function
        _dwc.configure(&_src, &_weights, &_biases, target_to_use, _pad_stride_info, _depth_multiplier, _act_info, _dilation);

        ARM_COMPUTE_ASSERT(_src.info()->is_resizable());
        ARM_COMPUTE_ASSERT(_weights.info()->is_resizable());
        ARM_COMPUTE_ASSERT(_biases.info()->is_resizable());
        ARM_COMPUTE_ASSERT(_target.info()->is_resizable());
    }

    void allocate_and_run_target()
    {
        // Allocate tensors
        _src.allocator()->allocate();
        _weights.allocator()->allocate();
        _biases.allocator()->allocate();

        ARM_COMPUTE_ASSERT(!_src.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!_weights.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!_biases.info()->is_resizable());

        if(!_in_place)
        {
            _target.allocator()->allocate();
            ARM_COMPUTE_ASSERT(!_target.info()->is_resizable());
        }

        // Fill tensors
        fill(AccessorType(_src), 0);
        fill(AccessorType(_weights), 1);
        fill(AccessorType(_biases), 2);

        if(_mixed_layout)
        {
            mix_layout(_dwc, _src, _target);
        }
        else
        {
            // Compute function
            _dwc.run();
        }
    }

    void compute_reference()
    {
        SimpleTensor<T>     src{ _input_shape, _input_data_type, 1, _input_quantization_info };
        SimpleTensor<TW>    weights{ _weights_shape, _weights_data_type, 1, _weights_quantization_info };
        SimpleTensor<TBias> biases{ _biases_shape, _bias_data_type, 1, _input_quantization_info };

        fill(src, 0);
        fill(weights, 1);
        fill(biases, 2);

        SimpleTensor<T> depth_out = reference::depthwise_convolution(src, weights, biases, _output_shape, _pad_stride_info, _depth_multiplier, _dilation, _output_quantization_info);
        _reference                = (_act_info.enabled()) ? reference::activation_layer<T>(depth_out, _act_info) : depth_out;
    }

protected:
    void mix_layout(FunctionType &layer, TensorType &src, TensorType &dst)
    {
        ARM_COMPUTE_ERROR_ON(_in_place);
        // Test Multi DataLayout graph cases, when the data layout changes after configure
        src.info()->set_data_layout(_data_layout == DataLayout::NCHW ? DataLayout::NHWC : DataLayout::NCHW);
        dst.info()->set_data_layout(_data_layout == DataLayout::NCHW ? DataLayout::NHWC : DataLayout::NCHW);

        // Compute Convolution function
        layer.run();

        // Reinstating original data layout for the test suite to properly check the values
        src.info()->set_data_layout(_data_layout);
        dst.info()->set_data_layout(_data_layout);
    }

    template <typename U>
    void fill(U &&tensor, int i)
    {
        switch(tensor.data_type())
        {
            case DataType::QASYMM8:
            {
                std::uniform_int_distribution<uint32_t> distribution(0, 15);
                library->fill(tensor, distribution, i);
                break;
            }
            case DataType::QASYMM8_SIGNED:
            case DataType::QSYMM8_PER_CHANNEL:
            {
                std::uniform_int_distribution<int32_t> distribution(-10, 10);
                library->fill(tensor, distribution, i);
                break;
            }
            case DataType::F16:
            {
                arm_compute::utils::uniform_real_distribution_16bit<half> distribution{ -1.0f, 1.0f };
                library->fill(tensor, distribution, i);
                break;
            }
            case DataType::F32:
            {
                std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
                library->fill(tensor, distribution, i);
                break;
            }
            case DataType::S32:
            {
                std::uniform_int_distribution<int32_t> distribution(-100, 100);
                library->fill(tensor, distribution, i);
                break;
            }
            default:
                library->fill_tensor_uniform(tensor, i);
        }
    }

    TensorType      _target{};
    SimpleTensor<T> _reference{};

    TensorType   _src{};
    TensorType   _weights{};
    TensorType   _biases{};
    FunctionType _dwc{};

    TensorShape         _input_shape{};
    TensorShape         _weights_shape{};
    TensorShape         _biases_shape{};
    TensorShape         _output_shape{};
    DataType            _input_data_type{};
    DataType            _weights_data_type{};
    DataType            _bias_data_type{};
    QuantizationInfo    _input_quantization_info{};
    QuantizationInfo    _weights_quantization_info{};
    QuantizationInfo    _output_quantization_info{};
    DataLayout          _data_layout{};
    PadStrideInfo       _pad_stride_info{};
    ActivationLayerInfo _act_info{};
    unsigned int        _depth_multiplier{};
    Size2D              _dilation{};
    bool                _mixed_layout{ false };
    bool                _in_place{ false };
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T, bool mixed_layout = false, bool in_place = false>
class DepthwiseConvolutionLayerValidationFixture : public DepthwiseConvolutionLayerValidationGenericFixture<TensorType, AccessorType, FunctionType, T, T>
{
public:
    template <typename...>
    void setup(TensorShape in_shape, Size2D kernel_size, PadStrideInfo pad_stride_info, Size2D dilation, unsigned int depth_multiplier, DataType data_type, DataLayout data_layout,
               ActivationLayerInfo act_info)
    {
        DepthwiseConvolutionLayerValidationGenericFixture<TensorType, AccessorType, FunctionType, T, T>::setup(in_shape, kernel_size, pad_stride_info, dilation, depth_multiplier,
                                                                                                               data_type, data_type, QuantizationInfo(), QuantizationInfo(), QuantizationInfo(),
                                                                                                               data_layout, act_info, mixed_layout, in_place);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class DepthwiseConvolutionLayerNativeValidationFixture : public DepthwiseConvolutionLayerValidationGenericFixture<TensorType, AccessorType, FunctionType, T, T>
{
public:
    template <typename...>
    void setup(size_t width, size_t height, size_t channel, size_t batch, Size2D kernel_size, size_t depth_multiplier, Size2D dilation, Size2D stride, bool padding_valid, DataType data_type,
               DataLayout data_layout)
    {
        _dilation         = dilation;
        _depth_multiplier = depth_multiplier;
        _data_type        = data_type;
        _data_layout      = data_layout;

        _input_shape   = TensorShape(width, height, channel, batch);
        _weights_shape = TensorShape(kernel_size.width, kernel_size.height, channel * _depth_multiplier);
        _biases_shape  = TensorShape(_weights_shape.z());

        if(padding_valid)
        {
            _conv_info = PadStrideInfo(stride.width, stride.height);
        }
        else
        {
            _conv_info = calculate_same_pad(_input_shape, _weights_shape, PadStrideInfo(stride.width, stride.height), DataLayout::NCHW, _dilation);
        }
    }

    void configure_target()
    {
        TensorShape input_shape   = _input_shape;
        TensorShape weights_shape = _weights_shape;

        if(_data_layout == DataLayout::NHWC)
        {
            permute(input_shape, PermutationVector(2U, 0U, 1U));
            permute(weights_shape, PermutationVector(2U, 0U, 1U));
        }

        // Create tensors
        _src     = create_tensor<TensorType>(input_shape, _data_type, 1, QuantizationInfo(), _data_layout);
        _weights = create_tensor<TensorType>(weights_shape, _data_type, 1, QuantizationInfo(), _data_layout);
        _biases  = create_tensor<TensorType>(_biases_shape, _data_type, 1, QuantizationInfo(), _data_layout);
        _target  = create_tensor<TensorType>(TensorShape(), _data_type, 1, QuantizationInfo(), _data_layout);

        add_padding_x({ &_src, &_biases, &_target }, _data_layout);
        add_padding_x({ &_weights }, _data_layout, true);
        add_padding_y({ &_src, &_target }, _data_layout);

        // Create Depthwise Convolution configure function
        const ConvolutionInfo info
        {
            _conv_info, _depth_multiplier, ActivationLayerInfo(), _dilation
        };
        _dwc.configure(_src.info(), _weights.info(), _biases.info(), _target.info(), info);

        ARM_COMPUTE_ASSERT(_src.info()->is_resizable());
        ARM_COMPUTE_ASSERT(_weights.info()->is_resizable());
        ARM_COMPUTE_ASSERT(_biases.info()->is_resizable());
        ARM_COMPUTE_ASSERT(_target.info()->is_resizable());
    }

    void allocate_and_run_target()
    {
        // Allocate tensors
        _src.allocator()->allocate();
        _weights.allocator()->allocate();
        _biases.allocator()->allocate();
        _target.allocator()->allocate();

        ARM_COMPUTE_ASSERT(!_src.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!_weights.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!_biases.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!_target.info()->is_resizable());

        // Fill tensors
        fill(AccessorType(_src), 0);
        fill(AccessorType(_weights), 1);
        fill(AccessorType(_biases), 2);

        arm_compute::ITensorPack pack;
        pack.add_const_tensor(arm_compute::TensorType::ACL_SRC_0, &_src);
        pack.add_const_tensor(arm_compute::TensorType::ACL_SRC_1, &_weights);
        pack.add_const_tensor(arm_compute::TensorType::ACL_SRC_2, &_biases);
        pack.add_tensor(arm_compute::TensorType::ACL_DST, &_target);

        // Compute function
        _dwc.run(pack);
    }

    void compute_reference()
    {
        SimpleTensor<T> src{ _input_shape, _data_type };
        SimpleTensor<T> weights{ _weights_shape, _data_type };
        SimpleTensor<T> biases{ _biases_shape, _data_type };

        fill(src, 0);
        fill(weights, 1);
        fill(biases, 2);

        const ConvolutionInfo info{ _conv_info, _depth_multiplier, ActivationLayerInfo(), _dilation };
        const TensorShape     dst_shape = compute_depthwise_convolution_shape(TensorInfo(_input_shape, 1, _data_type), TensorInfo(_weights_shape, 1, _data_type), info);
        _reference                      = reference::depthwise_convolution(src, weights, biases, dst_shape, _conv_info, _depth_multiplier, _dilation);
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i)
    {
        switch(tensor.data_type())
        {
            case DataType::F32:
            {
                std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
                library->fill(tensor, distribution, i);
                break;
            }
            default:
                library->fill_tensor_uniform(tensor, i);
        }
    }

    TensorType      _target{};
    SimpleTensor<T> _reference{};

    TensorType   _src{};
    TensorType   _weights{};
    TensorType   _biases{};
    FunctionType _dwc{};

    TensorShape   _input_shape{};
    TensorShape   _weights_shape{};
    TensorShape   _biases_shape{};
    DataType      _data_type{};
    DataLayout    _data_layout{};
    PadStrideInfo _conv_info{};
    Size2D        _dilation{};
    unsigned int  _depth_multiplier{};
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T, bool in_place = false>
class DepthwiseConvolutionLayerNativeConfigurableValidationFixture : public DepthwiseConvolutionLayerValidationGenericFixture<TensorType, AccessorType, FunctionType, T, T>
{
public:
    template <typename...>
    void setup(size_t width, size_t height, size_t channel, size_t batch, Size2D kernel_size, size_t depth_multiplier, Size2D dilation, Size2D stride, bool padding_valid, DataType data_type,
               DataLayout data_layout, const ActivationLayerInfo &act_info, unsigned int n0, bool export_to_cl_image)
    {
        _dilation           = dilation;
        _depth_multiplier   = depth_multiplier;
        _data_type          = data_type;
        _data_layout        = data_layout;
        _act_info           = act_info;
        _n0                 = n0;
        _export_to_cl_image = export_to_cl_image;
        _in_place           = in_place;

        _input_shape   = TensorShape(width, height, channel, batch);
        _weights_shape = TensorShape(kernel_size.width, kernel_size.height, channel * _depth_multiplier);
        _biases_shape  = TensorShape(_weights_shape.z());

        if(padding_valid)
        {
            _conv_info = calculate_same_pad(_input_shape, _weights_shape, PadStrideInfo(stride.width, stride.height), DataLayout::NCHW, _dilation);
        }
        else
        {
            _conv_info = PadStrideInfo(stride.width, stride.height);
        }
    }

    void configure_target()
    {
#if defined(ARM_COMPUTE_OPENCL_ENABLED)
        if(_export_to_cl_image)
        {
            _validate_output &= image2d_from_buffer_supported(CLKernelLibrary::get().get_device());
            _validate_output &= (get_cl_image_pitch_alignment(CLKernelLibrary::get().get_device()) != 0);
        }
#endif // ARM_COMPUTE_OPENCL_ENABLED

        if(!_validate_output)
        {
            return;
        }

        TensorShape input_shape   = _input_shape;
        TensorShape weights_shape = _weights_shape;

        if(_data_layout == DataLayout::NHWC)
        {
            permute(input_shape, PermutationVector(2U, 0U, 1U));
            permute(weights_shape, PermutationVector(2U, 0U, 1U));
        }

        // Create tensors
        _src                      = create_tensor<TensorType>(input_shape, _data_type, 1, QuantizationInfo(), _data_layout);
        _weights                  = create_tensor<TensorType>(weights_shape, _data_type, 1, QuantizationInfo(), _data_layout);
        _biases                   = create_tensor<TensorType>(_biases_shape, _data_type, 1, QuantizationInfo(), _data_layout);
        TensorType *target_to_use = nullptr;
        if(!_in_place)
        {
            _target       = create_tensor<TensorType>(TensorShape(), _data_type, 1, QuantizationInfo(), _data_layout);
            target_to_use = &_target;
        }

        DWCComputeKernelInfo dwc_info;
        dwc_info.n0                         = _n0;
        dwc_info.m0                         = _conv_info.stride().first == 1 && _dilation.x() == 1 ? 8 : 1;
        dwc_info.export_input_to_cl_image   = false;
        dwc_info.export_weights_to_cl_image = _export_to_cl_image;

        const ConvolutionInfo conv_kernel_info
        {
            _conv_info, _depth_multiplier, _act_info, _dilation
        };

        add_padding_x({ &_src, &_biases, &_target }, _data_layout);
        add_padding_x({ &_weights }, _data_layout, _export_to_cl_image); // Don't add left padding if cl image will be used

        // Create Depthwise Convolution configure function
        _dwc.configure(&_src, &_weights, &_biases, target_to_use, dwc_info, conv_kernel_info);

        ARM_COMPUTE_ASSERT(_src.info()->is_resizable());
        ARM_COMPUTE_ASSERT(_weights.info()->is_resizable());
        ARM_COMPUTE_ASSERT(_biases.info()->is_resizable());
        ARM_COMPUTE_ASSERT(_target.info()->is_resizable());
    }

    void allocate_and_run_target()
    {
        if(!_validate_output)
        {
            return;
        }

        // Allocate tensors
        _src.allocator()->allocate();
        _weights.allocator()->allocate();
        _biases.allocator()->allocate();

        ARM_COMPUTE_ASSERT(!_src.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!_weights.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!_biases.info()->is_resizable());
        if(!_in_place)
        {
            _target.allocator()->allocate();
            ARM_COMPUTE_ASSERT(!_target.info()->is_resizable());
        }

        // Fill tensors
        fill(AccessorType(_src), 0);
        fill(AccessorType(_weights), 1);
        fill(AccessorType(_biases), 2);

        // Test Multi DataLayout graph cases, when the data layout changes after configure
        _src.info()->set_data_layout(_data_layout == DataLayout::NCHW ? DataLayout::NHWC : DataLayout::NCHW);
        if(!_in_place)
        {
            _target.info()->set_data_layout(_data_layout == DataLayout::NCHW ? DataLayout::NHWC : DataLayout::NCHW);
        }

        // Compute function
        _dwc.run();

        // Reinstating original data layout for the test suite to properly check the values
        if(!_in_place)
        {
            _target.info()->set_data_layout(_data_layout);
        }
    }

    void compute_reference()
    {
        if(!_validate_output)
        {
            return;
        }

        SimpleTensor<T> src{ _input_shape, _data_type };
        SimpleTensor<T> weights{ _weights_shape, _data_type };
        SimpleTensor<T> biases{ _biases_shape, _data_type };

        fill(src, 0);
        fill(weights, 1);
        fill(biases, 2);

        const ConvolutionInfo info{ _conv_info, _depth_multiplier, _act_info, _dilation };
        const TensorShape     dst_shape = compute_depthwise_convolution_shape(TensorInfo(_input_shape, 1, _data_type), TensorInfo(_weights_shape, 1, _data_type), info);
        _reference                      = reference::activation_layer(reference::depthwise_convolution(src, weights, biases, dst_shape, _conv_info, _depth_multiplier, _dilation), _act_info);
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i)
    {
        switch(tensor.data_type())
        {
            case DataType::F32:
            {
                std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
                library->fill(tensor, distribution, i);
                break;
            }
            case DataType::F16:
            {
                arm_compute::utils::uniform_real_distribution_16bit<half> distribution{ -1.0f, 1.0f };
                library->fill(tensor, distribution, i);
                break;
            }
            default:
                library->fill_tensor_uniform(tensor, i);
        }
    }

    TensorType      _target{};
    SimpleTensor<T> _reference{};

    TensorType   _src{};
    TensorType   _weights{};
    TensorType   _biases{};
    FunctionType _dwc{};

    TensorShape         _input_shape{};
    TensorShape         _weights_shape{};
    TensorShape         _biases_shape{};
    DataType            _data_type{};
    DataLayout          _data_layout{};
    PadStrideInfo       _conv_info{};
    ActivationLayerInfo _act_info{};
    Size2D              _dilation{};
    unsigned int        _depth_multiplier{};
    unsigned int        _n0{};
    bool                _export_to_cl_image{};
    bool                _validate_output{ true };
    bool                _in_place{ false };
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T, bool mixed_layout = false, bool in_place = false>
class DepthwiseConvolutionLayerValidationQuantizedFixture : public DepthwiseConvolutionLayerValidationGenericFixture<TensorType, AccessorType, FunctionType, T, T>
{
public:
    template <typename...>
    void setup(TensorShape in_shape, Size2D kernel_size, PadStrideInfo pad_stride_info, Size2D dilation, unsigned int depth_multiplier, DataType data_type,
               QuantizationInfo input_quantization_info, QuantizationInfo output_quantization_info, DataLayout data_layout, ActivationLayerInfo act_info)
    {
        DepthwiseConvolutionLayerValidationGenericFixture<TensorType, AccessorType, FunctionType, T, T>::setup(in_shape, kernel_size, pad_stride_info, dilation, depth_multiplier, data_type,
                                                                                                               data_type, input_quantization_info, input_quantization_info, output_quantization_info,
                                                                                                               data_layout, act_info, mixed_layout, in_place);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T, typename TW, bool in_place = false>
class DepthwiseConvolutionLayerValidationQuantizedPerChannelFixture : public DepthwiseConvolutionLayerValidationGenericFixture<TensorType, AccessorType, FunctionType, T, TW>
{
public:
    template <typename...>
    void setup(TensorShape in_shape, Size2D kernel_size, PadStrideInfo pad_stride_info, Size2D dilation, unsigned int depth_multiplier, DataType input_data_type, DataType weights_data_type,
               QuantizationInfo input_quantization_info, QuantizationInfo output_quantization_info, DataLayout data_layout, ActivationLayerInfo act_info)
    {
        const float out_scale = output_quantization_info.uniform().scale;
        const float in_scale  = input_quantization_info.uniform().scale;

        std::vector<float>                    weights_scales{};
        std::mt19937                          gen(library->seed());
        std::uniform_real_distribution<float> dis(0.01f, out_scale / in_scale);
        for(size_t i = 0; i < in_shape.z() * depth_multiplier; ++i)
        {
            weights_scales.push_back(dis(gen));
        }

        DepthwiseConvolutionLayerValidationGenericFixture<TensorType, AccessorType, FunctionType, T, TW>::setup(in_shape, kernel_size, pad_stride_info, dilation, depth_multiplier,
                                                                                                                input_data_type, weights_data_type,
                                                                                                                input_quantization_info, QuantizationInfo(weights_scales), output_quantization_info,
                                                                                                                data_layout, act_info, false, in_place);
    }
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_DEPTHWISE_CONVOLUTION_FIXTURE */
