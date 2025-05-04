/*
 * Copyright (c) 2017-2021, 2023-2025 Arm Limited.
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
#ifndef ACL_TESTS_VALIDATION_FIXTURES_IM2COLFIXTURE_H
#define ACL_TESTS_VALIDATION_FIXTURES_IM2COLFIXTURE_H

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/runtime/Tensor.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/reference/Im2Col.h"

#include <type_traits>

namespace arm_compute
{
namespace test
{
namespace validation
{
using namespace arm_compute::misc::shape_calculator;

template <typename TensorType, typename AccessorType, typename FunctionType, typename T, bool batch_size_on_z>
class Im2ColOpValidationGenericFixture : public framework::Fixture
{
public:
    void setup(TensorShape input_shape, DataType data_type, const Size2D &kernel_dims, const PadStrideInfo &conv_info, const QuantizationInfo &quant_info, const DataLayout &data_layout,
               unsigned int num_groups, unsigned int channel_pad_right)
    {
        if(std::is_same<TensorType, Tensor>::value &&  // Cpu
            data_type == DataType::F16 && !CPUInfo::get().has_fp16())
        {
            return;
        }

        _kernel_dims     = kernel_dims;
        _conv_info       = conv_info;
        _quant_info      = quant_info;
        _data_layout     = data_layout;
        _has_bias        = data_type != DataType::QASYMM8;
        _num_groups      = num_groups;
        _channel_pad_right = channel_pad_right;

        if(_data_layout == DataLayout::NHWC)
        {
            permute(input_shape, PermutationVector(2U, 0U, 1U));
        }

        TensorInfo input_info(input_shape, 1, data_type);
        input_info.set_data_layout(_data_layout);

        const TensorShape output_shape = compute_im2col_conv_shape(&input_info, _kernel_dims, _conv_info,
            _has_bias, Size2D(1U, 1U), batch_size_on_z && _num_groups == 1, _num_groups, _channel_pad_right);

        _target                        = compute_target(input_shape, output_shape, data_type);

        compute_reference(input_shape, output_shape, data_type);
    }

protected:
    template <typename U>
    void fill(U &&tensor)
    {
        library->fill_tensor_uniform(tensor, 0);
    }

    TensorType compute_target(const TensorShape &input_shape, const TensorShape &output_shape, DataType data_type)
    {
        // Create tensors
        TensorType src = create_tensor<TensorType>(input_shape, data_type, 1, _quant_info, _data_layout);
        TensorType dst = create_tensor<TensorType>(output_shape, data_type, 1, _quant_info);

        // Create and configure function
        FunctionType im2col_func;
        configure_function<TensorType>(im2col_func, src.info(), dst.info(), _kernel_dims, _conv_info, _has_bias, Size2D(1U, 1U), _num_groups, _channel_pad_right);

        ARM_COMPUTE_ASSERT(src.info()->is_resizable());
        ARM_COMPUTE_ASSERT(dst.info()->is_resizable());

        // Allocate tensors
        src.allocator()->allocate();
        dst.allocator()->allocate();

        ARM_COMPUTE_ASSERT(!src.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!dst.info()->is_resizable());

        // Fill tensors
        fill(AccessorType(src));

         // Garbage values should be replaced by 0 when testing channel padding
        fill(AccessorType(dst));

        arm_compute::ITensorPack pack =
        {
            { arm_compute::TensorType::ACL_SRC, &src },
            { arm_compute::TensorType::ACL_DST, &dst }
        };

        // Compute function
        im2col_func.run(pack);

        return dst;
    }

    void compute_reference(const TensorShape &input_shape, const TensorShape &output_shape, DataType data_type)
    {
        // Create reference
        SimpleTensor<T> src{ input_shape, data_type, 1, _quant_info, _data_layout };
        _reference = SimpleTensor<T>(output_shape, data_type, 1, _quant_info, DataLayout::NCHW);

        // Fill reference
        fill(src);

        reference::im2col<T>(src, _reference, _kernel_dims, _conv_info, _has_bias, _num_groups, _channel_pad_right);
    }

    TensorType       _target{};
    SimpleTensor<T>  _reference{};
    Size2D           _kernel_dims{};
    PadStrideInfo    _conv_info{};
    DataLayout       _data_layout{};
    QuantizationInfo _quant_info{};
    bool             _has_bias{};
    unsigned int     _num_groups{};
    int              _channel_pad_right{};

private:
    template<typename TensorT>
    auto configure_function(FunctionType &func,
                   ITensorInfo            *src,
                   ITensorInfo            *dst,
                   const Size2D           &kernel_dims,
                   const PadStrideInfo    &conv_info,
                   bool                    has_bias,
                   const Size2D           &dilation,
                   unsigned int            num_groups,
                   unsigned int            channel_pad_right)
        -> typename std::enable_if<
            std::is_same<TensorT, Tensor>::value, // Cpu
        void>::type
    {
        static_assert(std::is_same<TensorT, TensorType>::value,
            "configure_function helper must be used with the class TensorType");

        func.configure(src, dst, kernel_dims, conv_info, has_bias, dilation, num_groups, channel_pad_right);
    }

    template<typename TensorT>
    auto configure_function(FunctionType &func,
                   ITensorInfo            *src,
                   ITensorInfo            *dst,
                   const Size2D           &kernel_dims,
                   const PadStrideInfo    &conv_info,
                   bool                    has_bias,
                   const Size2D           &dilation,
                   unsigned int            num_groups,
                   unsigned int            channel_pad_right)
        -> typename std::enable_if<
            !std::is_same<TensorT, Tensor>::value, // Gpu
        void>::type
    {
        static_assert(std::is_same<TensorT, TensorType>::value,
            "configure_function helper must be used with the class TensorType");

        ARM_COMPUTE_UNUSED(channel_pad_right);
        func.configure(src, dst, kernel_dims, conv_info, has_bias, dilation, num_groups);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T, bool batch_size_on_z>
class Im2ColOpValidationFixture : public Im2ColOpValidationGenericFixture<TensorType, AccessorType, FunctionType, T, batch_size_on_z>
{
public:
    void setup(TensorShape input_shape, DataType data_type, const Size2D &kernel_dims, const PadStrideInfo &conv_info,
        const QuantizationInfo &quant_info, const DataLayout &data_layout,
               unsigned int num_groups)
    {
        Im2ColOpValidationGenericFixture<TensorType, AccessorType, FunctionType, T, batch_size_on_z>::setup(
            input_shape,
            data_type,
            kernel_dims,
            conv_info,
            quant_info,
            data_layout,
            num_groups,
            0 /* channel_pad_right */
        );
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T, bool batch_size_on_z>
class Im2ColOpValidationWithChannelPadFixture : public Im2ColOpValidationGenericFixture<TensorType, AccessorType, FunctionType, T, batch_size_on_z>
{
public:
    void setup(TensorShape input_shape, DataType data_type, const Size2D &kernel_dims, const PadStrideInfo &conv_info, const QuantizationInfo &quant_info, const DataLayout &data_layout,
               unsigned int num_groups, unsigned int channel_pad_right)
    {
        Im2ColOpValidationGenericFixture<TensorType, AccessorType, FunctionType, T, batch_size_on_z>::setup(
            input_shape,
            data_type,
            kernel_dims,
            conv_info,
            quant_info,
            data_layout,
            num_groups,
            channel_pad_right
        );
    }
};

} // namespace validation
} // namespace test
} // namespace arm_compute
#endif // ACL_TESTS_VALIDATION_FIXTURES_IM2COLFIXTURE_H
