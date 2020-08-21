/*
 * Copyright (c) 2017-2020 Arm Limited.
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
#ifndef ARM_COMPUTE_TEST_IM2COL_FIXTURE
#define ARM_COMPUTE_TEST_IM2COL_FIXTURE

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

namespace arm_compute
{
namespace test
{
namespace validation
{
using namespace arm_compute::misc::shape_calculator;

template <typename TensorType, typename AccessorType, typename FunctionType, typename T, bool batch_size_on_z>
class Im2ColValidationFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape input_shape, DataType data_type, const Size2D &kernel_dims, const PadStrideInfo &conv_info, const QuantizationInfo &quant_info, const DataLayout &data_layout,
               unsigned int num_groups)
    {
        _kernel_dims = kernel_dims;
        _conv_info   = conv_info;
        _quant_info  = quant_info;
        _data_layout = data_layout;
        _has_bias    = data_type != DataType::QASYMM8;
        _num_groups  = num_groups;

        TensorInfo input_info(input_shape, 1, data_type);
        input_info.set_data_layout(_data_layout);

        const TensorShape output_shape = compute_im2col_conv_shape(&input_info, _kernel_dims, _conv_info, _has_bias, Size2D(1U, 1U), batch_size_on_z && _num_groups == 1, _num_groups);
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
        im2col_func.configure(&src, &dst, _kernel_dims, _conv_info, _has_bias, Size2D(1U, 1U), _num_groups);

        ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Allocate tensors
        src.allocator()->allocate();
        dst.allocator()->allocate();

        ARM_COMPUTE_EXPECT(!src.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!dst.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Fill tensors
        fill(AccessorType(src));

        // Compute function
        im2col_func.run();

        return dst;
    }

    void compute_reference(const TensorShape &input_shape, const TensorShape &output_shape, DataType data_type)
    {
        // Create reference
        SimpleTensor<T> src{ input_shape, data_type, 1, _quant_info, _data_layout };
        _reference = SimpleTensor<T>(output_shape, data_type, 1, _quant_info, DataLayout::NCHW);

        // Fill reference
        fill(src);

        reference::im2col<T>(src, _reference, _kernel_dims, _conv_info, _has_bias, _num_groups);
    }
    TensorType       _target{};
    SimpleTensor<T>  _reference{};
    Size2D           _kernel_dims{};
    PadStrideInfo    _conv_info{};
    DataLayout       _data_layout{};
    QuantizationInfo _quant_info{};
    bool             _has_bias{};
    unsigned int     _num_groups{};
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_IM2COL_FIXTURE */
