/*
 * Copyright (c) 2022 Arm Limited.
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
#ifndef ARM_COMPUTE_TEST_INDIRECT_CONV2D_ADDRESS_PRECALCULATION_FIXTURE
#define ARM_COMPUTE_TEST_INDIRECT_CONV2D_ADDRESS_PRECALCULATION_FIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "tests/Globals.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/Helpers.h"
#include "tests/validation/reference/IndirectConv2dAddressPrecalculation.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
using namespace arm_compute::misc::shape_calculator;

template <typename TensorType, typename AccessorType, typename OperatorType>
class IndirectConv2dAddressPrecalculationValidationFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(unsigned int src_w,
               unsigned int src_h,
               unsigned int src_b,
               unsigned int wei_w,
               unsigned int wei_h,
               unsigned int pad,
               unsigned int stride,
               unsigned int m0)
    {
        DirectConvComputeKernelInfo desc;
        desc.m0                         = m0;
        desc.n0                         = 1;     // Not used by the kernel
        desc.k0                         = 1;     // Not used by the kernel
        desc.export_weights_to_cl_image = false; // Not used by the kernel

        const PadStrideInfo conv_info(stride, stride, pad, pad);

        const TensorShape shape_conv_src(23, // The input channels are not used by the kernel
                                         src_w,
                                         src_h,
                                         src_b);

        const TensorShape shape_conv_wei(23, // The input channels are not used by the kernel
                                         wei_w,
                                         wei_h,
                                         23 // The output channels are not used by the kernel
                                        );

        // The result of the kernel does not change with the datatype. Hence, we can fix it to Fp16 for validation purposes
        const DataType data_type = DataType::F16;

        _target    = compute_target(shape_conv_src, shape_conv_wei, data_type, conv_info, desc);
        _reference = compute_reference(shape_conv_src, shape_conv_wei, data_type, conv_info, desc);
    }

protected:
    TensorType compute_target(TensorShape shape_conv_src, TensorShape shape_conv_wei, DataType data_type, const PadStrideInfo &conv_info, const DirectConvComputeKernelInfo &desc)
    {
        TensorInfo src_conv_info(shape_conv_src, 1, data_type, DataLayout::NHWC);
        TensorInfo wei_conv_info(shape_conv_wei, 1, data_type, DataLayout::NHWC);
        TensorType dst;

        // The output tensor will be auto-initialized within the function

        // Create and configure function
        OperatorType func;
        func.configure(&src_conv_info, &wei_conv_info, dst.info(), conv_info, desc);

        add_padding_x({ &dst });

        // Allocate tensors
        dst.allocator()->allocate();

        // Compute GEMM LHS matrix reshape function
        ITensorPack tensors = { { ACL_DST, &dst } };
        func.run(tensors);

        return dst;
    }

    SimpleTensor<int32_t> compute_reference(TensorShape shape_conv_src, TensorShape shape_conv_wei, DataType data_type, const PadStrideInfo &conv_info, const DirectConvComputeKernelInfo &desc)
    {
        ARM_COMPUTE_UNUSED(data_type);
        TensorShape shape_out         = compute_indirect_buffer_shape(shape_conv_src, DataLayout::NHWC, shape_conv_wei, conv_info, desc);
        TensorShape output_conv_shape = compute_deep_convolution_shape(shape_conv_src, DataLayout::NHWC, shape_conv_wei, conv_info);

        return reference::indirect_conv2d_addr_precalculation(shape_conv_src, shape_conv_wei, output_conv_shape, shape_out, conv_info);
    }

    TensorType            _target{};
    SimpleTensor<int32_t> _reference{};
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_INDIRECT_CONV2D_ADDRESS_PRECALCULATION_FIXTURE */