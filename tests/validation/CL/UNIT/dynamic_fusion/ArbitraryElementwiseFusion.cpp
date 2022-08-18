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
#ifdef ENABLE_EXPERIMENTAL_DYNAMIC_FUSION

#include "src/core/experimental/dynamic_fusion/ClKernelBuildingAPI.h"
#include "src/core/utils/helpers/float_ops.h"
#include "tests/CL/CLAccessor.h"
#include "tests/framework/Macros.h"
#include "tests/validation/Validation.h"
#include "tests/validation/reference/ConvolutionLayer.h"
#include "tests/validation/reference/ElementwiseOperations.h"
#include "tests/validation/reference/Permute.h"

#include "arm_compute/runtime/experimental/ClCompositeOperator.h"
#include "tests/validation/reference/Floor.h"

#include "arm_compute/core/ITensor.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "tests/validation/CL/UNIT/dynamic_fusion/Utils.h"

using namespace arm_compute::experimental::dynamic_fusion;
using namespace arm_compute::test::validation::utils;

namespace arm_compute
{
namespace test
{
namespace validation
{
TEST_SUITE(CL)
TEST_SUITE(UNIT)
TEST_SUITE(DYNAMIC_FUSION)
TEST_SUITE(ArbitraryFusion)

TEST_CASE(ElementwiseBroadcasting, framework::DatasetMode::ALL)
{
    // Test elementwise broadcasting
    const auto data_type   = DataType::F32;
    const auto data_layout = DataLayout::NHWC;

    const auto input_shape = TensorShape(7, 9, 5);
    const auto rhs_shape   = TensorShape(7, 1, 1);
    const auto dst_shape   = TensorShape(7, 9, 5);

    // Tensor Info
    auto input_info  = TensorInfo(input_shape, 1, data_type, data_layout);
    auto addend_info = TensorInfo(rhs_shape, 1, data_type, data_layout);
    auto dst_info    = TensorInfo();

    ElementwiseDescriptor add_desc{ ArithmeticOperation::ADD };

    CLScheduler::get().default_reinit();
    const auto    cl_compile_ctx = CLKernelLibrary::get().get_compile_context();
    OperatorGraph op_graph;

    const auto op_input  = add_tensor(op_graph, input_info);
    const auto op_addend = add_tensor(op_graph, addend_info);
    const auto op_dst    = add_tensor(op_graph, dst_info);

    add_op_elementwise_op(op_graph, add_desc, op_input, op_addend, op_dst);

    const ClWorkloadContext workload_ctx{ GpuInfo{ CLScheduler::get().target() } };
    ClWorkload              workload;
    build(workload, op_graph, workload_ctx);

    ClCompositeOperator op;
    op.configure(cl_compile_ctx, workload);

    // Construct tensors
    CLTensor t_input{};
    CLTensor t_addend{};
    CLTensor t_dst{};

    // Init tensors
    t_input.allocator()->init(input_info);
    t_addend.allocator()->init(addend_info);
    t_dst.allocator()->init(dst_info);

    // Allocate and fill tensors
    t_input.allocator()->allocate();
    t_addend.allocator()->allocate();
    t_dst.allocator()->allocate();

    // Fill
    fill<float>(CLAccessor(t_input), 0, library.get());
    fill<float>(CLAccessor(t_addend), 1, library.get());

    // Pack tensors
    OpTensorBinding bp_tensors({ { op_input, &t_input },
        { op_addend, &t_addend },
        { op_dst, &t_dst }
    });

    // Populate prepare and run pack-maps (including allocating aux tensors)
    ClAuxTensorData aux_tensor_data{};
    TensorPackMap   prepare_pack_map{};
    TensorPackMap   run_pack_map{};
    bind_tensors(aux_tensor_data, prepare_pack_map, run_pack_map, workload, bp_tensors);

    op.prepare(prepare_pack_map);
    op.run(run_pack_map);

    // Create reference
    SimpleTensor<float> ref_input{ input_shape, data_type, 1, QuantizationInfo(), DataLayout::NHWC };
    SimpleTensor<float> ref_addend{ rhs_shape, data_type, 1, QuantizationInfo(), DataLayout::NHWC };

    // Fill reference
    fill<float>(ref_input, 0, library.get());
    fill<float>(ref_addend, 1, library.get());

    auto ref_input_nchw  = reference::permute(ref_input, PermutationVector(1U, 2U, 0U));
    auto ref_addend_nchw = reference::permute(ref_addend, PermutationVector(1U, 2U, 0U));

    auto dst_shape_nchw = dst_shape;
    permute(dst_shape_nchw, PermutationVector(1U, 2U, 0U));

    auto ref_t_dst_nchw = reference::arithmetic_operation(
                              ArithmeticOperation::ADD,
                              ref_input_nchw,
                              ref_addend_nchw,
                              data_type,
                              ConvertPolicy{});

    const auto ref_t_dst = reference::permute(ref_t_dst_nchw, PermutationVector(2U, 0U, 1U));

    RelativeTolerance<float> tolerance_f32(0.001f);
    validate(CLAccessor(t_dst), ref_t_dst_nchw, tolerance_f32);
}
TEST_CASE(DivFloor, framework::DatasetMode::ALL)
{
    // x = floor(div(input, input2))
    const auto data_type    = DataType::F32;
    const auto eltwise_info = ElementwiseDescriptor{ ArithmeticOperation::DIV };

    // Tensor Values
    const auto width  = 7U;
    const auto height = 6U;

    // Shapes
    const auto input1_shape = TensorShape(width, height);
    const auto input2_shape = TensorShape(width, height);
    const auto dst_shape    = TensorShape(width, height);

    // Create reference
    SimpleTensor<float> ref_src_nhwc{ input1_shape, data_type, 1, QuantizationInfo(), DataLayout::NHWC };
    SimpleTensor<float> ref_src2_nhwc{ input2_shape, data_type, 1, QuantizationInfo(), DataLayout::NHWC };

    // Fill reference
    fill<float>(ref_src_nhwc, 0, library.get());
    fill<float>(ref_src2_nhwc, 1, library.get());

    auto ref_src  = reference::permute(ref_src_nhwc, PermutationVector(1U, 2U, 0U));
    auto ref_src2 = reference::permute(ref_src2_nhwc, PermutationVector(1U, 2U, 0U));

    TensorShape dst_shape_nchw{ dst_shape };
    permute(dst_shape_nchw, PermutationVector(1U, 2U, 0U));

    const auto ref_dst_nchw = reference::floor_layer(reference::arithmetic_operation(
                                                         ArithmeticOperation::DIV,
                                                         ref_src,
                                                         ref_src2,
                                                         data_type,
                                                         ConvertPolicy::SATURATE));

    const auto ref_t_dst = reference::permute(ref_dst_nchw, PermutationVector(2U, 0U, 1U));

    // Tensor Info
    auto input1_info = TensorInfo(input1_shape, 1, data_type, DataLayout::NHWC);
    auto input2_info = TensorInfo(input2_shape, 1, data_type, DataLayout::NHWC);
    auto dst_info    = TensorInfo();
    auto acc_info    = TensorInfo(); // Intermediate tensor for division

    // Initialise Scheduler
    CLScheduler::get().default_reinit();
    const auto    cl_compile_ctx = CLKernelLibrary::get().get_compile_context();
    OperatorGraph op_graph;

    // add tensors
    auto op_input1 = add_tensor(op_graph, input1_info);
    auto op_input2 = add_tensor(op_graph, input2_info);
    auto op_acc    = add_tensor(op_graph, acc_info);
    auto op_dst    = add_tensor(op_graph, dst_info);

    add_op_elementwise_op(op_graph, eltwise_info, op_input1, op_input2, op_acc);
    add_op_floor(op_graph, FloorDescriptor(), op_acc, op_dst);

    const ClWorkloadContext workload_ctx{ GpuInfo{ CLScheduler::get().target() } };
    ClWorkload              workload;
    build(workload, op_graph, workload_ctx);

    ClCompositeOperator op;
    op.configure(cl_compile_ctx, workload);

    // Configure and add tensors.
    CLTensor t_input1{};
    CLTensor t_input2{};
    CLTensor t_dst{};

    // Init Tensors
    t_input1.allocator()->init(input1_info);
    t_input2.allocator()->init(input2_info);
    t_dst.allocator()->init(dst_info);

    // Allocate and fill tensors
    t_input1.allocator()->allocate();
    t_input2.allocator()->allocate();
    t_dst.allocator()->allocate();

    fill<float>(CLAccessor(t_input1), 0, library.get());
    fill<float>(CLAccessor(t_input2), 1, library.get());

    // "Pack" tensors
    OpTensorBinding bp_tensors({ { op_input1, &t_input1 },
        { op_input2, &t_input2 },
        { op_dst, &t_dst }
    });

    // Populate prepare and run pack-maps (including allocating aux tensors)
    ClAuxTensorData aux_tensor_data{};
    TensorPackMap   prepare_pack_map{};
    TensorPackMap   run_pack_map{};
    bind_tensors(aux_tensor_data, prepare_pack_map, run_pack_map, workload, bp_tensors);

    op.prepare(prepare_pack_map);
    op.run(run_pack_map);

    RelativeTolerance<float> tolerance_f32(0.001f); /**< Tolerance value for comparing reference's output against implementation's output for floating point data types */
    validate(CLAccessor(t_dst), ref_dst_nchw, tolerance_f32);
}
TEST_CASE(Dconv2dAddDiv, framework::DatasetMode::ALL)
{
    // output = div(divend, add(addend, conv2d1x1(direct_conv)(input, weights, bias)))
    const auto data_type   = DataType::F32;
    const auto data_layout = DataLayout::NHWC;

    const auto input_shape  = TensorShape(384, 12, 12);
    const auto weight_shape = TensorShape(384, 1, 1, 16);
    const auto dst_shape    = TensorShape(16, 12, 12);

    // Tensor Info
    auto input_info  = TensorInfo(input_shape, 1, data_type, data_layout);
    auto weight_info = TensorInfo(weight_shape, 1, data_type, data_layout);
    auto addend_info = TensorInfo(dst_shape, 1, data_type, data_layout);
    auto divend_info = TensorInfo(dst_shape, 1, data_type, data_layout);
    auto acc_info    = TensorInfo(); // Intermediate tensor for conv
    auto acc_1_info  = TensorInfo();
    auto dst_info    = TensorInfo();

    Conv2dDescriptor      conv2d_desc{};
    ElementwiseDescriptor add_desc{ ArithmeticOperation::ADD };
    ElementwiseDescriptor div_desc{ ArithmeticOperation::DIV };

    CLScheduler::get().default_reinit();
    const auto    cl_compile_ctx = CLKernelLibrary::get().get_compile_context();
    OperatorGraph op_graph;

    const auto op_input  = add_tensor(op_graph, input_info);
    const auto op_weight = add_tensor(op_graph, weight_info);
    const auto op_addend = add_tensor(op_graph, addend_info);
    const auto op_divend = add_tensor(op_graph, divend_info);
    const auto op_acc    = add_tensor(op_graph, acc_info);   // temp accumulator; TensorInfo to be inferred
    const auto op_acc_1  = add_tensor(op_graph, acc_1_info); // temp accumulator; TensorInfo to be inferred
    const auto op_dst    = add_tensor(op_graph, dst_info);

    auto conv2d = add_op_conv2d(op_graph, conv2d_desc, op_input, op_weight, op_acc);
    force_conv2d_method(op_graph, conv2d, ConvolutionMethod::DIRECT);
    add_op_elementwise_op(op_graph, add_desc, op_acc, op_addend, op_acc_1);
    add_op_elementwise_op(op_graph, div_desc, op_acc_1, op_divend, op_dst);

    const ClWorkloadContext workload_ctx{ GpuInfo{ CLScheduler::get().target() } };
    ClWorkload              workload;
    build(workload, op_graph, workload_ctx);

    ClCompositeOperator op;
    op.configure(cl_compile_ctx, workload);

    // Construct tensors
    CLTensor t_input{};
    CLTensor t_weight{};
    CLTensor t_addend{};
    CLTensor t_divend{};
    CLTensor t_dst{};

    // Init tensors
    t_input.allocator()->init(input_info);
    t_weight.allocator()->init(weight_info);
    t_divend.allocator()->init(divend_info);
    t_addend.allocator()->init(addend_info);
    t_dst.allocator()->init(dst_info);

    // Allocate and fill tensors
    t_input.allocator()->allocate();
    t_weight.allocator()->allocate();
    t_divend.allocator()->allocate();
    t_addend.allocator()->allocate();
    t_dst.allocator()->allocate();

    // Fill
    fill<float>(CLAccessor(t_input), 0, library.get());
    fill<float>(CLAccessor(t_weight), 1, library.get());
    fill<float>(CLAccessor(t_addend), 2, library.get());
    fill<float>(CLAccessor(t_divend), 3, library.get());

    // Pack tensors
    OpTensorBinding bp_tensors({ { op_input, &t_input },
        { op_weight, &t_weight },
        { op_addend, &t_addend },
        { op_divend, &t_divend },
        { op_dst, &t_dst }
    });

    // Populate prepare and run pack-maps (including allocating aux tensors)
    ClAuxTensorData aux_tensor_data{};
    TensorPackMap   prepare_pack_map{};
    TensorPackMap   run_pack_map{};
    bind_tensors(aux_tensor_data, prepare_pack_map, run_pack_map, workload, bp_tensors);

    op.prepare(prepare_pack_map);
    op.run(run_pack_map);

    // Create reference
    SimpleTensor<float> ref_input{ input_shape, data_type, 1, QuantizationInfo(), DataLayout::NHWC };
    SimpleTensor<float> ref_weight{ weight_shape, data_type, 1, QuantizationInfo(), DataLayout::NHWC };
    SimpleTensor<float> ref_bias_placeholder{ dst_shape, data_type, 1, QuantizationInfo(), DataLayout::NHWC };
    SimpleTensor<float> ref_addend{ dst_shape, data_type, 1, QuantizationInfo(), DataLayout::NHWC };
    SimpleTensor<float> ref_divend{ dst_shape, data_type, 1, QuantizationInfo(), DataLayout::NHWC };

    // Fill reference
    fill<float>(ref_input, 0, library.get());
    fill<float>(ref_weight, 1, library.get());
    fill<float>(ref_addend, 2, library.get());
    fill<float>(ref_divend, 3, library.get());

    auto ref_input_nchw            = reference::permute(ref_input, PermutationVector(1U, 2U, 0U));
    auto ref_weight_nchw           = reference::permute(ref_weight, PermutationVector(1U, 2U, 0U));
    auto ref_bias_placeholder_nchw = reference::permute(ref_bias_placeholder, PermutationVector(1U, 2U, 0U));
    auto ref_addend_nchw           = reference::permute(ref_addend, PermutationVector(1U, 2U, 0U));
    auto ref_divend_nchw           = reference::permute(ref_divend, PermutationVector(1U, 2U, 0U));

    auto dst_shape_nchw = dst_shape;
    permute(dst_shape_nchw, PermutationVector(1U, 2U, 0U));

    PadStrideInfo legacy_pad_stride(conv2d_desc.stride.x(), conv2d_desc.stride.y(), conv2d_desc.pad.left, conv2d_desc.pad.right, conv2d_desc.pad.top, conv2d_desc.pad.bottom, DimensionRoundingType{});
    auto          ref_acc_nchw = reference::arithmetic_operation(
                                     ArithmeticOperation::ADD,
                                     ref_addend_nchw,
                                     reference::convolution_layer(ref_input_nchw, ref_weight_nchw, ref_bias_placeholder_nchw, dst_shape_nchw, legacy_pad_stride, conv2d_desc.dilation),
                                     data_type,
                                     ConvertPolicy{});

    auto ref_t_dst_nchw = reference::arithmetic_operation(
                              ArithmeticOperation::DIV,
                              ref_acc_nchw,
                              ref_divend_nchw,
                              data_type,
                              ConvertPolicy{});

    const auto ref_t_dst = reference::permute(ref_t_dst_nchw, PermutationVector(2U, 0U, 1U));

    RelativeTolerance<float> tolerance_f32(0.001f);
    validate(CLAccessor(t_dst), ref_t_dst_nchw, tolerance_f32);
}

TEST_SUITE_END() // ArbitraryFusion
TEST_SUITE_END() // DYNAMIC_FUSION
TEST_SUITE_END() // UNIT
TEST_SUITE_END() // CL

} // namespace validation
} // namespace test
} // namespace arm_compute

#endif /* ENABLE_EXPERIMENTAL_DYNAMIC_FUSION */
