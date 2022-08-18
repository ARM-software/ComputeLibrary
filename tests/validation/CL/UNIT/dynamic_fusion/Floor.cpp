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
#include "arm_compute/core/TensorInfo.h"

#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/experimental/ClWorkload.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "arm_compute/runtime/experimental/ClCompositeOperator.h"
#include "src/core/experimental/dynamic_fusion/WorkloadImpl/ClKernelDescriptors.h"
#include "tests/CL/CLAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/validation/CL/UNIT/dynamic_fusion/Utils.h"
#include "tests/validation/Validation.h"

#include "tests/validation/reference/Floor.h"
#include "tests/validation/reference/Permute.h"

#ifdef ARM_COMPUTE_ASSERTS_ENABLED
#include "tests/SimpleTensorPrinter.h"
#endif /* ARM_COMPUTE_ASSERTS_ENABLED */

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
TEST_CASE(Operator_Floor_1_F32, framework::DatasetMode::ALL)
{
    /* Computation:
     * out = floor(input)
     */
    const auto data_type    = DataType::F32;
    const auto data_layout  = DataLayout::NHWC;
    const auto t_shape      = TensorShape(32, 16);
    auto       t_input_info = TensorInfo(t_shape, 1, data_type, data_layout);
    auto       t_dst_info   = TensorInfo();

    FloorDescriptor floor_desc{};

    // Create reference
    SimpleTensor<float> ref_t_input{ t_shape, data_type, 1, QuantizationInfo(), DataLayout::NHWC };

    // Fill reference
    fill<float>(ref_t_input, 0, library.get());

    auto ref_t_input_nchw = reference::permute(ref_t_input, PermutationVector(1U, 2U, 0U));
    auto t_dst_shape_nchw = t_shape;
    permute(t_dst_shape_nchw, PermutationVector(1U, 2U, 0U));

    auto       ref_t_dst_nchw = reference::floor_layer(ref_t_input_nchw);
    const auto ref_t_dst      = reference::permute(ref_t_dst_nchw, PermutationVector(2U, 0U, 1U));

    CLScheduler::get().default_reinit();
    const auto    cl_compile_ctx = CLKernelLibrary::get().get_compile_context();
    OperatorGraph op_graph;

    const auto op_t_input = add_tensor(op_graph, t_input_info);
    const auto op_t_dst   = add_tensor(op_graph, t_dst_info);

    add_op_floor(op_graph, floor_desc, op_t_input, op_t_dst);

    const ClWorkloadContext workload_ctx{ GpuInfo{ CLScheduler::get().target() } };
    ClWorkload              workload;
    build(workload, op_graph, workload_ctx);

    ClCompositeOperator op;
    op.configure(cl_compile_ctx, workload);

    // Construct tensors
    CLTensor t_input{};
    CLTensor t_dst{};

    // Init tensors
    t_input.allocator()->init(t_input_info);
    t_dst.allocator()->init(t_dst_info);

    // Allocate and fill tensors
    t_input.allocator()->allocate();
    t_dst.allocator()->allocate();
    fill<float>(CLAccessor(t_input), 0, library.get());
    // "Pack" tensors
    OpTensorBinding bp_tensors({ { op_t_input, &t_input },
        { op_t_dst, &t_dst }
    });

    // Populate prepare and run pack-maps (including allocating aux tensors)
    ClAuxTensorData aux_tensor_data{};
    TensorPackMap   prepare_pack_map{};
    TensorPackMap   run_pack_map{};
    bind_tensors(aux_tensor_data, prepare_pack_map, run_pack_map, workload, bp_tensors);

    op.prepare(prepare_pack_map);
    op.run(run_pack_map);
    RelativeTolerance<float> tolerance_f32(0.001f); /**< Tolerance value for comparing reference's output against implementation's output for floating point data types */
    validate(CLAccessor(t_dst), ref_t_dst_nchw, tolerance_f32);
}

TEST_SUITE_END() // DYNAMIC_FUSION
TEST_SUITE_END() // UNIT
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ENABLE_EXPERIMENTAL_DYNAMIC_FUSION */