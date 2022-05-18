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

#include "src/gpu/cl/kernels/experimental/dynamic_fusion/ClCompositeKernel.h"
#include "src/core/experimental/dynamic_fusion/ClKernelBuildingAPI.h"

#include "src/core/utils/helpers/float_ops.h"
#include "src/gpu/cl/kernels/ClElementwiseKernel.h"
#include "src/gpu/cl/kernels/ClGemmMatrixMultiplyNativeKernel.h"
#include "tests/CL/CLAccessor.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/reference/ConvolutionLayer.h"
#include "tests/validation/reference/ElementwiseOperations.h"
#include "tests/validation/reference/GEMM.h"
#include "tests/validation/reference/Permute.h"

#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "src/core/AccessWindowStatic.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"

#include "tests/validation/CL/UNIT/dynamic_fusion/Utils.h"

#include <chrono>

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
TEST_SUITE(ClCompositeKernel)
TEST_SUITE(Validate)

TEST_CASE(MoveNet_SubGraph_1_DirectConv2d, framework::DatasetMode::ALL)
{
    /* Computation:
     * out = add(addend, direct_conv2d(lhs, rhs, bias)) (non-broadcast)
     */

    ClCompositeKernel     kernel{};
    ClKernelBlueprint     bp{};
    ClKernelCode          cl_code{};
    ClExecutionDescriptor exec_desc{};
    Status                st{};

    const auto data_type = DataType::F32;
    const auto conv_info = Conv2dDescriptor{ Padding2D{ 1U, 1U, 1U, 1U }, { 1U, 1U } /* stride */ };

    const auto width     = 7U;
    const auto height    = 6U;
    const auto IFM       = 5U;
    const auto OFM       = 4U;
    const auto kernel_sz = 3U;

    const auto src_shape    = TensorShape(IFM, width, height);
    const auto wei_shape    = TensorShape(IFM, kernel_sz, kernel_sz, OFM);
    const auto bia_shape    = TensorShape(OFM);
    const auto addend_shape = TensorShape(1, 1);
    const auto dst_shape    = TensorShape(OFM, width, height);

    auto src_info    = TensorInfo(src_shape, 1, data_type, DataLayout::NHWC);
    auto wei_info    = TensorInfo(wei_shape, 1, data_type, DataLayout::NHWC);
    auto bia_info    = TensorInfo(bia_shape, 1, data_type, DataLayout::NHWC);
    auto addend_info = TensorInfo(addend_shape, 1, data_type, DataLayout::NHWC);
    auto dst_info    = TensorInfo(dst_shape, 1, data_type, DataLayout::NHWC);

    const auto n0 = std::min(OFM, 4u);
    const auto m0 = (OFM > 16) ? ((data_type == DataType::F32) ? 2U : 4U) : 1U;

    const ClDirectConv2dKernelDescriptor direct_conv2d_desc{ conv_info };
    const ClEltwiseAddKernelDescriptor   eltwise_add_desc{};
    const TileDescriptor                 store_tile_info{ Size2D(n0, m0), Size2D(width, height), ClippingStrategy::TOP_LEFT };

    ArgumentID src_id{ g_arg_placeholder };
    ArgumentID wei_id{ g_arg_placeholder };
    ArgumentID bia_id{ g_arg_placeholder };
    ArgumentID acc_id{ g_arg_placeholder };
    ArgumentID acc_1_id{ g_arg_placeholder };
    ArgumentID addend_id{ g_arg_placeholder };
    ArgumentID dst_id{ g_arg_placeholder };

    st = add_tensor(bp, &src_info, src_id);
    st = add_tensor(bp, &wei_info, wei_id);
    st = add_tensor(bp, &bia_info, bia_id);
    st = add_tensor(bp, &dst_info, acc_id);
    st = add_tensor(bp, &dst_info, acc_1_id);
    st = add_tensor(bp, &addend_info, addend_id);
    st = add_tensor(bp, &dst_info, dst_id);

    st = add_kcomp_direct_conv2d(bp, direct_conv2d_desc, src_id, wei_id, bia_id, acc_id);
    st = add_kcomp_eltwise_add(bp, eltwise_add_desc, addend_id, acc_id, acc_1_id);
    st = add_kcomp_store(bp, StoreType::TStoreIndirectWidthSelect, acc_1_id, dst_id);

    exec_desc.skip_sliding_window = true;

    st = set_tile_info(bp, store_tile_info);
    st = build(cl_code, ClCodeBuilderContext{ GpuInfo{ GPUTarget::G71 } }, bp);
    st = tune_static(exec_desc, cl_code);

    CLScheduler::get().default_reinit();
    kernel.configure(CLKernelLibrary::get().get_compile_context(), cl_code);

    // Construct tensors
    CLTensor src{};
    CLTensor wei{};
    CLTensor bia{};
    CLTensor addend{};
    CLTensor dst{};

    // Init tensors
    src.allocator()->init(src_info);
    wei.allocator()->init(wei_info);
    bia.allocator()->init(bia_info);
    addend.allocator()->init(dst_info);
    dst.allocator()->init(dst_info);

    // "Pack" tensors
    ITensorPack tensors{ { src_id, &src },
        { wei_id, &wei },
        { bia_id, &bia },
        { addend_id, &addend },
        { dst_id, &dst } };

    // Allocate and fill tensors
    src.allocator()->allocate();
    wei.allocator()->allocate();
    bia.allocator()->allocate();
    addend.allocator()->allocate();
    dst.allocator()->allocate();

    fill<float>(CLAccessor(src), 0, library.get());
    fill<float>(CLAccessor(wei), 1, library.get());
    fill<float>(CLAccessor(bia), 2, library.get());
    fill<float>(CLAccessor(addend), 3, library.get());

    CLScheduler::get().enqueue_op(kernel, tensors, exec_desc, true);

    // Create reference
    SimpleTensor<float> ref_src_nhwc{ src_shape, data_type, 1, QuantizationInfo(), DataLayout::NHWC };
    SimpleTensor<float> ref_wei_nhwc{ wei_shape, data_type, 1, QuantizationInfo(), DataLayout::NHWC };
    SimpleTensor<float> ref_bia_nhwc{ bia_shape, data_type, 1, QuantizationInfo(), DataLayout::NHWC };
    SimpleTensor<float> ref_addend_nhwc{ dst_shape, data_type, 1, QuantizationInfo(), DataLayout::NHWC };

    // Fill reference
    fill<float>(ref_src_nhwc, 0, library.get());
    fill<float>(ref_wei_nhwc, 1, library.get());
    fill<float>(ref_bia_nhwc, 2, library.get());
    fill<float>(ref_addend_nhwc, 3, library.get());

    auto ref_src    = reference::permute(ref_src_nhwc, PermutationVector(1U, 2U, 0U));
    auto ref_wei    = reference::permute(ref_wei_nhwc, PermutationVector(1U, 2U, 0U));
    auto ref_bia    = reference::permute(ref_bia_nhwc, PermutationVector(1U, 2U, 0U));
    auto ref_addend = reference::permute(ref_addend_nhwc, PermutationVector(1U, 2U, 0U));

    TensorShape dst_shape_nchw{ dst_shape };
    permute(dst_shape_nchw, PermutationVector(1U, 2U, 0U));

    const auto ref_dst = reference::arithmetic_operation(
                             ArithmeticOperation::ADD,
                             ref_addend,
                             reference::convolution_layer<float>(ref_src, ref_wei, ref_bia, dst_shape_nchw,
                                                                 PadStrideInfo
    {
        static_cast<unsigned int>(conv_info.stride.x()),
        static_cast<unsigned int>(conv_info.stride.y()),
        static_cast<unsigned int>(conv_info.pad.left),
        static_cast<unsigned int>(conv_info.pad.top) }),
    data_type,
    ConvertPolicy::SATURATE);

    RelativeTolerance<float> tolerance_f32(0.001f); /**< Tolerance value for comparing reference's output against implementation's output for floating point data types */
    validate(CLAccessor(dst), ref_dst, tolerance_f32);
}

TEST_SUITE_END() // Validate
TEST_SUITE_END() // ClCompositeKernel
TEST_SUITE_END() // DYNAMIC_FUSION
TEST_SUITE_END() // UNIT
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ENABLE_EXPERIMENTAL_DYNAMIC_FUSION */