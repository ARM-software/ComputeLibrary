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
#if defined(ENABLE_EXPERIMENTAL_DYNAMIC_FUSION)

#include "src/gpu/cl/kernels/experimental/dynamic_fusion/ClCompositeKernel.h"

#include "src/core/utils/helpers/float_ops.h"
#include "src/gpu/cl/kernels/ClElementwiseKernel.h"
#include "src/gpu/cl/kernels/ClGemmMatrixMultiplyNativeKernel.h"
#include "tests/CL/CLAccessor.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/reference/ElementwiseOperations.h"
#include "tests/validation/reference/GEMM.h"

#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "src/core/AccessWindowStatic.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"

#include <chrono>

using namespace arm_compute::experimental::dynamic_fusion;

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
/** Macros which measures the wall clock time, and records it into a map measurement_map with name clock_name */
#define TICK(clock_name) \
    auto clock_name##_tick = std::chrono::high_resolution_clock::now();
#define TOCK(clock_name, measurement_map)                                               \
    auto clock_name##_tock                 = std::chrono::high_resolution_clock::now(); \
    measurement_map["\"" #clock_name "\""] = duration_cast<microseconds>(clock_name##_tock - clock_name##_tick);
#define TOCK_AVG(clock_name, measurement_map, num_iterations)                           \
    auto clock_name##_tock                 = std::chrono::high_resolution_clock::now(); \
    measurement_map["\"" #clock_name "\""] = duration_cast<microseconds>((clock_name##_tock - clock_name##_tick) / (num_iterations));

template <typename T, typename U>
void fill(U &&tensor, int seed)
{
    static_assert(std::is_floating_point<T>::value || std::is_same<T, half>::value, "Only floating point data types supported.");
    using DistributionType = typename std::conditional<std::is_same<T, half>::value, arm_compute::utils::uniform_real_distribution_16bit<T>, std::uniform_real_distribution<T>>::type;

    DistributionType distribution{ T(-1.0f), T(1.0f) };
    library->fill(tensor, distribution, seed);

    // Fill border with infinity in order to check the presence of NaN values (i.e. inf * 0)
    DistributionType distribution_inf{ T(std::numeric_limits<float>::infinity()), T(std::numeric_limits<float>::infinity()) };
    library->fill_borders_with_garbage(tensor, distribution_inf, seed);
}

void set_build_options(ClKernelCode &cl_code, GemmNativeDescriptor gemm_native_desc,
                       const TensorInfo &t_lhs_info,
                       const TensorInfo &t_rhs_info,
                       const TensorInfo *t_bias_info,
                       const TensorInfo &t_dst_info)
{
    CLBuildOptions ref_cl_build_options;
    {
        // If reinterpret_input_as_3d = reinterpret_output_as_3d = true,
        // we will dispatch a batched-GEMM to reduce the complexity of the address calculation within the OpenCL kernel.
        // This means that the actual m used by the kernel is given by dst->dimension(1) and not by gemm_info.m
        auto reinterpret_input_as_3d  = gemm_native_desc.reinterpret_input_as_3d;
        auto reinterpret_output_as_3d = gemm_native_desc.depth_output_gemm3d != 0;
        auto _slide_matrix_b          = (t_rhs_info.num_dimensions() >= t_lhs_info.num_dimensions());
        auto _use_dummy_work_items    = false;
        // In case both input and dst have to be reinterpreted as 3D tensors,
        // force reinterpret_input_as_3d and reinterpret_output_as_3d to be false.
        if(reinterpret_input_as_3d == reinterpret_output_as_3d)
        {
            reinterpret_input_as_3d  = false;
            reinterpret_output_as_3d = false;
        }

        const unsigned int internal_m = reinterpret_output_as_3d ? gemm_native_desc.m : t_dst_info.dimension(1);

        const unsigned int h_gemm_3d = reinterpret_output_as_3d ? t_dst_info.dimension(1) : t_lhs_info.dimension(1);
        const unsigned int d_gemm_3d = reinterpret_output_as_3d ? t_dst_info.dimension(2) : t_lhs_info.dimension(2);

        // Calculate partial (store instead of load) M0 and partial N0 for the partial blocks at the end of a row/column if any. This is to avoid padding.
        const unsigned int partial_store_m0 = internal_m % gemm_native_desc.lhs_info.m0;
        const unsigned int partial_store_n0 = gemm_native_desc.n % gemm_native_desc.rhs_info.n0;

        // Shrink M0 to be always <= M (internal_m) to prevent out-of-bounds reads.
        const unsigned int internal_m0 = std::min(internal_m, gemm_native_desc.lhs_info.m0);

        ref_cl_build_options.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(t_dst_info.data_type()));
        ref_cl_build_options.add_option_if(!(helpers::float_ops::is_one(gemm_native_desc.alpha)), "-DALPHA=" + float_to_string_with_full_precision(gemm_native_desc.alpha));
        ref_cl_build_options.add_option_if(t_bias_info != nullptr, "-DBETA=" + float_to_string_with_full_precision(gemm_native_desc.beta));
        ref_cl_build_options.add_option_if(helpers::float_ops::is_one(gemm_native_desc.beta), "-DUNIT_BETA");
        ref_cl_build_options.add_option_if(gemm_native_desc.broadcast_bias, "-DBROADCAST_BIAS");
        ref_cl_build_options.add_option_if(reinterpret_input_as_3d, "-DREINTERPRET_INPUT_AS_3D");
        ref_cl_build_options.add_option_if(reinterpret_output_as_3d, "-DREINTERPRET_OUTPUT_AS_3D");
        ref_cl_build_options.add_option_if(reinterpret_input_as_3d || reinterpret_output_as_3d, "-DHEIGHT_GEMM3D=" + support::cpp11::to_string(h_gemm_3d));
        ref_cl_build_options.add_option_if(reinterpret_input_as_3d || reinterpret_output_as_3d, "-DDEPTH_GEMM3D=" + support::cpp11::to_string(d_gemm_3d));
        ref_cl_build_options.add_option_if(!_slide_matrix_b, "-DMATRIX_B_DEPTH=" + support::cpp11::to_string(t_rhs_info.dimension(2)));
        ref_cl_build_options.add_option_if(_use_dummy_work_items, "-DDUMMY_WORK_ITEMS");
        ref_cl_build_options.add_option("-DM=" + support::cpp11::to_string(internal_m));
        ref_cl_build_options.add_option("-DN=" + support::cpp11::to_string(gemm_native_desc.n));
        ref_cl_build_options.add_option("-DK=" + support::cpp11::to_string(gemm_native_desc.k));
        ref_cl_build_options.add_option("-DM0=" + support::cpp11::to_string(internal_m0));
        ref_cl_build_options.add_option("-DN0=" + support::cpp11::to_string(gemm_native_desc.rhs_info.n0));
        ref_cl_build_options.add_option("-DK0=" + support::cpp11::to_string(gemm_native_desc.rhs_info.k0));
        ref_cl_build_options.add_option("-DPARTIAL_STORE_M0=" + support::cpp11::to_string(partial_store_m0));
        ref_cl_build_options.add_option("-DPARTIAL_STORE_N0=" + support::cpp11::to_string(partial_store_n0));
        // Manually add PostOps
        {
            ref_cl_build_options.add_option("-DOP=ADD_X_POS_1");
            ref_cl_build_options.add_option("-DP2_ELTWISE_ARG1_HEIGHT=" + support::cpp11::to_string(t_dst_info.dimension(1)));
            ref_cl_build_options.add_option("-DP2_ELTWISE_ARG1_WIDTH=" + support::cpp11::to_string(t_dst_info.dimension(0)));
        }
    }
    cl_code.build_options = ref_cl_build_options;
}
} // namespace

TEST_SUITE(CL)
TEST_SUITE(UNIT)
TEST_SUITE(DYNAMIC_FUSION)
TEST_SUITE(ClCompositeKernel)
TEST_SUITE(Validate)

TEST_CASE(MoveNet_SubGraph_1, framework::DatasetMode::ALL)
{
    /* Computation:
     * out = add(addend, gemm_native(lhs, rhs, bias)) (non-broadcast)
     */
    const auto data_type   = DataType::F32;
    const auto m           = 5U;
    const auto n           = 4U;
    const auto k           = 3U;
    const auto t_lhs_shape = TensorShape(k, m);
    const auto t_rhs_shape = TensorShape(n, k);
    const auto t_dst_shape = TensorShape(n, m);
    auto       t_lhs_info  = TensorInfo(t_lhs_shape, 1, data_type);
    auto       t_rhs_info  = TensorInfo(t_rhs_shape, 1, data_type);
    auto       t_bias_info = TensorInfo(TensorShape(), 1, DataType::F32);
    auto       t_dst_info  = TensorInfo(t_dst_shape, 1, data_type);

    const ClTensorDescriptor t_lhs_desc{ &t_lhs_info, 2 };
    const ClTensorDescriptor t_rhs_desc{ &t_rhs_info, 2 };
    const ClTensorDescriptor t_bias_desc{ &t_bias_info, 2 };
    const ClTensorDescriptor t_addend_desc{ &t_dst_info, 2 };
    const ClTensorDescriptor t_dst_desc{ &t_dst_info, 2 };

    ClKernelBlueprint bp;
    ArgumentID        tid_lhs;
    ArgumentID        tid_rhs;
    ArgumentID        tid_l0_bias = g_arg_placeholder;
    ArgumentID        tid_l1_addend;
    ArgumentID        tid_dst;
    auto              st = add_tensor_argument(bp, t_lhs_desc, tid_lhs);
    st                   = add_tensor_argument(bp, t_rhs_desc, tid_rhs);
    st                   = add_tensor_argument(bp, t_addend_desc, tid_l1_addend);
    st                   = add_tensor_argument(bp, t_dst_desc, tid_dst);

    const auto                 common_kernel_desc = ClKernelComponentDescriptor{};
    const GemmNativeDescriptor gemm_native_desc{ 1.0, 1.0, m, n, k };
    const GEMMKernelInfo       gemm_info{ m, n, k, 0, false, false, false, false, ActivationLayerInfo{}, 1, 1, gemm_native_desc.lhs_info, gemm_native_desc.rhs_info, 0, 0 };
    const EltwiseAddDescriptor eltwise_add_desc{ ConvertPolicy::WRAP };
    const TileDescriptor       store_tile_info{};

    ArgumentID tid_acc;
    st = add_tensor_intermed(bp, tid_acc);
    st = add_kcomp_gemm_native(bp, common_kernel_desc, gemm_native_desc, tid_lhs, tid_rhs, tid_l0_bias, tid_acc);
    st = add_kcomp_eltwise_add(bp, common_kernel_desc, EltwiseAddDescriptor{}, tid_l1_addend, tid_acc, tid_acc);
    st = add_kcomp_store(bp, common_kernel_desc, tid_acc, tid_dst, StoreType::StoreBlockBoundaryAware);

    ClKernelCode cl_code;

    st = set_tile_info(bp, store_tile_info);
    st = build(cl_code, ClCodeBuilderContext{ GpuInfo{ GPUTarget::G71 } }, bp);
    set_build_options(cl_code, gemm_native_desc, t_lhs_info, t_rhs_info, nullptr, t_dst_info);

    ClExecutionDescriptor exec_desc;
    st = tune_static(exec_desc, cl_code);

    CLScheduler::get().default_init();
    ClCompositeKernel kernel;
    kernel.configure(CLKernelLibrary::get().get_compile_context(), cl_code);

    // Construct tensors
    CLTensor t_lhs{};
    CLTensor t_rhs{};
    CLTensor t_l1_addend{};
    CLTensor t_dst{};
    // Init tensors
    {
        t_lhs.allocator()->init(t_lhs_info);
        t_rhs.allocator()->init(t_rhs_info);
        t_l1_addend.allocator()->init(t_dst_info);
        t_dst.allocator()->init(t_dst_info);
    }
    // "Pack" tensors
    TensorBinding tensors({ { tid_lhs, &t_lhs },
        { tid_rhs, &t_rhs },
        { tid_l1_addend, &t_l1_addend },
        { tid_dst, &t_dst }
    });
    // Allocate and fill tensors
    {
        t_lhs.allocator()->allocate();
        t_rhs.allocator()->allocate();
        t_l1_addend.allocator()->allocate();
        t_dst.allocator()->allocate();
        fill<float>(CLAccessor(t_lhs), 0);
        fill<float>(CLAccessor(t_rhs), 1);
        fill<float>(CLAccessor(t_l1_addend), 2);
    }

    CLScheduler::get().enqueue_op(kernel, tensors, exec_desc, true);

    // Create reference
    SimpleTensor<float> ref_t_lhs{ t_lhs_shape, data_type, 1 };
    SimpleTensor<float> ref_t_rhs{ t_rhs_shape, data_type, 1 };
    SimpleTensor<float> ref_t_bias_placeholder{ t_dst_shape, data_type, 1 };
    SimpleTensor<float> ref_t_l1_addend{ t_dst_shape, data_type, 1 };

    // Fill reference
    fill<float>(ref_t_lhs, 0);
    fill<float>(ref_t_rhs, 1);
    fill<float>(ref_t_l1_addend, 2);
    const auto ref_t_dst = reference::arithmetic_operation(
                               ArithmeticOperation::ADD,
                               ref_t_l1_addend,
                               reference::gemm(ref_t_lhs, ref_t_rhs, ref_t_bias_placeholder, gemm_native_desc.alpha, 0.f /* To disable bias */),
                               data_type,
                               eltwise_add_desc.convert_policy);

    RelativeTolerance<float> tolerance_f32(0.001f); /**< Tolerance value for comparing reference's output against implementation's output for floating point data types */
    validate(CLAccessor(t_dst), ref_t_dst, tolerance_f32);
}

TEST_SUITE_END() // Validate

TEST_SUITE(Benchmark)
TEST_CASE(MoveNet_SubGraph_1, framework::DatasetMode::ALL)
{
    using std::chrono::duration_cast;
    using std::chrono::microseconds;
    const int num_iterations = 200;
    std::map<std::string, std::chrono::microseconds> measurements;
    /* Computation:
     * out = add(addend, gemm_native(lhs, rhs, bias))
     */
    const auto         data_type     = DataType::F32;
    const unsigned int m             = 12 * 12;
    const unsigned int n             = 64;
    const unsigned int k             = 384;
    const auto         t_lhs_shape   = TensorShape(k, m);
    const auto         t_rhs_shape   = TensorShape(n, k);
    const auto         t_dst_shape   = TensorShape(n, m);
    auto               t_lhs_info    = TensorInfo(t_lhs_shape, 1, data_type);
    auto               t_rhs_info    = TensorInfo(t_rhs_shape, 1, data_type);
    auto               t_bias_info   = TensorInfo(TensorShape(), 1, data_type);
    auto               t_l0_dst_info = TensorInfo(t_dst_shape, 1, data_type); // Intermediate tensor for cond3
    auto               t_l1_rhs_info = TensorInfo(t_dst_shape, 1, data_type);
    auto               t_dst_info    = TensorInfo(t_dst_shape, 1, data_type);

    const auto                 common_kernel_desc = ClKernelComponentDescriptor{};
    const GemmNativeDescriptor gemm_native_desc{ 1.0, 0.0, m, n, k };
    const GEMMKernelInfo       gemm_info{ m, n, k, 0, false, false, false, false, ActivationLayerInfo{}, 1, 1, gemm_native_desc.lhs_info, gemm_native_desc.rhs_info, 0, 0 };
    const EltwiseAddDescriptor eltwise_add_desc{ ConvertPolicy::WRAP };
    const TileDescriptor       store_tile_info{};

    // Create reference
    SimpleTensor<float> ref_t_lhs{ t_lhs_shape, data_type, 1 };
    SimpleTensor<float> ref_t_rhs{ t_rhs_shape, data_type, 1 };
    SimpleTensor<float> ref_t_bias_placeholder{ t_dst_shape, data_type, 1 };
    SimpleTensor<float> ref_t_l1_addend{ t_dst_shape, data_type, 1 };

    // Fill reference
    fill<float>(ref_t_lhs, 0);
    fill<float>(ref_t_rhs, 1);
    fill<float>(ref_t_l1_addend, 2);
    const auto ref_t_dst = reference::arithmetic_operation(
                               ArithmeticOperation::ADD,
                               ref_t_l1_addend,
                               reference::gemm(ref_t_lhs, ref_t_rhs, ref_t_bias_placeholder, gemm_native_desc.alpha, 0.f /* To disable bias */),
                               data_type,
                               eltwise_add_desc.convert_policy);

    CLScheduler::get().default_init();

    /* Condition 0: Dynamic Fused Kernel */
    CLTensor cond0_t_dst{};
    {
        TICK(cond0_0_startup_time);

        ClKernelBlueprint bp;
        ArgumentID        tid_lhs;
        ArgumentID        tid_rhs;
        ArgumentID        tid_l0_bias = g_arg_placeholder;
        ArgumentID        tid_l1_addend;
        ArgumentID        tid_dst;

        const ClTensorDescriptor t_lhs_desc{ &t_lhs_info, 2 };
        const ClTensorDescriptor t_rhs_desc{ &t_rhs_info, 2 };
        const ClTensorDescriptor t_bias_desc{ &t_bias_info, 2 };
        const ClTensorDescriptor t_addend_desc{ &t_dst_info, 2 };
        const ClTensorDescriptor t_dst_desc{ &t_dst_info, 2 };

        ClKernelCode cl_code;
        TICK(cond0_build_time)
        auto st = add_tensor_argument(bp, t_lhs_desc, tid_lhs);
        st      = add_tensor_argument(bp, t_rhs_desc, tid_rhs);
        st      = add_tensor_argument(bp, t_addend_desc, tid_l1_addend);
        st      = add_tensor_argument(bp, t_dst_desc, tid_dst);

        ArgumentID tid_acc;
        st = add_tensor_intermed(bp, tid_acc);
        st = add_kcomp_gemm_native(bp, common_kernel_desc, gemm_native_desc, tid_lhs, tid_rhs, tid_l0_bias, tid_acc);

        st = add_kcomp_eltwise_add(bp, common_kernel_desc, EltwiseAddDescriptor{}, tid_l1_addend, tid_acc, tid_acc);

        st = add_kcomp_store(bp, common_kernel_desc, tid_acc, tid_dst, StoreType::StoreBlockBoundaryAware);

        st = set_tile_info(bp, store_tile_info);
        st = build(cl_code, ClCodeBuilderContext{ GpuInfo{ GPUTarget::G71 } }, bp);
        set_build_options(cl_code, gemm_native_desc, t_lhs_info, t_rhs_info, nullptr, t_dst_info);
        TOCK(cond0_build_time, measurements)

        TICK(cond0_tune_time)
        ClExecutionDescriptor exec_desc;
        st = tune_static(exec_desc, cl_code);
        TOCK(cond0_tune_time, measurements)

        TICK(cond0_configure_time)
        ClCompositeKernel kernel;
        kernel.configure(CLKernelLibrary::get().get_compile_context(), cl_code);
        TOCK(cond0_configure_time, measurements)

        // Construct tensors
        CLTensor t_lhs{};
        CLTensor t_rhs{};
        CLTensor t_l1_addend{};

        // Init tensors
        {
            t_lhs.allocator()->init(t_lhs_info);
            t_rhs.allocator()->init(t_rhs_info);
            t_l1_addend.allocator()->init(t_dst_info);
            cond0_t_dst.allocator()->init(t_dst_info);
        }
        // Allocate tensors
        {
            t_lhs.allocator()->allocate();
            t_rhs.allocator()->allocate();
            t_l1_addend.allocator()->allocate();
            cond0_t_dst.allocator()->allocate();
            fill<float>(CLAccessor(t_lhs), 0);
            fill<float>(CLAccessor(t_rhs), 1);
            fill<float>(CLAccessor(t_l1_addend), 2);
        }

        // "Pack" tensors
        TensorBinding tensors({ { tid_lhs, &t_lhs }, { tid_rhs, &t_rhs }, { tid_l1_addend, &t_l1_addend }, { tid_dst, &cond0_t_dst } });

        CLScheduler::get().enqueue_op(kernel, tensors, exec_desc, true);
        CLScheduler::get().sync();
        TOCK(cond0_0_startup_time, measurements)

        TICK(cond0_1_latency)
        for(int i = 0; i < num_iterations; ++i)
        {
            CLScheduler::get().enqueue_op(kernel, tensors, exec_desc, true);
        }
        CLScheduler::get().sync();
        TOCK_AVG(cond0_1_latency, measurements, num_iterations)
    }
    /* Condition 1: Dynamic Unfused Kernel */
    /* Condition 2: Static Fused Kernel (current) */
    CLTensor cond2_t_dst{};
    {
        TICK(cond2_0_startup_time);
        arm_compute::opencl::kernels::ClGemmMatrixMultiplyNativeKernel l0_gemm_mm;

        TICK(cond2_configure_time);
        experimental::PostOpList<ITensorInfo *> post_ops;
        post_ops.push_back_op<experimental::PostOpEltwiseAdd<ITensorInfo *>>(&t_dst_info, 1, eltwise_add_desc.convert_policy);
        GEMMKernelInfo gemm_info{ m, n, k, 0, false, false, false, false, ActivationLayerInfo{}, 1, 1, gemm_native_desc.lhs_info, gemm_native_desc.rhs_info, 0, 0, post_ops };
        l0_gemm_mm.configure(CLKernelLibrary::get().get_compile_context(), &t_lhs_info, &t_rhs_info, nullptr, &t_dst_info, gemm_native_desc.alpha, gemm_native_desc.beta, gemm_native_desc.lhs_info,
                             gemm_native_desc.rhs_info, gemm_info);
        TOCK(cond2_configure_time, measurements);

        // Construct tensors
        CLTensor t_lhs{};
        CLTensor t_rhs{};
        CLTensor t_l1_addend{};

        // Init tensors
        {
            t_lhs.allocator()->init(t_lhs_info);
            t_rhs.allocator()->init(t_rhs_info);
            t_l1_addend.allocator()->init(t_dst_info);
            cond2_t_dst.allocator()->init(t_dst_info);
        }
        // Allocate tensors
        {
            t_lhs.allocator()->allocate();
            t_rhs.allocator()->allocate();
            t_l1_addend.allocator()->allocate();
            cond2_t_dst.allocator()->allocate();
            fill<float>(CLAccessor(t_lhs), 0);
            fill<float>(CLAccessor(t_rhs), 1);
            fill<float>(CLAccessor(t_l1_addend), 2);
        }

        // "Pack" tensors
        ITensorPack tensors
        {
            { ACL_SRC_0, &t_lhs },
            { ACL_SRC_1, &t_rhs },
            { EXPERIMENTAL_ACL_POST_OP_ARG_FIRST, &t_l1_addend },
            { ACL_DST, &cond2_t_dst },
        };
        CLScheduler::get().enqueue_op(l0_gemm_mm, tensors, true);
        CLScheduler::get().sync();
        TOCK(cond2_0_startup_time, measurements);

        TICK(cond2_1_latency);
        for(int i = 0; i < num_iterations; ++i)
        {
            CLScheduler::get().enqueue_op(l0_gemm_mm, tensors, true);
        }
        CLScheduler::get().sync();
        TOCK_AVG(cond2_1_latency, measurements, num_iterations);
    }
    /* Condition 3: Static Unfused Kernel (current) */
    CLTensor cond3_t_dst{};
    {
        TICK(cond3_0_startup_time);
        arm_compute::opencl::kernels::ClGemmMatrixMultiplyNativeKernel l0_gemm_mm;
        arm_compute::opencl::kernels::ClSaturatedArithmeticKernel      l1_add;

        TICK(cond3_configure_time);
        GEMMKernelInfo gemm_info{ m, n, k, 0, false, false, false, false, ActivationLayerInfo{}, 1, 1, gemm_native_desc.lhs_info, gemm_native_desc.rhs_info, 0, 0 };
        l0_gemm_mm.configure(CLKernelLibrary::get().get_compile_context(), &t_lhs_info, &t_rhs_info, nullptr, &t_l0_dst_info, gemm_native_desc.alpha, gemm_native_desc.beta, gemm_native_desc.lhs_info,
                             gemm_native_desc.rhs_info, gemm_info);
        l1_add.configure(CLKernelLibrary::get().get_compile_context(), ArithmeticOperation::ADD, &t_l0_dst_info, &t_l1_rhs_info, &t_dst_info, eltwise_add_desc.convert_policy);
        TOCK(cond3_configure_time, measurements);

        // Construct tensors
        CLTensor t_lhs{};
        CLTensor t_rhs{};
        CLTensor t_l0_dst{};
        CLTensor t_l1_addend{};

        // Init tensors
        {
            t_lhs.allocator()->init(t_lhs_info);
            t_rhs.allocator()->init(t_rhs_info);
            t_l0_dst.allocator()->init(t_l0_dst_info);
            t_l1_addend.allocator()->init(t_dst_info);
            cond3_t_dst.allocator()->init(t_dst_info);
        }
        // Allocate tensors
        {
            t_lhs.allocator()->allocate();
            t_rhs.allocator()->allocate();
            t_l0_dst.allocator()->allocate();
            t_l1_addend.allocator()->allocate();
            cond3_t_dst.allocator()->allocate();
            fill<float>(CLAccessor(t_lhs), 0);
            fill<float>(CLAccessor(t_rhs), 1);
            fill<float>(CLAccessor(t_l1_addend), 2);
        }

        // "Pack" tensors
        ITensorPack tensors_l0
        {
            { ACL_SRC_0, &t_lhs },
            { ACL_SRC_1, &t_rhs },
            { ACL_DST, &t_l0_dst },
        };
        ITensorPack tensors_l1
        {
            { ACL_SRC_0, &t_l0_dst },
            { ACL_SRC_1, &t_l1_addend },
            { ACL_DST, &cond3_t_dst },
        };
        CLScheduler::get().enqueue_op(l0_gemm_mm, tensors_l0, true);
        CLScheduler::get().enqueue_op(l1_add, tensors_l1, true);
        CLScheduler::get().sync();
        TOCK(cond3_0_startup_time, measurements);

        TICK(cond3_1_latency);
        for(int i = 0; i < num_iterations; ++i)
        {
            CLScheduler::get().enqueue_op(l0_gemm_mm, tensors_l0, true);
            CLScheduler::get().enqueue_op(l1_add, tensors_l1, true);
        }
        CLScheduler::get().sync();
        TOCK_AVG(cond3_1_latency, measurements, num_iterations);
    }

    RelativeTolerance<float> tolerance_f32(0.001f); /**< Tolerance value for comparing reference's output against implementation's output for floating point data types */
    std::cout << "cond0 validation: " << std::endl;
    validate(CLAccessor(cond0_t_dst), ref_t_dst, tolerance_f32);
    std::cout << "cond2 validation: " << std::endl;
    validate(CLAccessor(cond2_t_dst), ref_t_dst, tolerance_f32);
    std::cout << "cond3 validation: " << std::endl;
    validate(CLAccessor(cond3_t_dst), ref_t_dst, tolerance_f32);

    /* Report */
    std::cout << "Performance comparison (gemm native + add)" << std::endl;
    std::cout << "cond0: dynamic fusion module" << std::endl;
    std::cout << "cond2: static fused with post ops" << std::endl;
    std::cout << "cond3: static unfused" << std::endl;
    for(auto m : measurements)
    {
        std::cout << m.first << ": " << m.second.count() << "us" << std::endl;
    }
}
TEST_SUITE_END() // Benchmark
TEST_SUITE_END() // ClCompositeKernel
TEST_SUITE_END() // DYNAMIC_FUSION
TEST_SUITE_END() // UNIT
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute

#endif // defined(ENABLE_EXPERIMENTAL_DYNAMIC_FUSION)