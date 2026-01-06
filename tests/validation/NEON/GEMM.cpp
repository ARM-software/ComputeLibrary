/*
 * Copyright (c) 2017-2026 Arm Limited.
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
#include "arm_compute/core/Types.h"
#include "arm_compute/core/utils/StringUtils.h"
#include "arm_compute/runtime/NEON/functions/NEGEMM.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"

#include "src/core/helpers/MemoryHelpers.h"
#include "src/cpu/kernels/CpuGemmInterleave4x4Kernel.h"
#include "src/cpu/kernels/CpuGemmMatrixMultiplyKernel.h"
#include "src/cpu/kernels/CpuGemmTranspose1xWKernel.h"
#include "src/cpu/operators/CpuDynamicGemm.h"
#include "src/cpu/operators/CpuGemm.h"
#include "tests/datasets/LargeGEMMDataset.h"
#include "tests/datasets/SmallGEMMDataset.h"
#include "tests/datasets/TinyGEMMDataset.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/framework/Macros.h"
#include "tests/NEON/Accessor.h"
#include "tests/NEON/Helper.h"
#include "tests/PaddingCalculator.h"
#include "tests/validation/fixtures/GEMMFixture.h"
#include "tests/validation/fixtures/GEMMInterleave4x4Fixture.h"
#include "tests/validation/fixtures/GEMMTranspose1xWFixture.h"
#include "tests/validation/Validation.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
using framework::dataset::make;

namespace
{
constexpr AbsoluteTolerance<float> tolerance_f(
    0.001f); /**< Tolerance value for comparing reference's output against implementation's output for FP32 data types */
#ifdef ARM_COMPUTE_ENABLE_FP16
RelativeTolerance<half_float::half> rel_tolerance_f16(half(
    0.2)); /**< Relative tolerance value for comparing reference's output against implementation's output for FP16 data types */
const AbsoluteTolerance<float>      abs_tolerance_f16(
         0.2f); /**< Absolute tolerance value for comparing reference's output against implementation's output for FP16 data types */
constexpr float tolerance_num = 0.07f; /**< Tolerance number for FP16 data types */
#endif                                 /* ARM_COMPUTE_ENABLE_FP16 */
/** CNN data types */
const auto CNNDataTypes = make("DataType",
                               {
#ifdef ARM_COMPUTE_ENABLE_FP16
                                   DataType::F16,
#endif /* ARM_COMPUTE_ENABLE_FP16 */
                                   DataType::F32,
                               });

const auto data_interleave = make("M", 8, 12) * make("N", 8, 12);
const auto data_transpose  = make("M", 8, 14) * make("N", 7, 14);

/** Zero padding test */
template <typename FunctionType>
bool validate_zero_padding(unsigned int dim0_value, unsigned int dim1_value)
{
    const TensorShape in_shape(dim0_value, dim1_value);
    TensorInfo        in(in_shape, 1, DataType::U32);
    TensorInfo        dst;

    ARM_COMPUTE_EXPECT(in.is_resizable(), framework::LogLevel::ERRORS);

    // Validate zero-padding
    FunctionType func;

    func.configure(&in, &dst);

    return in.padding().empty();
}

/* Zero padding test for GEMM kernels */
bool validate_gemm_zero_padding(const TensorShape shape0, const TensorShape shape1)
{
    // Create tensors
    TensorInfo in0(shape0, 1, DataType::F32);
    TensorInfo in1(shape1, 1, DataType::F32);
    TensorInfo dst;

    // Validate zero-padding
    cpu::kernels::CpuGemmMatrixMultiplyKernel gemm;
    gemm.configure(&in0, &in1, &dst, 1.0, false);

    return in0.padding().empty() && in1.padding().empty() && dst.padding().empty();
}
} // namespace

TEST_SUITE(NEON)
TEST_SUITE(GEMM)

/** Test case for memory injection in @ref cpu::CpuGemm.
 *
 * Configure the operator once and inject memory at run-time in multiple executions.
 *
 * Checks performed in order:
 * - Both runs compute the same output
 */
TEST_CASE(MemoryInjection, framework::DatasetMode::ALL)
{
    auto       gemm      = std::make_unique<cpu::CpuGemm>();
    const auto lhs_info  = TensorInfo(TensorShape(3U, 3U), 1, DataType::F32);
    const auto rhs_info  = TensorInfo(TensorShape(4U, 3U), 1, DataType::F32);
    const auto c_info    = TensorInfo(TensorShape(4U, 3U), 1, DataType::F32);
    auto       dst_info  = TensorInfo(TensorShape(4U, 3U), 1, DataType::F32);
    const auto gemm_info = GEMMInfo{};
    gemm->configure(&lhs_info, &rhs_info, &c_info, &dst_info, 1.f, 1.f, gemm_info);

    // telhs are newly created every call of this lambda function
    auto lhs = create_tensor<Tensor>(lhs_info);
    auto rhs = create_tensor<Tensor>(rhs_info);
    auto c   = create_tensor<Tensor>(c_info);
    lhs.allocator()->allocate();
    rhs.allocator()->allocate();
    c.allocator()->allocate();

    ITensorPack run_pack{{TensorType::ACL_SRC_0, &lhs}, {TensorType::ACL_SRC_1, &rhs}, {TensorType::ACL_SRC_2, &c}};
    ITensorPack prep_pack{{TensorType::ACL_SRC_1, &rhs}, {TensorType::ACL_SRC_2, &c}};

    auto mg = MemoryGroup{};
    auto ws = manage_workspace<Tensor>(gemm->workspace(), mg, run_pack, prep_pack);

    auto run_conv = [&]() -> Tensor
    {
        auto dst = create_tensor<Tensor>(dst_info);
        dst.allocator()->allocate();
        run_pack.add_tensor(TensorType::ACL_DST, &dst);

        library->fill_tensor_value(Accessor(lhs), 1.f);
        library->fill_tensor_value(Accessor(rhs), 2.f);
        library->fill_tensor_value(Accessor(c), 3.f);
        // This operator is configured once and captured by this lambda.
        gemm->prepare(prep_pack);
        gemm->run(run_pack);
        return dst;
    };
    auto result_0 = run_conv();
    auto result_1 = run_conv();
    for (size_t i = 0; i < result_0.info()->tensor_shape().total_size(); ++i)
    {
        ARM_COMPUTE_EXPECT(((float *)result_0.buffer())[i] == ((float *)result_1.buffer())[i],
                           framework::LogLevel::ERRORS);
    }
}

/** Test case for memory injection in @ref NEGEMM.
 *
 * Make sure @ref NEGEMM still works through injecting the memory at configure time using the old API.
 *
 * Checks performed in order:
 * - Both runs compute the same output
 */
TEST_CASE(MultipleExecutionWithConfigure, framework::DatasetMode::ALL)
{
    auto       gemm      = std::make_unique<NEGEMM>();
    const auto lhs_info  = TensorInfo(TensorShape(3U, 3U), 1, DataType::F32);
    const auto rhs_info  = TensorInfo(TensorShape(4U, 3U), 1, DataType::F32);
    const auto c_info    = TensorInfo(TensorShape(4U, 3U), 1, DataType::F32);
    auto       dst_info  = TensorInfo(TensorShape(4U, 3U), 1, DataType::F32);
    const auto gemm_info = GEMMInfo{};
    auto       run_conv  = [&]()
    {
        auto lhs = create_tensor<Tensor>(lhs_info);
        auto rhs = create_tensor<Tensor>(rhs_info);
        auto c   = create_tensor<Tensor>(c_info);
        auto dst = create_tensor<Tensor>(dst_info);
        gemm->configure(&lhs, &rhs, &c, &dst, 1.f, 1.f, gemm_info);
        lhs.allocator()->allocate();
        rhs.allocator()->allocate();
        c.allocator()->allocate();
        dst.allocator()->allocate();
        library->fill_tensor_value(Accessor(lhs), 1.f);
        library->fill_tensor_value(Accessor(rhs), 2.f);
        library->fill_tensor_value(Accessor(c), 3.f);
        gemm->run();
        return dst;
    };
    auto result_0 = run_conv();
    auto result_1 = run_conv();
    for (size_t i = 0; i < result_0.info()->tensor_shape().total_size(); ++i)
    {
        ARM_COMPUTE_EXPECT(((float *)result_0.buffer())[i] == ((float *)result_1.buffer())[i],
                           framework::LogLevel::ERRORS);
    }
}

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(
               make("LhsInfo", { TensorInfo(TensorShape(27U, 13U), 1, DataType::S32), // Unsupported data type
                                                       TensorInfo(TensorShape(27U, 13U), 1, DataType::F32),
                                                     }),
               make("RhsInfo",{ TensorInfo(TensorShape(8U, 27U), 1, DataType::S32),
                                                        TensorInfo(TensorShape(8U, 27U), 1, DataType::F32),
                                                     }),
               make("OutputInfo",{ TensorInfo(TensorShape(8U, 13U), 1, DataType::S32),
                                                        TensorInfo(TensorShape(8U, 13U), 1, DataType::F32),
                                                     }),
               make("Expected", { false, true })
               ),
               lhs_info, rhs_info, output_info, expected)
{
    constexpr float alpha = 1.0;
    constexpr float beta = 0.0;
    const auto gemm_info = GEMMInfo();
    bool is_valid = bool(NEGEMM::validate(&lhs_info.clone()->set_is_resizable(true), &rhs_info.clone()->set_is_resizable(true), nullptr, &output_info.clone()->set_is_resizable(true), alpha, beta, gemm_info));
    ARM_COMPUTE_EXPECT(is_valid == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*
TEST_SUITE(KERNEL_SELECTION)
DATA_TEST_CASE(KernelSelection_mul_and_add,
               framework::DatasetMode::ALL,
               combine(make("CpuExt", std::string("NEON")), make("DataType", {DataType::F32, DataType::F16})),
               cpu_ext,
               data_type)
{
    using namespace cpu::kernels;

    cpuinfo::CpuIsaInfo cpu_isa{};
    cpu_isa.neon = (cpu_ext == "NEON");
    cpu_isa.fp16 = (data_type == DataType::F16);

    const auto *selected_impl_mul = CpuGemmMatrixMultiplyKernel::get_implementation(
        DataTypeISASelectorData{data_type, cpu_isa}, cpu::KernelSelectionType::Preferred);

    ARM_COMPUTE_ERROR_ON_NULLPTR(selected_impl_mul);

    std::string expected = lower_string(cpu_ext) + "_" + cpu_impl_dt(data_type) + "_gemm_matrix_mul";
    std::string actual   = selected_impl_mul->name;

    ARM_COMPUTE_EXPECT_EQUAL(expected, actual, framework::LogLevel::ERRORS);

    const auto *selected_impl_add = CpuGemmMatrixAdditionKernel::get_implementation(
        DataTypeISASelectorData{data_type, cpu_isa}, cpu::KernelSelectionType::Preferred);

    ARM_COMPUTE_ERROR_ON_NULLPTR(selected_impl_add);

    expected = lower_string(cpu_ext) + "_" + cpu_impl_dt(data_type) + "_gemm_matrix_add";
    actual   = selected_impl_add->name;

    ARM_COMPUTE_EXPECT_EQUAL(expected, actual, framework::LogLevel::ERRORS);
}
TEST_SUITE_END() // KERNEL_SELECTION

TEST_SUITE(TRANSPOSE_1XW)
using CpuGemmTranspose1xW = NESynthetizeFunctionWithZeroConstantKernelBorder<cpu::kernels::CpuGemmTranspose1xWKernel>;
DATA_TEST_CASE(ValidateZeroPadding,
               framework::DatasetMode::ALL,
               zip(make("N", {1, 23, 63, 101}), make("K", {1, 47, 29, 27})),
               n_value,
               k_value)
{
    bool status = validate_zero_padding<CpuGemmTranspose1xW>(n_value, k_value);
    ARM_COMPUTE_EXPECT(status, framework::LogLevel::ERRORS);
}

TEST_SUITE(U32)
using CpuGemmTranspose1xWFixture = GEMMTranspose1xWValidationFixture<Tensor, Accessor, CpuGemmTranspose1xW, uint32_t>;
FIXTURE_DATA_TEST_CASE(RunSmall,
                       CpuGemmTranspose1xWFixture,
                       framework::DatasetMode::PRECOMMIT,
                       data_transpose *make("DataType", DataType::U32))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // U32

TEST_SUITE(U16)
using CpuGemmTranspose1xWFixture = GEMMTranspose1xWValidationFixture<Tensor, Accessor, CpuGemmTranspose1xW, uint16_t>;
FIXTURE_DATA_TEST_CASE(RunSmall,
                       CpuGemmTranspose1xWFixture,
                       framework::DatasetMode::PRECOMMIT,
                       data_transpose *make("DataType", DataType::U16))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // U16

TEST_SUITE(U8)
using CpuGemmTranspose1xWFixture = GEMMTranspose1xWValidationFixture<Tensor, Accessor, CpuGemmTranspose1xW, uint8_t>;
FIXTURE_DATA_TEST_CASE(RunSmall,
                       CpuGemmTranspose1xWFixture,
                       framework::DatasetMode::PRECOMMIT,
                       data_transpose *make("DataType", DataType::U8))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // U8

TEST_SUITE_END() // TRANSPOSE_1XW

TEST_SUITE(INTERLEAVE_4X4)
using CpuGemmInterleave4x4 = NESynthetizeFunctionWithZeroConstantKernelBorder<cpu::kernels::CpuGemmInterleave4x4Kernel>;

DATA_TEST_CASE(ValidateZeroPadding,
               framework::DatasetMode::ALL,
               zip(make("M", {1, 23, 63, 101}), make("K", {1, 47, 29, 27})),
               m_value,
               k_value)
{
    bool status = validate_zero_padding<cpu::kernels::CpuGemmInterleave4x4Kernel>(m_value, k_value);
    ARM_COMPUTE_EXPECT(status, framework::LogLevel::ERRORS);
}

TEST_SUITE(U32)
using CpuGemmInterleave4x4Fixture =
    GEMMInterleave4x4ValidationFixture<Tensor, Accessor, CpuGemmInterleave4x4, uint32_t>;
FIXTURE_DATA_TEST_CASE(RunSmall,
                       CpuGemmInterleave4x4Fixture,
                       framework::DatasetMode::PRECOMMIT,
                       data_interleave *make("DataType", DataType::U32))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // U32

TEST_SUITE(U16)
using CpuGemmInterleave4x4Fixture =
    GEMMInterleave4x4ValidationFixture<Tensor, Accessor, CpuGemmInterleave4x4, uint16_t>;
FIXTURE_DATA_TEST_CASE(RunSmall,
                       CpuGemmInterleave4x4Fixture,
                       framework::DatasetMode::PRECOMMIT,
                       data_interleave *make("DataType", DataType::U16))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // U16

TEST_SUITE(U8)
using CpuGemmInterleave4x4Fixture = GEMMInterleave4x4ValidationFixture<Tensor, Accessor, CpuGemmInterleave4x4, uint8_t>;
FIXTURE_DATA_TEST_CASE(RunSmall,
                       CpuGemmInterleave4x4Fixture,
                       framework::DatasetMode::PRECOMMIT,
                       data_interleave *make("DataType", DataType::QASYMM8))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // U8

TEST_SUITE_END() // INTERLEAVE_4X4

template <typename T>
using NEGEMMFixture = GEMMValidationFixture<Tensor, Accessor, NEGEMM, T>;

#if defined(__aarch64__)
template <typename T>
using NEDynamicGEMMFixture = GEMMDynamicValidationFixture<Tensor, Accessor, NEGEMM, T>;

// Runs twice to exercise code paths with buffer reuse.
template <typename T>
using NEDynamicGEMMFixtureRunTwice =
    GEMMDynamicValidationFixture<Tensor, Accessor, NEGEMM, T, false, false, false, false, false, true>;
#endif // __aarch64__

template <typename T>
using NEBatchedMatMulFixture =
    GEMMValidationFixture<Tensor, Accessor, NEGEMM, T, true, false, false, false, false, true>;

template <typename T>
using NEGEMMAccumulateFixture = GEMMAccumulateValidationFixture<Tensor, Accessor, NEGEMM, T>;

TEST_SUITE(Float)
DATA_TEST_CASE(ValidateZeroPadding,
               framework::DatasetMode::ALL,
               zip(make("In0",
                        {TensorShape(21U, 13U), TensorShape(31U, 1U), TensorShape(31U, 1U), TensorShape(8U, 2U),
                         TensorShape(38U, 12U), TensorShape(32U, 1U)}),
                   make("In1",
                        {TensorShape(33U, 21U), TensorShape(23U, 31U), TensorShape(23U, 31U), TensorShape(16U, 8U),
                         TensorShape(21U, 38U), TensorShape(17U, 32U)})),
               shape0,
               shape1)
{
    bool status = validate_gemm_zero_padding(shape0, shape1);
    ARM_COMPUTE_EXPECT(status, framework::LogLevel::ERRORS);
}

DATA_TEST_CASE(ValidateAccumulate,
               framework::DatasetMode::ALL,
               combine(zip(make("In0", {TensorShape(21U, 13U)}),
                           make("In1", {TensorShape(33U, 21U)}),
                           make("Dst", {TensorShape(33U, 13U)})),
                       zip(make("alpha", {1.0, 100.0, 1.0, 1.0}),
                           make("beta", {0.0, 0.0, 1.0, 1.0}),
                           make("is_c_null", {false, false, false, true}),
                           make("Expected", {true, false, false, true}))),
               shape_a,
               shape_b,
               shape_dst,
               alpha,
               beta,
               is_c_null,
               expected)
{
    /* Accumulation test for GEMM kernels */
    // Create tensors
    TensorInfo in_a(shape_a, 1, DataType::F32);
    TensorInfo in_b(shape_b, 1, DataType::F32);
    TensorInfo in_c(shape_dst, 1, DataType::F32);
    TensorInfo dst(shape_dst, 1, DataType::F32);

    GEMMInfo gemm_info = GEMMInfo();
    gemm_info.set_accumulate(true);

    // Validate accumulation
    cpu::CpuGemm gemm;
    Status       status = gemm.validate(&in_a, &in_b, (is_c_null ? nullptr : &in_c), &dst, alpha, beta, gemm_info);
    ARM_COMPUTE_EXPECT((expected == bool(status)), framework::LogLevel::ERRORS);
}

#ifdef ARM_COMPUTE_ENABLE_FP16
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NEGEMMFixture<half>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(datasets::SmallGEMMDataset(),
                               make("ReshapeWeights", {true, false}),
                               make("DataType", DataType::F16)))
{
    if (CPUInfo::get().has_fp16())
    {
        // Validate output
        validate(Accessor(_target), _reference, rel_tolerance_f16, tolerance_num, abs_tolerance_f16);
    }
    else
    {
        ARM_COMPUTE_TEST_WARNING("Device does not support fp16 vector operations. Test SKIPPED.");
        framework::ARM_COMPUTE_PRINT_WARNING();
    }
}
FIXTURE_DATA_TEST_CASE(RunLarge,
                       NEGEMMFixture<half>,
                       framework::DatasetMode::NIGHTLY,
                       combine(datasets::LargeGEMMDataset(),
                               make("ReshapeWeights", {true, false}),
                               make("DataType", DataType::F16)))
{
    if (CPUInfo::get().has_fp16())
    {
        // Validate output
        validate(Accessor(_target), _reference, rel_tolerance_f16, tolerance_num, abs_tolerance_f16);
    }
    else
    {
        ARM_COMPUTE_TEST_WARNING("Device does not support fp16 vector operations. Test SKIPPED.");
        framework::ARM_COMPUTE_PRINT_WARNING();
    }
}

TEST_SUITE(BATCHED_MATMUL)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NEBatchedMatMulFixture<half>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(datasets::SmallBatchedMatMulDataset(),
                               make("ReshapeWeights", {false}),
                               make("DataType", DataType::F16)))
{
    if (CPUInfo::get().has_fp16())
    {
        // Validate output
        validate(Accessor(_target), _reference, rel_tolerance_f16, tolerance_num, abs_tolerance_f16);
    }
    else
    {
        ARM_COMPUTE_TEST_WARNING("Device does not support fp16 vector operations. Test SKIPPED.");
        framework::ARM_COMPUTE_PRINT_WARNING();
    }
}
TEST_SUITE_END() // BATCHED_MATMUL

TEST_SUITE_END() // FP16
#endif           /* ARM_COMPUTE_ENABLE_FP16 */

TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NEGEMMFixture<float>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(datasets::SmallGEMMDataset(),
                               make("ReshapeWeights", {true, false}),
                               make("DataType", DataType::F32)))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f);
}
FIXTURE_DATA_TEST_CASE(RunLarge,
                       NEGEMMFixture<float>,
                       framework::DatasetMode::NIGHTLY,
                       combine(datasets::LargeGEMMDataset(),
                               make("ReshapeWeights", {true, false}),
                               make("DataType", DataType::F32)))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f);
}

#if defined(__aarch64__)
TEST_SUITE(DynamicShape)
DATA_TEST_CASE(Validate,
               framework::DatasetMode::ALL,
               combine(zip(make("A", TensorShape{13U, 1U}),
                           make("B", TensorShape{21U, 13U}),
                           make("C", TensorShape{21U, 1U}),
                           make("D", TensorShape{21U, 1U})),
                       make("DataType", {DataType::U32, DataType::F32}),
                       make("Null_C", {true, false}),
                       make("Alpha", {0.0f, 0.5f, 1.0f}),
                       make("Beta", {0.0f, 0.5f, 1.0f}),
                       make("Dynamic_A", {true, false}),
                       make("Dynamic_B", {true, false}),
                       make("Dynamic_C", {true, false}),
                       make("Dynamic_D", {true, false}),
                       make("ReshapeFirstRun", {true, false})),
               shape_a,
               shape_b,
               shape_c,
               shape_d,
               data_type,
               null_c,
               alpha,
               beta,
               dynamic_a,
               dynamic_b,
               dynamic_c,
               dynamic_d,
               reshape_first_run)
{
    TensorInfo a{shape_a, 1, data_type};
    a.set_dynamic(dynamic_a);
    a.set_are_values_constant(!dynamic_a);
    TensorInfo b{shape_b, 1, DataType::F32};
    b.set_dynamic(dynamic_b);
    b.set_are_values_constant(!dynamic_b);
    TensorInfo c{shape_c, 1, DataType::F32};
    c.set_dynamic(dynamic_c);
    c.set_are_values_constant(!dynamic_c);
    TensorInfo d{shape_d, 1, DataType::F32};
    d.set_dynamic(dynamic_d);
    d.set_are_values_constant(!dynamic_d);

    GEMMInfo            gemm_info{false, false, reshape_first_run};
    cpu::CpuDynamicGemm gemm;
    Status              status = gemm.validate(&a, &b, null_c ? nullptr : &c, &d, alpha, beta, gemm_info);

    bool valid_data_type         = data_type == DataType::F32;
    bool valid_null_c            = !null_c;
    bool valid_alpha             = (alpha == 1.0f);
    bool valid_beta              = (beta == 1.0f);
    bool valid_dynamic_ab        = (dynamic_a || dynamic_b);
    bool valid_dynamic_d         = dynamic_d;
    bool valid_reshape_first_run = (!reshape_first_run || (!dynamic_b && !dynamic_c));
    bool validity                = valid_data_type && valid_null_c && valid_alpha && valid_beta && valid_dynamic_ab &&
                    valid_dynamic_d && valid_reshape_first_run;
    ARM_COMPUTE_EXPECT((validity == bool(status)), framework::LogLevel::ERRORS);
}
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NEDynamicGEMMFixture<float>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(datasets::SmallGEMMVectorBiasDataset(),
                               make("ReshapeWeights", {true, false}),
                               make("DataType", DataType::F32),
                               make("ConstantRHS", false)))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f);
}
FIXTURE_DATA_TEST_CASE(RunSmallConstantRHS,
                       NEDynamicGEMMFixtureRunTwice<float>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(datasets::SmallGEMMVectorBiasDataset(),
                               make("ReshapeWeights", false),
                               make("DataType", DataType::F32),
                               make("ConstantRHS", true)))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f);
}
FIXTURE_DATA_TEST_CASE(RunLarge,
                       NEDynamicGEMMFixture<float>,
                       framework::DatasetMode::NIGHTLY,
                       combine(datasets::LargeGEMMVectorBiasDataset(),
                               make("ReshapeWeights", {true, false}),
                               make("DataType", DataType::F32),
                               make("ConstantRHS", false)))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f);
}
TEST_SUITE_END() // DynamicShape
#endif           // __aarch64__

TEST_SUITE(BATCHED_MATMUL)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NEBatchedMatMulFixture<float>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(datasets::SmallBatchedMatMulDataset(),
                               make("ReshapeWeights", {false}),
                               make("DataType", DataType::F32)))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f);
}
TEST_SUITE_END() // BATCHED_MATMUL

TEST_SUITE(ACCUMULATE)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NEGEMMAccumulateFixture<float>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(datasets::SmallAccumulateGEMMDataset(),
                               make("ReshapeWeights", {false}),
                               make("DataType", DataType::F32)))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f);
}
FIXTURE_DATA_TEST_CASE(RunLarge,
                       NEGEMMAccumulateFixture<float>,
                       framework::DatasetMode::NIGHTLY,
                       combine(datasets::LargeAccumulateGEMMDataset(),
                               make("ReshapeWeights", {false}),
                               make("DataType", DataType::F32)))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f);
}
TEST_SUITE_END() // ACCUMULATE

TEST_SUITE_END() // FP32

TEST_SUITE_END() // Float
TEST_SUITE_END() // GEMM
TEST_SUITE_END() // NEON
} // namespace validation
} // namespace test
} // namespace arm_compute
