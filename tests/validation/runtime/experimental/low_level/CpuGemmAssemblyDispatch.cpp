/*
 * Copyright (c) 2017-2025 Arm Limited.
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
#include "arm_compute/runtime/experimental/low_level/CpuGemmAssemblyDispatch.h"

#include "src/core/helpers/MemoryHelpers.h"
#include "src/cpu/operators/internal/CpuGemmAssemblyDispatch.h"
#include "tests/datasets/DatatypeDataset.h"
#include "tests/datasets/LargeGEMMDataset.h"
#include "tests/datasets/SmallGEMMDataset.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/framework/Macros.h"
#include "tests/NEON/Accessor.h"
#include "tests/validation/fixtures/CpuGemmAssemblyDispatchFixture.h"
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

} // namespace

TEST_SUITE(NEON)
TEST_SUITE(LOW_LEVEL)
TEST_SUITE(CpuGemmAssemblyDispatch)

/** Test case for memory injection in @ref experimental::op::ll::CpuGemmAssemblyDispatch.
 *
 * Configure the operator once and inject memory at run-time in multiple executions.
 *
 * Checks performed in order:
 * - Both runs compute the same output
 */
TEST_CASE(MemoryInjection, framework::DatasetMode::ALL)
{
    auto       gemm      = std::make_unique<experimental::op::ll::CpuGemmAssemblyDispatch>();
    const auto lhs_info  = TensorInfo(TensorShape(3U, 3U), 1, DataType::F32);
    const auto rhs_info  = TensorInfo(TensorShape(4U, 3U), 1, DataType::F32);
    const auto c_info    = TensorInfo(TensorShape(4U, 3U), 1, DataType::F32);
    auto       dst_info  = TensorInfo(TensorShape(4U, 3U), 1, DataType::F32);
    const auto gemm_info = GEMMInfo{};
    gemm->configure(&lhs_info, &rhs_info, &c_info, &dst_info, gemm_info);

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
        ARM_COMPUTE_EXPECT(reinterpret_cast<float *>(result_0.buffer())[i] ==
                               reinterpret_cast<float *>(result_1.buffer())[i],
                           framework::LogLevel::ERRORS);
    }
}

/** Test case for memory injection in @ref experimental::op::ll::CpuGemmAssemblyDispatch.
 *
 * Make sure @ref experimental::op::ll::CpuGemmAssemblyDispatch still works through injecting the memory at configure time using the old API.
 *
 * Checks performed in order:
 * - Both runs compute the same output
 */
TEST_CASE(MultipleExecutionWithConfigure, framework::DatasetMode::ALL)
{
    auto       gemm      = std::make_unique<experimental::op::ll::CpuGemmAssemblyDispatch>();
    const auto lhs_info  = TensorInfo(TensorShape(3U, 3U), 1, DataType::F32);
    const auto rhs_info  = TensorInfo(TensorShape(4U, 3U), 1, DataType::F32);
    const auto c_info    = TensorInfo(TensorShape(4U, 3U), 1, DataType::F32);
    auto       dst_info  = TensorInfo(TensorShape(4U, 3U), 1, DataType::F32);
    const auto gemm_info = GEMMInfo{};
    auto       run_conv  = [&]()
    {
        Tensor lhs = create_tensor<Tensor>(lhs_info);
        Tensor rhs = create_tensor<Tensor>(rhs_info);
        Tensor c   = create_tensor<Tensor>(c_info);
        Tensor dst = create_tensor<Tensor>(dst_info);
        gemm->configure(&lhs_info, &rhs_info, &c_info, &dst_info, gemm_info);
        lhs.allocator()->allocate();
        rhs.allocator()->allocate();
        c.allocator()->allocate();
        dst.allocator()->allocate();
        library->fill_tensor_value(Accessor(lhs), 1.f);
        library->fill_tensor_value(Accessor(rhs), 2.f);
        library->fill_tensor_value(Accessor(c), 3.f);

        ITensorPack run_pack{{TensorType::ACL_SRC_0, &lhs},
                             {TensorType::ACL_SRC_1, &rhs},
                             {TensorType::ACL_SRC_2, &c},
                             {TensorType::ACL_DST_0, &dst}};
        ITensorPack prep_pack{{TensorType::ACL_SRC_1, &rhs}, {TensorType::ACL_SRC_2, &c}};
        auto        mg = MemoryGroup{};
        auto        ws = manage_workspace<Tensor>(gemm->workspace(), mg, run_pack, prep_pack);

        gemm->prepare(prep_pack);
        gemm->run(run_pack);
        lhs.allocator()->free();
        rhs.allocator()->free();
        c.allocator()->free();

        return dst;
    };
    auto result_0 = run_conv();
    auto result_1 = run_conv();
    for (size_t i = 0; i < result_0.info()->tensor_shape().total_size(); ++i)
    {
        ARM_COMPUTE_EXPECT((reinterpret_cast<float *>(result_0.buffer()))[i] ==
                               (reinterpret_cast<float *>(result_1.buffer()))[i],
                           framework::LogLevel::ERRORS);
    };
}

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(ValidateAllDataTypes,
               framework::DatasetMode::ALL,
               combine(
                    datasets::AllDataTypes("DataType"),
                    datasets::AllDataTypes("DataType"),
                    datasets::AllDataTypes("DataType"),
                    make("fixed_format", {true, false})),
               lhs_data_type, rhs_data_type, output_data_type, fixed_format)
{
    auto gemm_info = GEMMInfo();
    auto asm_info = arm_compute::cpu::AsmGemmInfo();
    auto lhs_info = TensorInfo(TensorShape(21U, 13U), 1, lhs_data_type);
    auto rhs_info = TensorInfo(TensorShape(33U, 21U), 1, rhs_data_type);
    auto output_info = TensorInfo(TensorShape(33U, 13U), 1, output_data_type);
    gemm_info.set_fixed_format(fixed_format);
    asm_info.fixed_format = fixed_format;

    if (fixed_format) {
        WeightFormat wf = WeightFormat::ANY;
        gemm_info.set_accumulate(false);
        asm_info.accumulate = false;
        gemm_info.set_weight_format(wf);
        asm_info.weight_format = wf;
        gemm_info.set_fast_math(rhs_data_type == DataType::BFLOAT16 && fixed_format);
        asm_info.fast_mode = rhs_data_type == DataType::BFLOAT16 && fixed_format;

        experimental::op::ll::CpuGemmAssemblyDispatch::has_opt_impl(wf, &lhs_info, &rhs_info, nullptr, &output_info, gemm_info);
        gemm_info.set_weight_format(wf);
        asm_info.weight_format = wf;
        rhs_info.set_data_layout(DataLayout::NCHW);
    }

    const auto supports = {
        std::make_tuple(DataType::F32, DataType::F32, DataType::F32),
        std::make_tuple(DataType::F16, DataType::F16, DataType::F32),
        std::make_tuple(DataType::F16, DataType::F16, DataType::F16),
        std::make_tuple(DataType::BFLOAT16, DataType::BFLOAT16, DataType::BFLOAT16),
        std::make_tuple(DataType::BFLOAT16, DataType::BFLOAT16, DataType::F32),
        std::make_tuple(DataType::F32, DataType::BFLOAT16, DataType::F32),
    };
    const auto config = std::make_tuple(lhs_data_type, rhs_data_type, output_data_type);

    bool expected = arm_compute::cpu::CpuGemmAssemblyDispatch::validate(&lhs_info.clone()->set_is_resizable(true), &rhs_info.clone()->set_is_resizable(true), nullptr, &output_info.clone()->set_is_resizable(true), asm_info) &&
                    (std::find(supports.begin(), supports.end(), config) != supports.end());

    bool is_valid = bool(experimental::op::ll::CpuGemmAssemblyDispatch::validate(&lhs_info.clone()->set_is_resizable(true), &rhs_info.clone()->set_is_resizable(true), nullptr, &output_info.clone()->set_is_resizable(true), gemm_info));
    ARM_COMPUTE_EXPECT(is_valid == expected, framework::LogLevel::ERRORS);
}

template <typename T>
using CpuGemmAssemblyDispatchFixture = CpuGemmAssemblyDispatchValidationFixture<Tensor, Accessor, experimental::op::ll::CpuGemmAssemblyDispatch, T>;

template <typename T>
using CpuGemmDstF32AssemblyDispatchFixture = CpuGemmDstF32AssemblyDispatchValidationFixture<Tensor, Accessor, experimental::op::ll::CpuGemmAssemblyDispatch, T>;

#ifdef ARM_COMPUTE_ENABLE_FIXED_FORMAT_KERNELS
template <typename T>
using CpuGemmFixedFormatFixture = CpuGemmAssemblyDispatchFixedFormatFixture<Tensor, Accessor, experimental::op::ll::CpuGemmAssemblyDispatch, T>;
#endif // ARM_COMPUTE_ENABLE_FIXED_FORMAT_KERNELS

TEST_SUITE(Float)

DATA_TEST_CASE(ValidateAccumulate,
               framework::DatasetMode::ALL,
               combine(
                    make("In0",{ TensorShape(21U, 13U) }),
                    make("In1", { TensorShape(33U, 21U) }),
                    make("Dst", { TensorShape(33U, 13U) }),
                    make("Expected", { true })),
               shape_a, shape_b, shape_dst, expected)
{
    /* Accumulation test for GEMM kernels */
    // Create tensors
    TensorInfo in_a(shape_a, 1, DataType::F32);
    TensorInfo in_b(shape_b, 1, DataType::F32);
    TensorInfo dst(shape_dst, 1, DataType::F32);

    GEMMInfo gemm_info = GEMMInfo();
    gemm_info.set_accumulate(true);

    // Validate accumulation
    Status status = experimental::op::ll::CpuGemmAssemblyDispatch::validate(&in_a, &in_b, nullptr, &dst, gemm_info);
    ARM_COMPUTE_EXPECT((expected ==  bool(status)), framework::LogLevel::ERRORS);
}

#ifdef ARM_COMPUTE_ENABLE_FP16

TEST_SUITE(FP16)

FIXTURE_DATA_TEST_CASE(RunSmall,
                       CpuGemmAssemblyDispatchFixture<half>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(datasets::SmallGEMMDataset(),
                            make("DataType", DataType::F16),
                            make("Accumulate", false),
                            make("Pretranspose_B", {false, true}),
                            make("ActivationInfo", {
                            ActivationLayerInfo(),
                            ActivationLayerInfo(ActivationFunction::RELU),
                            ActivationLayerInfo(ActivationFunction::BOUNDED_RELU, 1.f),
                            ActivationLayerInfo(ActivationFunction::LU_BOUNDED_RELU, 1.f)
                        })))
{
    if(CPUInfo::get().has_fp16())
    {
        // Validate output
        validate(Accessor(_target), _reference, rel_tolerance_f16, tolerance_num, abs_tolerance_f16);
    }
    else
    {
        ARM_COMPUTE_TEST_INFO("Device does not support fp16 vector operations. Test SKIPPED.");
        framework::ARM_COMPUTE_PRINT_INFO();
    }
}
FIXTURE_DATA_TEST_CASE(RunLarge,
                       CpuGemmAssemblyDispatchFixture<half>,
                       framework::DatasetMode::NIGHTLY,
                       combine(datasets::LargeGEMMDataset(),
                            make("DataType", DataType::F16),
                            make("Accumulate", false),
                            make("Pretranspose_B", {false, true}),
                            make("ActivationInfo", {
                                ActivationLayerInfo(),
                                ActivationLayerInfo(ActivationFunction::RELU),
                                ActivationLayerInfo(ActivationFunction::BOUNDED_RELU, 1.f),
                                ActivationLayerInfo(ActivationFunction::LU_BOUNDED_RELU, 1.f)
                            })))
{
    if(CPUInfo::get().has_fp16())
    {
        // Validate output
        validate(Accessor(_target), _reference, rel_tolerance_f16, tolerance_num, abs_tolerance_f16);
    }
    else
    {
        ARM_COMPUTE_TEST_INFO("Device does not support fp16 vector operations. Test SKIPPED.");
        framework::ARM_COMPUTE_PRINT_INFO();
    }
}

TEST_SUITE(F32Dst)

FIXTURE_DATA_TEST_CASE(RunSmall,
                       CpuGemmDstF32AssemblyDispatchFixture<half>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(datasets::SmallGEMMDataset(),
                            make("data_type", DataType::F16),
                            make("Pretranspose_B", {false, true}),
                            make("ActivationInfo", {
                            ActivationLayerInfo(),
                            ActivationLayerInfo(ActivationFunction::RELU),
                            ActivationLayerInfo(ActivationFunction::BOUNDED_RELU, 1.f),
                            ActivationLayerInfo(ActivationFunction::LU_BOUNDED_RELU, 1.f)
                        })))
{
    if(CPUInfo::get().has_fp16() && CPUInfo::get().has_fhm())
    {
        // Validate output
        validate(Accessor(_target), _reference, tolerance_f);
    }
    else
    {
        ARM_COMPUTE_TEST_INFO("Device does not support fp16 or FHM vector operations. Test SKIPPED.");
        framework::ARM_COMPUTE_PRINT_INFO();
    }
}

FIXTURE_DATA_TEST_CASE(RunLarge,
                       CpuGemmDstF32AssemblyDispatchFixture<half>,
                       framework::DatasetMode::NIGHTLY,
                       combine(datasets::LargeGEMMDataset(),
                            make("data_type", DataType::F16),
                            make("Pretranspose_B", {false, true}),
                            make("ActivationInfo", {
                            ActivationLayerInfo()
                        })))
{
    if(CPUInfo::get().has_fp16() && CPUInfo::get().has_fhm())
    {
        // Validate output
        validate(Accessor(_target), _reference, tolerance_f);
    }
    else
    {
        ARM_COMPUTE_TEST_INFO("Device does not support fp16 or FHM vector operations. Test SKIPPED.");
        framework::ARM_COMPUTE_PRINT_INFO();
    }
}

TEST_SUITE_END() // F32Dst

TEST_SUITE_END() // FP16
#endif /* ARM_COMPUTE_ENABLE_FP16 */

TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       CpuGemmAssemblyDispatchFixture<float>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(datasets::SmallGEMMDataset(),
                            make("DataType", DataType::F32),
                            make("Accumulate", {false, true}),
                            make("Pretranspose_B", {false, true}),
                            make("ActivationInfo", {
                                ActivationLayerInfo(),
                                ActivationLayerInfo(ActivationFunction::RELU),
                                ActivationLayerInfo(ActivationFunction::BOUNDED_RELU, 1.f),
                                ActivationLayerInfo(ActivationFunction::LU_BOUNDED_RELU, 1.f)
                       })))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f);
}
FIXTURE_DATA_TEST_CASE(RunLarge,
                       CpuGemmAssemblyDispatchFixture<float>,
                       framework::DatasetMode::NIGHTLY,
                       combine(datasets::LargeGEMMDataset(),
                            make("DataType", DataType::F32),
                            make("Accumulate", {false, true}),
                            make("Pretranspose_B", {false, true}),
                            make("ActivationInfo", {
                                ActivationLayerInfo(),
                                ActivationLayerInfo(ActivationFunction::RELU),
                                ActivationLayerInfo(ActivationFunction::BOUNDED_RELU, 1.f),
                                ActivationLayerInfo(ActivationFunction::LU_BOUNDED_RELU, 1.f)
                       })))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f);
}


#ifdef ARM_COMPUTE_ENABLE_FIXED_FORMAT_KERNELS

TEST_SUITE(FIXED_FORMAT)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       CpuGemmFixedFormatFixture<float>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(
                            datasets::SmallGEMMDataset(),
                            make("DataType", DataType::F32)
                        ))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f);
}
FIXTURE_DATA_TEST_CASE(RunLarge,
                       CpuGemmFixedFormatFixture<float>,
                       framework::DatasetMode::NIGHTLY,
                       combine(
                            datasets::LargeGEMMDataset(),
                            make("DataType", DataType::F32)
                        ))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f);
}
TEST_SUITE_END() // FIXED_FORMAT
#endif // ARM_COMPUTE_FIXED_FORMAT_KERNELS

TEST_SUITE_END() // FP32
TEST_SUITE_END() // Float

TEST_SUITE_END() // CpuGemmAseemblyDispatch
TEST_SUITE_END() // LOW_LEVEL
TEST_SUITE_END() // NEON
} // namespace validation
} // namespace test
} // namespace arm_compute
