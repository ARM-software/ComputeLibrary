/*
 * Copyright (c) 2017-2021 Arm Limited.
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
#include "arm_compute/runtime/NEON/functions/NEGEMM.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "src/core/cpu/kernels/CpuGemmInterleave4x4Kernel.h"
#include "src/core/cpu/kernels/CpuGemmMatrixMultiplyKernel.h"
#include "src/core/cpu/kernels/CpuGemmTranspose1xWKernel.h"
#include "src/core/helpers/MemoryHelpers.h"
#include "src/runtime/cpu/operators/CpuGemm.h"
#include "tests/NEON/Accessor.h"
#include "tests/NEON/Helper.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/LargeGEMMDataset.h"
#include "tests/datasets/SmallGEMMDataset.h"
#include "tests/datasets/TinyGEMMDataset.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/GEMMFixture.h"
#include "tests/validation/fixtures/GEMMInterleave4x4Fixture.h"
#include "tests/validation/fixtures/GEMMTranspose1xWFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
constexpr AbsoluteTolerance<float> tolerance_f(0.001f); /**< Tolerance value for comparing reference's output against implementation's output for FP32 data types */
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
RelativeTolerance<half_float::half> rel_tolerance_f16(half(0.2)); /**< Relative tolerance value for comparing reference's output against implementation's output for FP16 data types */
const AbsoluteTolerance<float>      abs_tolerance_f16(0.2f);      /**< Absolute tolerance value for comparing reference's output against implementation's output for FP16 data types */
constexpr float                     tolerance_num = 0.07f;        /**< Tolerance number for FP16 data types */
#endif                                                            /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
/** CNN data types */
const auto CNNDataTypes = framework::dataset::make("DataType",
{
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    DataType::F16,
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
    DataType::F32,
});

const auto data_interleave = framework::dataset::make("M", 8, 12) * framework::dataset::make("N", 8, 12);
const auto data_transpose  = framework::dataset::make("M", 8, 14) * framework::dataset::make("N", 7, 14);

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

    ITensorPack run_pack{ { TensorType::ACL_SRC_0, &lhs }, { TensorType::ACL_SRC_1, &rhs }, { TensorType::ACL_SRC_2, &c } };
    ITensorPack prep_pack{ { TensorType::ACL_SRC_1, &rhs }, { TensorType::ACL_SRC_2, &c } };

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
    for(size_t i = 0; i < result_0.info()->tensor_shape().total_size(); ++i)
    {
        ARM_COMPUTE_EXPECT(((float *)result_0.buffer())[i] == ((float *)result_1.buffer())[i], framework::LogLevel::ERRORS);
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
    for(size_t i = 0; i < result_0.info()->tensor_shape().total_size(); ++i)
    {
        ARM_COMPUTE_EXPECT(((float *)result_0.buffer())[i] == ((float *)result_1.buffer())[i], framework::LogLevel::ERRORS);
    }
}

TEST_SUITE(TRANSPOSE_1XW)
using CpuGemmTranspose1xW = NESynthetizeFunctionWithZeroConstantKernelBorder<cpu::kernels::CpuGemmTranspose1xWKernel>;
DATA_TEST_CASE(ValidateZeroPadding, framework::DatasetMode::ALL, zip(
                   framework::dataset::make("N", { 1, 23, 63, 101 }),
                   framework::dataset::make("K", { 1, 47, 29, 27 })),
               n_value, k_value)
{
    bool status = validate_zero_padding<CpuGemmTranspose1xW>(n_value, k_value);
    ARM_COMPUTE_EXPECT(status, framework::LogLevel::ERRORS);
}

TEST_SUITE(U32)
using CpuGemmTranspose1xWFixture = GEMMTranspose1xWValidationFixture<Tensor, Accessor, CpuGemmTranspose1xW, uint32_t>;
FIXTURE_DATA_TEST_CASE(RunSmall, CpuGemmTranspose1xWFixture, framework::DatasetMode::PRECOMMIT, data_transpose * framework::dataset::make("DataType", DataType::U32))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // U32

TEST_SUITE(U16)
using CpuGemmTranspose1xWFixture = GEMMTranspose1xWValidationFixture<Tensor, Accessor, CpuGemmTranspose1xW, uint16_t>;
FIXTURE_DATA_TEST_CASE(RunSmall, CpuGemmTranspose1xWFixture, framework::DatasetMode::PRECOMMIT, data_transpose * framework::dataset::make("DataType", DataType::U16))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // U16

TEST_SUITE(U8)
using CpuGemmTranspose1xWFixture = GEMMTranspose1xWValidationFixture<Tensor, Accessor, CpuGemmTranspose1xW, uint8_t>;
FIXTURE_DATA_TEST_CASE(RunSmall, CpuGemmTranspose1xWFixture, framework::DatasetMode::PRECOMMIT, data_transpose * framework::dataset::make("DataType", DataType::U8))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // U8

TEST_SUITE_END() // TRANSPOSE_1XW

TEST_SUITE(INTERLEAVE_4X4)
using CpuGemmInterleave4x4 = NESynthetizeFunctionWithZeroConstantKernelBorder<cpu::kernels::CpuGemmInterleave4x4Kernel>;

DATA_TEST_CASE(ValidateZeroPadding, framework::DatasetMode::ALL, zip(
                   framework::dataset::make("M", { 1, 23, 63, 101 }),
                   framework::dataset::make("K", { 1, 47, 29, 27 })),
               m_value, k_value)
{
    bool status = validate_zero_padding<cpu::kernels::CpuGemmInterleave4x4Kernel>(m_value, k_value);
    ARM_COMPUTE_EXPECT(status, framework::LogLevel::ERRORS);
}

TEST_SUITE(U32)
using CpuGemmInterleave4x4Fixture = GEMMInterleave4x4ValidationFixture<Tensor, Accessor, CpuGemmInterleave4x4, uint32_t>;
FIXTURE_DATA_TEST_CASE(RunSmall, CpuGemmInterleave4x4Fixture, framework::DatasetMode::PRECOMMIT, data_interleave * framework::dataset::make("DataType", DataType::U32))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // U32

TEST_SUITE(U16)
using CpuGemmInterleave4x4Fixture = GEMMInterleave4x4ValidationFixture<Tensor, Accessor, CpuGemmInterleave4x4, uint16_t>;
FIXTURE_DATA_TEST_CASE(RunSmall, CpuGemmInterleave4x4Fixture, framework::DatasetMode::PRECOMMIT, data_interleave * framework::dataset::make("DataType", DataType::U16))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // U16

TEST_SUITE(U8)
using CpuGemmInterleave4x4Fixture = GEMMInterleave4x4ValidationFixture<Tensor, Accessor, CpuGemmInterleave4x4, uint8_t>;
FIXTURE_DATA_TEST_CASE(RunSmall, CpuGemmInterleave4x4Fixture, framework::DatasetMode::PRECOMMIT, data_interleave * framework::dataset::make("DataType", DataType::QASYMM8))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // U8

TEST_SUITE_END() // INTERLEAVE_4X4

template <typename T>
using NEGEMMFixture = GEMMValidationFixture<Tensor, Accessor, NEGEMM, T>;

template <typename T>
using NEGEMMFixtureDisabledC = GEMMValidationFixture<Tensor, Accessor, NEGEMM, T, true>;

TEST_SUITE(Float)
DATA_TEST_CASE(ValidateZeroPadding, framework::DatasetMode::ALL, zip(framework::dataset::make("In0", { TensorShape(21U, 13U),
                                                                                                       TensorShape(31U, 1U),
                                                                                                       TensorShape(31U, 1U),
                                                                                                       TensorShape(8U, 2U),
                                                                                                       TensorShape(38U, 12U),
                                                                                                       TensorShape(32U, 1U)
                                                                                                     }),
                                                                     framework::dataset::make("In1", { TensorShape(33U, 21U),
                                                                                                       TensorShape(23U, 31U),
                                                                                                       TensorShape(23U, 31U),
                                                                                                       TensorShape(16U, 8U),
                                                                                                       TensorShape(21U, 38U),
                                                                                                       TensorShape(17U, 32U)
                                                                                                     })),
               shape0, shape1)
{
    bool status = validate_gemm_zero_padding(shape0, shape1);
    ARM_COMPUTE_EXPECT(status, framework::LogLevel::ERRORS);
}

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, NEGEMMFixture<half>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SmallGEMMDataset(),
                                                                                                         framework::dataset::make("ReshapeWeights", { true, false })),
                                                                                                 framework::dataset::make("DataType", DataType::F16)))
{
    // Validate output
    validate(Accessor(_target), _reference, rel_tolerance_f16, tolerance_num, abs_tolerance_f16);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEGEMMFixture<half>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::LargeGEMMDataset(),
                                                                                                       framework::dataset::make("ReshapeWeights", { true, false })),

                                                                                               framework::dataset::make("DataType", DataType::F16)))
{
    // Validate output
    validate(Accessor(_target), _reference, rel_tolerance_f16, tolerance_num, abs_tolerance_f16);
}
TEST_SUITE_END()
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */

TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, NEGEMMFixture<float>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SmallGEMMDataset(),
                                                                                                          framework::dataset::make("ReshapeWeights", { true, false })),

                                                                                                  framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEGEMMFixture<float>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::LargeGEMMDataset(),
                                                                                                        framework::dataset::make("ReshapeWeights", { true, false })),

                                                                                                framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f);
}
TEST_SUITE(DisabledC)
FIXTURE_DATA_TEST_CASE(RunSmall, NEGEMMFixtureDisabledC<float>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SmallGEMMDataset(),
                                                                                                                   framework::dataset::make("ReshapeWeights", { true, false })),

                                                                                                           framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f);
}
TEST_SUITE_END()

TEST_SUITE(BatchedGEMMDisabledC)
FIXTURE_DATA_TEST_CASE(RunSmall, NEGEMMFixtureDisabledC<float>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SmallBatchedGEMMDataset(),
                                                                                                                   framework::dataset::make("ReshapeWeights", { true, false })),

                                                                                                           framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f);
}
TEST_SUITE_END()

TEST_SUITE_END()
TEST_SUITE_END()

TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
