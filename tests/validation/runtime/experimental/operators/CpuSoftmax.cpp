/*
 * Copyright (c) 2017-2020, 2022-2026 Arm Limited.
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

#include "arm_compute/runtime/experimental/operators/CpuSoftmax.h"

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"

#include "src/common/cpuinfo/CpuIsaInfo.h"
#include "src/core/helpers/MemoryHelpers.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/framework/Macros.h"
#include "tests/NEON/Accessor.h"
#include "tests/validation/fixtures/CpuSoftmaxFixture.h"
#include "tests/validation/Validation.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
using framework::dataset::make;

/** Tolerance for float operations */
RelativeTolerance<half>              tolerance_f16(half(0.2));
constexpr AbsoluteTolerance<float>   tolerance_f32(0.000001f);
constexpr AbsoluteTolerance<int8_t>  tolerance_qasymm8_signed(1);
constexpr AbsoluteTolerance<uint8_t> tolerance_qasymm8(1);
} // namespace

TEST_SUITE(NEON)
TEST_SUITE(OPERATORS)
TEST_SUITE(CpuSoftmax)

// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(
    make("InputInfo", { TensorInfo(TensorShape(27U, 13U), 1, DataType::F32),    // Mismatching data types
                        TensorInfo(TensorShape(27U, 13U), 1, DataType::F32),    // Mismatching shapes
                        TensorInfo(TensorShape(27U, 13U), 1, DataType::QASYMM8, // Invalid output quantization info
                                    QuantizationInfo(1.f/256, 12)),
                        TensorInfo(TensorShape(32U, 13U), 1, DataType::F32),
                        TensorInfo(TensorShape(32U, 13U), 1, DataType::QASYMM8,
                                    QuantizationInfo(1.f/256, 12)),
                        TensorInfo(TensorShape(32U, 13U), 1, DataType::F32),
                        TensorInfo(TensorShape(32U, 13U), 1, DataType::QASYMM8,  //Invalid axis high
                                    QuantizationInfo(1.f/256, 12)),
                        TensorInfo(TensorShape(32U, 13U), 1, DataType::QASYMM8,  //Invalid axis low
                                    QuantizationInfo(1.f/256, 12)),
                        }),
    make("OutputInfo",{ TensorInfo(TensorShape(27U, 13U), 1, DataType::F16),
                        TensorInfo(TensorShape(27U, 11U), 1, DataType::F32),
                        TensorInfo(TensorShape(27U, 13U), 1, DataType::QASYMM8,
                                    QuantizationInfo(1.f/256, 12)),
                        TensorInfo(TensorShape(32U, 13U), 1, DataType::F32),
                        TensorInfo(TensorShape(32U, 13U), 1, DataType::QASYMM8,
                                    QuantizationInfo(1.f/256, 0)),
                        TensorInfo(TensorShape(32U, 13U), 1, DataType::F32),
                        TensorInfo(TensorShape(32U, 13U), 1, DataType::QASYMM8,
                                    QuantizationInfo(1.f/256, 0)),
                        TensorInfo(TensorShape(32U, 13U), 1, DataType::QASYMM8,
                                    QuantizationInfo(1.f/256, 0)),
                        }),
    make("beta", { 1.0,
                   2.0,
                   1.0,
                   2.0,
                   1.0,
                   1.0,
                   2.0,
                   1.0,
                }),
    make("axis", { 0,
                   0,
                   0,
                   1,
                   0,
                   -1,
                   2,
                   -3,
                }),
    make("Expected", { false, false, false, true, true, true, false, false })),
    input_info, output_info, beta, axis, expected)
{
    ARM_COMPUTE_EXPECT(bool(arm_compute::experimental::op::CpuSoftmax::validate(&input_info.clone()->set_is_resizable(false), &output_info.clone()->set_is_resizable(false), beta, axis)) == expected, framework::LogLevel::ERRORS);
}

TEST_CASE(OpCpuSoftmaxMemoryInjection, framework::DatasetMode::ALL)
{
    auto       softmax   = std::make_unique<arm_compute::experimental::op::CpuSoftmax>();
    const auto src_info  = TensorInfo(TensorShape{ 1U, 9U }, 1, DataType::F32);
    auto dst_info = TensorInfo(TensorShape{ 1U, 9U }, 1, DataType::F32);

    const float beta = (1.0F);
    const int32_t axis = 0;
    const bool is_log = false;

    softmax->configure(&src_info, &dst_info, beta, axis, is_log);

    // the lhs are newly created every call of this lambda function
    auto src = create_tensor<Tensor>(src_info);
    auto dst = create_tensor<Tensor>(dst_info);
    src.allocator()->allocate();

    ITensorPack run_pack{ { TensorType::ACL_SRC_0, &src }};
    auto mg = MemoryGroup{};
    auto ws = manage_workspace<Tensor>(softmax->workspace(), mg, run_pack);

    auto run_softmax = [&]() -> Tensor
    {
        auto dst = create_tensor<Tensor>(dst_info);
        dst.allocator()->allocate();
        run_pack.add_tensor(TensorType::ACL_DST, &dst);

        library->fill_tensor_value(Accessor(src), 1.f);
        // This operator is configured once and captured by this lambda.
        softmax->run(run_pack);
        return dst;
    };
    auto result_0 = run_softmax();
    auto result_1 = run_softmax();
    for(size_t i = 0; i < result_0.info()->tensor_shape().total_size(); ++i)
    {
        ARM_COMPUTE_EXPECT((reinterpret_cast<float *>(result_0.buffer()))[i] == (reinterpret_cast<float *>(result_1.buffer()))[i], framework::LogLevel::ERRORS);
    }
}

template <typename T>
using CpuOpSoftmaxFixture = CpuSoftmaxValidationFixture<Tensor, Accessor, arm_compute::experimental::op::CpuSoftmax, T>;

template <typename T>
using CpuSoftmaxThreadSafeFixture = CpuSoftmaxThreadSafeValidationFixture<Tensor, Accessor, arm_compute::experimental::op::CpuSoftmax, T>;

template <typename T>
using CpuSoftmaxQuantizedThreadSafeFixture = CpuSoftmaxQuantizedThreadSafeValidationFixture<Tensor, Accessor, arm_compute::experimental::op::CpuSoftmax, T>;

TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(SmokeTest, CpuOpSoftmaxFixture<float>, framework::DatasetMode::PRECOMMIT,
    combine(
        datasets::SoftmaxLayerSmallShapes(),
        make("DataType", DataType::F32),
        make("Beta", { 1.0f, 2.0f }),
        make("Axis", { 0, -1 })))
{
    // Validate output
    for(int i = 0; i < num_parallel_runs_; ++i)
    {
        validate(Accessor(target_[i]), reference_[i], tolerance_f32);
    }
}
TEST_SUITE_END() //FP32
#ifdef ARM_COMPUTE_ENABLE_FP16
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(SmokeTest, CpuOpSoftmaxFixture<half>, framework::DatasetMode::PRECOMMIT,
    combine(
        datasets::SoftmaxLayerSmallShapes(),
        make("DataType", DataType::F16),
        make("Beta", { 1.0f, 2.0f }),
        make("Axis", { 0, -1 })))
{
    if(CPUInfo::get().has_fp16())
    {
        // Validate output
        for(int i = 0; i < num_parallel_runs_; ++i)
        {
            validate(Accessor(target_[i]), reference_[i], tolerance_f16);
        }
    }
    else
    {
        ARM_COMPUTE_TEST_WARNING("Device does not support fp16. Test SKIPPED.");
        framework::ARM_COMPUTE_PRINT_WARNING();
    }
}
TEST_SUITE_END() //FP16
#endif           // ARM_COMPUTE_ENABLE_FP16

#ifndef BARE_METAL
TEST_SUITE(ThreadSafety)
TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(ConfigureOnceUseFromDifferentThreads, CpuSoftmaxThreadSafeFixture<float>,
                       framework::DatasetMode::PRECOMMIT,
    combine(
        datasets::SoftmaxLayerSmallShapes(),
        make("DataType", DataType::F32),
        make("Beta", { 1.0f, 2.0f }),
        make("Axis", { 0, -1 })))
{
    // Validate output
    for(int i = 0; i < num_parallel_runs_; ++i)
    {
        validate(Accessor(target_[i]), reference_[i], tolerance_f32);
    }
}
TEST_SUITE_END() //FP32

#ifdef ARM_COMPUTE_ENABLE_FP16
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(ConfigureOnceUseFromDifferentThreads, CpuSoftmaxThreadSafeFixture<half>,
                       framework::DatasetMode::ALL,
    combine(
        datasets::SoftmaxLayerSmallShapes(),
        make("DataType", DataType::F16),
        make("Beta", { 1.0f, 2.0f }),
        make("Axis", { 0, -1 })))
{
    if(CPUInfo::get().has_fp16())
    {
        // Validate output
        for(int i = 0; i < num_parallel_runs_; ++i)
        {
            validate(Accessor(target_[i]), reference_[i], tolerance_f16);
        }
    }
    else
    {
        ARM_COMPUTE_TEST_WARNING("Device does not support fp16. Test SKIPPED.");
        framework::ARM_COMPUTE_PRINT_WARNING();
    }
}
TEST_SUITE_END() //F16
#endif           // ARM_COMPUTE_ENABLE_FP16

TEST_SUITE(Quantized)
TEST_SUITE(QASYMM8_SIGNED)
FIXTURE_DATA_TEST_CASE(ConfigureOnceUseFromDifferentThreads, CpuSoftmaxQuantizedThreadSafeFixture<int8_t>, framework::DatasetMode::ALL,
    combine(
        datasets::SoftmaxLayerSmallShapes(),
        make("DataType", DataType::QASYMM8_SIGNED),
        make("Beta", { 1.0f, 2.0f }),
        make("Axis", { 0, -1 }),
        make("QuantizationInfo", {QuantizationInfo(0.5f, 10), QuantizationInfo(0.25f, 0)})
    ))
{
    // Validate output
    for(int i = 0; i < num_parallel_runs_; ++i)
    {
        validate(Accessor(target_[i]), reference_[i], tolerance_qasymm8_signed);
    }
}
TEST_SUITE_END() // QASYMM8_SIGNED
TEST_SUITE(QASYMM8)
FIXTURE_DATA_TEST_CASE(ConfigureOnceUseFromDifferentThreads, CpuSoftmaxQuantizedThreadSafeFixture<uint8_t>, framework::DatasetMode::ALL,
    combine(
        datasets::SoftmaxLayerSmallShapes(),
        make("DataType", DataType::QASYMM8),
        make("Beta", { 1.0f, 2.0f }),
        make("Axis", { 0, -1 }),
        make("QuantizationInfo", {QuantizationInfo(0.5f, 10), QuantizationInfo(0.25f, 0)})
    ))
{
    // Validate output
    for(int i = 0; i < num_parallel_runs_; ++i)
    {
        validate(Accessor(target_[i]), reference_[i], tolerance_qasymm8);
    }
}
TEST_SUITE_END() // QASYMM8
TEST_SUITE_END() // Quantized
TEST_SUITE_END() // ThreadSafety
#endif // #ifndef BARE_METAL
TEST_SUITE_END() //CpuSoftmax
TEST_SUITE_END() //OPERATORS
TEST_SUITE_END() //NEON

} // namespace validation
} // namespace test
} // namespace arm_compute
