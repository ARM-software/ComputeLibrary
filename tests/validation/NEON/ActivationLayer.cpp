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
#include "arm_compute/Acl.hpp"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/utils/misc/Traits.h"
#include "arm_compute/core/utils/StringUtils.h"
#include "arm_compute/function_info/ActivationLayerInfo.h"
#include "arm_compute/runtime/NEON/functions/NEActivationLayer.h"
#include "arm_compute/runtime/RuntimeContext.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"

#include "src/common/cpuinfo/CpuIsaInfo.h"
#include "src/cpu/kernels/CpuActivationKernel.h"
#include "support/AclRequires.h"
#include "tests/datasets/ActivationFunctionsDataset.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/framework/Macros.h"
#include "tests/NEON/Accessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/validation/fixtures/ActivationLayerFixture.h"
#include "tests/validation/helpers/ActivationHelpers.h"
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

RelativeTolerance<float> tolerance_float_sqrt(0.0001f);

constexpr AbsoluteTolerance<int16_t> tolerance_qsymm16(1);

const auto NeonActivationFunctionsDataset =
    concat(datasets::ActivationFunctions(),
           make("ActivationFunction",
                {ActivationLayerInfo::ActivationFunction::HARD_SWISH, ActivationLayerInfo::ActivationFunction::SWISH}));

/** Input data sets. */
const auto ActivationDataset =
    combine(make("InPlace", {false, true}), NeonActivationFunctionsDataset, make("AlphaBeta", {0.5f, 1.f}));
const auto ActivationDatasetForPaddingAfterConfigure =
    combine(make("InPlace", {false, true}), NeonActivationFunctionsDataset, make("AlphaBeta", {0.5f}));

template <typename T, ARM_COMPUTE_REQUIRES_TA(arm_compute::utils::traits::is_floating_point<T>::value)>
void test_float_sqrt_boundary_value()
{
    constexpr auto vector_size = uint32_t{16};

    auto data_type = DataType::F32;
#ifdef ARM_COMPUTE_ENABLE_FP16
    data_type = std::is_same<T, half>::value ? DataType::F16 : data_type;
#endif /* ARM_COMPUTE_ENABLE_FP16 */

    if (data_type == DataType::F16 && !CPUInfo::get().has_fp16())
    {
        ARM_COMPUTE_TEST_WARNING("Device does not support fp16 vector operations. Test SKIPPED.");
        framework::ARM_COMPUTE_PRINT_WARNING();

        return;
    }

    const auto boundary_value_vector = std::vector<T>{
        std::numeric_limits<T>::min(),
        T(0),
        std::numeric_limits<T>::epsilon(),
        std::numeric_limits<T>::max(),
    };

    // the following size ensures that the whole logic (vector + left-over) to be tested
    // using all boundary values iff boundary_value_vecotr.size() is smaller than vector_size.
    auto shape = TensorShape{vector_size + boundary_value_vector.size()};
    auto info  = ActivationLayerInfo{ActivationLayerInfo::ActivationFunction::SQRT};
    auto src   = create_tensor<Tensor>(shape, data_type);

    auto act = NEActivationLayer{};
    act.configure(&src, nullptr, info);
    src.allocator()->allocate();
    library->fill_static_values(Accessor(src), boundary_value_vector);
    act.run();

    auto reference_src = SimpleTensor<T>{shape, data_type};
    library->fill_static_values(reference_src, boundary_value_vector);
    auto reference_dst = reference::activation_layer<T>(reference_src, info);

    validate(Accessor(src), reference_dst, tolerance_float_sqrt);
}
} // namespace

TEST_SUITE(NEON)
TEST_SUITE(ActivationLayer)

/** Test case for memory injection in @ref cpu::CpuWinogradConv2d.
 *
 * Configure the operator once and inject memory at run-time in multiple executions.
 *
 * Checks performed in order:
 * - Both runs compute the same output
 */
TEST_CASE(ActivationAPI, framework::DatasetMode::ALL)
{
    acl::StatusCode err = acl::StatusCode::Success;

    // Create context & Queue
    acl::Context ctx(acl::Target::Cpu, &err);
    ARM_COMPUTE_ASSERT(err == acl::StatusCode::Success);

    acl::Queue queue(ctx, &err);
    ARM_COMPUTE_ASSERT(err == acl::StatusCode::Success);

    // Create activation operator
    acl::TensorDescriptor src_info({2, 3}, acl::DataType::Float32);
    acl::TensorDescriptor dst_info({2, 3}, acl::DataType::Float32);
    acl::ActivationDesc   desc{AclRelu, 6.f, 0.f, false};

    acl::Activation act(ctx, src_info, dst_info, desc, &err);
    ARM_COMPUTE_ASSERT(err == acl::StatusCode::Success);

    // Create tensors and feed
    acl::Tensor src(ctx, src_info, &err);
    ARM_COMPUTE_ASSERT(err == acl::StatusCode::Success);
    acl::Tensor dst(ctx, dst_info, &err);
    ARM_COMPUTE_ASSERT(err == acl::StatusCode::Success);

    acl::TensorPack pack(ctx);
    err = pack.add(src, ACL_SRC);
    err = pack.add(dst, ACL_DST);
    ARM_COMPUTE_ASSERT(err == acl::StatusCode::Success);

    // Execute operator
    err = act.run(queue, pack);
    ARM_COMPUTE_ASSERT(err == acl::StatusCode::Success);
}

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(
    make("InputInfo", { TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32),     // Mismatching data types
                                            TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),
                                            TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32),     // Mismatching shapes
                                          }),
    make("OutputInfo",{ TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F16),
                                            TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),
                                            TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),
                                          }),
    make("ActivationInfo", { ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU),
                                                 ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU),
                                                 ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU),
                                               }),
    make("Expected", { false, true, false})
    ),
    input_info, output_info, act_info, expected)
{
    bool is_valid = bool(NEActivationLayer::validate(&input_info.clone()->set_is_resizable(false), &output_info.clone()->set_is_resizable(false), act_info));
    ARM_COMPUTE_EXPECT(is_valid == expected, framework::LogLevel::ERRORS);
}

// clang-format on
// *INDENT-ON*

template <typename T>
using NEActivationLayerFixture = ActivationValidationFixture<Tensor, Accessor, NEActivationLayer, T>;
template <typename T>
using NEActivationLayerWithPaddingFixture =
    ActivationWithPaddingValidationFixture<Tensor, Accessor, NEActivationLayer, T>;

TEST_SUITE(Float)
#ifdef ARM_COMPUTE_ENABLE_FP16
TEST_SUITE(FP16)
TEST_CASE(SqrtBoundaryValue, framework::DatasetMode::ALL)
{
    test_float_sqrt_boundary_value<half>();
}
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NEActivationLayerFixture<half>,
                       framework::DatasetMode::ALL,
                       combine(datasets::SmallShapes(), ActivationDataset, make("DataType", DataType::F16)))
{
    if (CPUInfo::get().has_fp16())
    {
        // Validate output
        validate(Accessor(_target), _reference, helper::relative_tolerance(_data_type, _function), 0.f,
                 helper::absolute_tolerance(_data_type, _function));
    }
    else
    {
        ARM_COMPUTE_TEST_WARNING("Device does not support fp16 vector operations. Test SKIPPED.");
        framework::ARM_COMPUTE_PRINT_WARNING();
    }
}

FIXTURE_DATA_TEST_CASE(PaddingAfterConfigure,
                       NEActivationLayerWithPaddingFixture<half>,
                       framework::DatasetMode::ALL,
                       combine(make("Shape", TensorShape{7U, 7U, 17U, 2U}),
                               ActivationDatasetForPaddingAfterConfigure,
                               make("DataType", DataType::F16)))
{
    if (CPUInfo::get().has_fp16())
    {
        // Validate output
        validate(Accessor(_target), _reference, helper::relative_tolerance(_data_type, _function), 0.f,
                 helper::absolute_tolerance(_data_type, _function));
    }
    else
    {
        ARM_COMPUTE_TEST_WARNING("Device does not support fp16 vector operations. Test SKIPPED.");
        framework::ARM_COMPUTE_PRINT_WARNING();
    }
}
TEST_SUITE_END() // FP16
#endif           /* ARM_COMPUTE_ENABLE_FP16 */

TEST_SUITE(FP32)
TEST_CASE(SqrtBoundaryValue, framework::DatasetMode::ALL)
{
    test_float_sqrt_boundary_value<float>();
}
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NEActivationLayerFixture<float>,
                       framework::DatasetMode::ALL,
                       combine(datasets::SmallShapes(), ActivationDataset, make("DataType", DataType::F32)))

{
    // Validate output
    validate(Accessor(_target), _reference, helper::relative_tolerance(_data_type, _function), 0.f,
             helper::absolute_tolerance(_data_type, _function));
}

FIXTURE_DATA_TEST_CASE(PaddingAfterConfigure,
                       NEActivationLayerWithPaddingFixture<float>,
                       framework::DatasetMode::ALL,
                       combine(make("Shape", TensorShape{7U, 7U, 17U, 2U}),
                               ActivationDatasetForPaddingAfterConfigure,
                               make("DataType", DataType::F32)))
{
    validate(Accessor(_target), _reference, helper::relative_tolerance(_data_type, _function), 0.f,
             helper::absolute_tolerance(_data_type, _function));
}
// Run only on SME Devices to stress Logistic SME kernel
#ifdef ARM_COMPUTE_ENABLE_SME2
TEST_SUITE(SME)
const auto LogisticDataset = combine(make("InPlace", {false}),
                                     make("Function", ActivationLayerInfo::ActivationFunction::LOGISTIC),
                                     make("AlphaBeta", {1.f}));
FIXTURE_DATA_TEST_CASE(RunLogistic5D,
                       NEActivationLayerFixture<float>,
                       framework::DatasetMode::ALL,
                       combine(datasets::Tiny5dShapes(), LogisticDataset, make("DataType", DataType::F32)))

{
    // Validate output
    validate(Accessor(_target), _reference, helper::relative_tolerance(_data_type, _function), 0.f,
             helper::absolute_tolerance(_data_type, _function));
}

FIXTURE_DATA_TEST_CASE(RunLogisticSME,
                       NEActivationLayerFixture<float>,
                       framework::DatasetMode::ALL,
                       combine(datasets::LogisticSMEStressShapesFp32(),
                               LogisticDataset,
                               make("DataType", DataType::F32)))

{
    // Validate output
    validate(Accessor(_target), _reference, helper::relative_tolerance(_data_type, _function), 0.f,
             helper::absolute_tolerance(_data_type, _function));
}
FIXTURE_DATA_TEST_CASE(PaddingAfterConfigure,
                       NEActivationLayerWithPaddingFixture<float>,
                       framework::DatasetMode::ALL,
                       combine(datasets::LogisticSMEStressShapesFp32(),
                               LogisticDataset,
                               make("DataType", DataType::F32)))

{
    // Validate output
    validate(Accessor(_target), _reference, helper::relative_tolerance(_data_type, _function), 0.f,
             helper::absolute_tolerance(_data_type, _function));
}
TEST_SUITE_END() // SME
#endif           // ARM_COMPUTE_ENABLE_SME2
TEST_SUITE_END() // FP32
TEST_SUITE_END() // Float

template <typename T>
using NEActivationLayerQuantizedFixture = ActivationValidationQuantizedFixture<Tensor, Accessor, NEActivationLayer, T>;
template <typename T>
using NEActivationLayerWithPaddingQuantizedFixture =
    ActivationWithPaddingValidationQuantizedFixture<Tensor, Accessor, NEActivationLayer, T>;

/** Input data sets. */
const auto QuantizedActivationFunctionsDataset = make("ActivationFunction",
                                                      {
                                                          ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU,
                                                          ActivationLayerInfo::ActivationFunction::RELU,
                                                          ActivationLayerInfo::ActivationFunction::BOUNDED_RELU,
                                                          ActivationLayerInfo::ActivationFunction::LOGISTIC,
                                                          ActivationLayerInfo::ActivationFunction::TANH,
                                                          ActivationLayerInfo::ActivationFunction::LEAKY_RELU,
#ifdef __aarch64__
                                                          ActivationLayerInfo::ActivationFunction::GELU,
#endif
                                                      });

const auto QuantizedActivationDataset =
    combine(make("InPlace", {false}),
            concat(QuantizedActivationFunctionsDataset,
                   make("ActivationFunction", ActivationLayerInfo::ActivationFunction::HARD_SWISH)),
            make("AlphaBeta", {0.5f, 1.f}));
const auto QuantizedActivationDatasetForPaddingAfterConfigure =
    combine(make("InPlace", {false}),
            concat(QuantizedActivationFunctionsDataset,
                   make("ActivationFunction", ActivationLayerInfo::ActivationFunction::HARD_SWISH)),
            make("AlphaBeta", {0.5f}));

TEST_SUITE(Quantized)
TEST_SUITE(QASYMM8)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NEActivationLayerQuantizedFixture<uint8_t>,
                       framework::DatasetMode::ALL,
                       combine(datasets::SmallShapes(),
                               QuantizedActivationDataset,
                               make("DataType", DataType::QASYMM8),
                               make("QuantizationInfo", {QuantizationInfo(0.1f, 128.0f)})))
{
    // Validate output
    validate(Accessor(_target), _reference, helper::tolerance_qasymm8(_function));
}
FIXTURE_DATA_TEST_CASE(PaddingAfterConfigure,
                       NEActivationLayerWithPaddingQuantizedFixture<uint8_t>,
                       framework::DatasetMode::ALL,
                       combine(make("Shape", TensorShape{7U, 7U, 17U, 2U}),
                               QuantizedActivationDatasetForPaddingAfterConfigure,
                               make("DataType", DataType::QASYMM8),
                               make("QuantizationInfo", {QuantizationInfo(0.1f, 128.0f)})))
{
    // Validate output
    validate(Accessor(_target), _reference, helper::tolerance_qasymm8(_function));
}
TEST_SUITE_END() // QASYMM8

TEST_SUITE(QASYMM8_SIGNED)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NEActivationLayerQuantizedFixture<int8_t>,
                       framework::DatasetMode::ALL,
                       combine(datasets::SmallShapes(),
                               QuantizedActivationDataset,
                               make("DataType", DataType::QASYMM8_SIGNED),
                               make("QuantizationInfo", {QuantizationInfo(0.5f, 10.0f)})))
{
    // Validate output
    validate(Accessor(_target), _reference, helper::tolerance_qasymm8(_function));
}
FIXTURE_DATA_TEST_CASE(PaddingAfterConfigure,
                       NEActivationLayerWithPaddingQuantizedFixture<int8_t>,
                       framework::DatasetMode::ALL,
                       combine(make("Shape", TensorShape{7U, 7U, 17U, 2U}),
                               QuantizedActivationDatasetForPaddingAfterConfigure,
                               make("DataType", DataType::QASYMM8_SIGNED),
                               make("QuantizationInfo", {QuantizationInfo(0.5f, 10.0f)})))
{
    // Validate output
    validate(Accessor(_target), _reference, helper::tolerance_qasymm8(_function));
}
TEST_SUITE_END() // QASYMM8_SIGNED

/** Input data sets. */
const auto Int16QuantizedActivationFunctionsDataset = make("ActivationFunction",
                                                           {
                                                               ActivationLayerInfo::ActivationFunction::LOGISTIC,
                                                               ActivationLayerInfo::ActivationFunction::TANH,
                                                               ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU,
                                                           });
const auto Int16QuantizedActivationDataset =
    combine(make("InPlace", {false}), Int16QuantizedActivationFunctionsDataset, make("AlphaBeta", {0.5f, 1.f}));

const auto Int16QuantizedActivationDatasetForPaddingAfterConfigure =
    combine(make("InPlace", {false}), Int16QuantizedActivationFunctionsDataset, make("AlphaBeta", {0.5f}));

TEST_SUITE(QSYMM16)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NEActivationLayerQuantizedFixture<int16_t>,
                       framework::DatasetMode::ALL,
                       combine(datasets::SmallShapes(),
                               Int16QuantizedActivationDataset,
                               make("DataType", DataType::QSYMM16),
                               make("QuantizationInfo", {QuantizationInfo(1.f / 32768.f, 0.f)})))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qsymm16);
}
FIXTURE_DATA_TEST_CASE(PaddingAfterConfigure,
                       NEActivationLayerWithPaddingQuantizedFixture<int16_t>,
                       framework::DatasetMode::ALL,
                       combine(make("Shape", TensorShape{7U, 7U, 17U, 2U}),
                               Int16QuantizedActivationDatasetForPaddingAfterConfigure,
                               make("DataType", DataType::QSYMM16),
                               make("QuantizationInfo", {QuantizationInfo(1.f / 32768.f, 0.f)})))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qsymm16);
}
TEST_SUITE_END() // QSYMM16
TEST_SUITE_END() // Quantized

TEST_SUITE_END() // ActivationLayer
TEST_SUITE_END() // Neon
} // namespace validation
} // namespace test
} // namespace arm_compute
