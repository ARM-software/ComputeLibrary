/*
 * Copyright (c) 2023 Arm Limited.
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
#ifdef ACL_INTERNAL_TEST_CKW_IN_DF
#include "arm_compute/dynamic_fusion/sketch/gpu/operators/GpuPool2d.h"

#include "tests/CL/CLAccessor.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/datasets/dynamic_fusion/PoolingLayerDataset.h"
#include "tests/framework/Fixture.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/dynamic_fusion/gpu/cl/Pool2dFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
TEST_SUITE(CL)
TEST_SUITE(DYNAMIC_FUSION)
TEST_SUITE(POOL2D)

constexpr AbsoluteTolerance<float> tolerance_f32(0.001f); /**< Tolerance value for comparing reference's output against implementation's output for 32-bit floating-point type */
constexpr AbsoluteTolerance<float> tolerance_f16(0.01f);  /**< Tolerance value for comparing reference's output against implementation's output for 16-bit floating-point type */

const auto PoolingLayerDatasetFP = combine(combine(combine(combine(framework::dataset::make("PoolingType", { PoolingType::MAX, PoolingType::AVG }), framework::dataset::make("PoolingSize", { Size2D(2, 2), Size2D(3, 3) })),
                                                           framework::dataset::make("Pad", { Padding2D() })),
                                                   framework::dataset::make("Stride", { Size2D(1, 1), Size2D(2, 1), Size2D(5, 7) })),
                                           framework::dataset::make("ExcludePadding", { true }));

const auto pool_fp_mixed_precision_dataset = framework::dataset::make("FpMixedPrecision", { true, false });

template <typename T>
using DynamicFusionGpuPool2dFixture = DynamicFusionGpuPool2dValidationFixture<CLTensor, CLAccessor, GpuPool2d, T>;

template <typename T>
using DFSpecialGpuPool2dFixture = DynamicFusionGpuPool2dSpecialValidationFixture<CLTensor, CLAccessor, GpuPool2d, T>;

template <typename T>
using DFPoolMixedPrecisionFixture = DynamicFusionGpuPool2dMixedPrecisionValidationFixture<CLTensor, CLAccessor, GpuPool2d, T>;
// *INDENT-OFF*
// clang-format off

DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(
            framework::dataset::make("InputInfo", { TensorInfo(TensorShape(2U, 27U, 13U), 1, DataType::F32, DataLayout::NHWC),     // Mismatching data type
                                                    TensorInfo(TensorShape(2U, 27U, 13U), 1, DataType::F32, DataLayout::NHWC),     // Invalid pad/size combination
                                                    TensorInfo(TensorShape(2U, 27U, 13U), 1, DataType::F32, DataLayout::NHWC),     // Invalid pad/size combination
                                                    TensorInfo(TensorShape(2U, 27U, 13U), 1, DataType::QASYMM8, DataLayout::NHWC), // Invalid parameters, unsupported pooling
                                                    TensorInfo(TensorShape(5U, 15U, 13U), 1, DataType::F32, DataLayout::NHWC),     // Valid Non-rectangular Global Pooling
                                                    TensorInfo(TensorShape(5U, 13U, 13U), 1, DataType::F32, DataLayout::NHWC),     // Invalid output Global Pooling
                                                    TensorInfo(TensorShape(5U, 13U, 13U), 1, DataType::QASYMM8, DataLayout::NHWC), // Invalid - Quantized not supported.
                                                    TensorInfo(TensorShape(5U, 13U, 13U), 1, DataType::F32, DataLayout::NHWC),     // Valid global pooling
                                                    TensorInfo(TensorShape(13U, 13U, 5U), 1, DataType::F32, DataLayout::NCHW),     // Unsupported data layout
                                                }),
            framework::dataset::make("OutputInfo",{ TensorInfo(TensorShape(2U, 25U, 11U), 1, DataType::F16, DataLayout::NHWC),
                                                    TensorInfo(TensorShape(2U, 30U, 11U), 1, DataType::F32, DataLayout::NHWC),
                                                    TensorInfo(TensorShape(2U, 25U, 16U), 1, DataType::F32, DataLayout::NHWC),
                                                    TensorInfo(TensorShape(2U, 27U, 13U), 1, DataType::QASYMM8, DataLayout::NHWC),
                                                    TensorInfo(TensorShape(5U, 1U, 1U), 1, DataType::F32, DataLayout::NHWC),
                                                    TensorInfo(TensorShape(5U, 2U, 2U), 1, DataType::F32, DataLayout::NHWC),
                                                    TensorInfo(TensorShape(5U, 12U, 12U), 1, DataType::QASYMM8, DataLayout::NHWC),
                                                    TensorInfo(TensorShape(5U, 1U, 1U), 1, DataType::F32, DataLayout::NHWC),
                                                    TensorInfo(TensorShape(1U, 1U, 5U), 1, DataType::F32, DataLayout::NHWC),
                                                })),
            framework::dataset::make("Pool2dAttributes", {
                                                    Pool2dAttributes().pool_type(PoolingType::AVG).pool_size(Size2D(3,3)).pad(Padding2D(0,0,0,0)).stride(Size2D(1,1)),
                                                    Pool2dAttributes().pool_type(PoolingType::AVG).pool_size(Size2D(2,2)).pad(Padding2D(2,2,0,0)).stride(Size2D(1,1)),
                                                    Pool2dAttributes().pool_type(PoolingType::AVG).pool_size(Size2D(2,2)).pad(Padding2D(0,0,2,2)).stride(Size2D(1,1)),
                                                    Pool2dAttributes().pool_type(PoolingType::L2).pool_size(Size2D(3,3)).pad(Padding2D(0,0,0,0)).stride(Size2D(1,1)),
                                                    Pool2dAttributes().pool_type(PoolingType::AVG).pool_size(Size2D(15U, 13U)),
                                                    Pool2dAttributes().pool_type(PoolingType::MAX).pool_size(Size2D(13U, 13U)),
                                                    Pool2dAttributes().pool_type(PoolingType::AVG).pool_size(Size2D(2,2)).pad(Padding2D()).stride(Size2D(1,1)),
                                                    Pool2dAttributes().pool_type(PoolingType::AVG).pool_size(Size2D(13U,13U)),
                                                    Pool2dAttributes().pool_type(PoolingType::AVG).pool_size(Size2D(13U,13U)),
                                                })),
            framework::dataset::make("Expected", { false, false, false, false, true, false, false, true, false })),
            input_info, output_info, pool2d_attr, expected)
{
    // Create a new workload sketch
    auto              cl_compile_ctx = CLKernelLibrary::get().get_compile_context();
    auto              context        = GpuWorkloadContext{ &cl_compile_ctx };
    GpuWorkloadSketch sketch{ &context };

    // Declare GpuPool2d settings
    const GpuPool2dSettings &settings = GpuPool2dSettings().mixed_precision(false);

    // Validate Pool2d Configuration
    auto                   src_info    = context.create_tensor_info(input_info);
    auto                   dst_info    = context.create_tensor_info(output_info);
    bool                   res         = bool(GpuPool2d::validate_op(sketch, &src_info, pool2d_attr, settings));
    ARM_COMPUTE_EXPECT(res == expected, framework::LogLevel::ERRORS);
}

// clang-format on
// *INDENT-ON*

TEST_SUITE(Float)
TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, DynamicFusionGpuPool2dFixture<float>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SmallNoneUnitShapes(), PoolingLayerDatasetFP),
                                                                                                                  framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunLarge, DynamicFusionGpuPool2dFixture<float>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::LargeShapes(), PoolingLayerDatasetFP),
                                                                                                                framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunSpecial, DFSpecialGpuPool2dFixture<float>, framework::DatasetMode::ALL, combine(datasets::PoolingLayerDatasetSpecialDynamicFusion(),
                                                                                                          framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}

TEST_SUITE(GlobalPooling)
FIXTURE_DATA_TEST_CASE(RunSmall, DynamicFusionGpuPool2dFixture<float>, framework::DatasetMode::ALL,
                       combine(combine(combine(combine(combine(combine(
                                                                   framework::dataset::make("InputShape", { TensorShape(27U, 13U, 2U),
                                                                                                            TensorShape(27U, 13U, 2U, 4U)
                                                                                                          }),
                                                                   framework::dataset::make("PoolingType", { PoolingType::AVG, PoolingType::MAX })),
                                                               framework::dataset::make("PoolingSize", { Size2D(27, 13) })),
                                                       framework::dataset::make("Pad", { Padding2D() })),
                                               framework::dataset::make("Stride", { Size2D(1, 1) })),
                                       framework::dataset::make("ExcludePadding", true)),
                               framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}

FIXTURE_DATA_TEST_CASE(RunLarge, DynamicFusionGpuPool2dFixture<float>, framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(combine(combine(combine(
                                                                   framework::dataset::make("InputShape", { TensorShape(79U, 37U, 11U),
                                                                                                            TensorShape(79U, 37U, 11U, 4U)
                                                                                                          }),
                                                                   framework::dataset::make("PoolingType", { PoolingType::AVG, PoolingType::MAX })),
                                                               framework::dataset::make("PoolingSize", { Size2D(79, 37) })),
                                                       framework::dataset::make("Pad", { Padding2D() })),
                                               framework::dataset::make("Stride", { Size2D(1, 1) })),
                                       framework::dataset::make("ExcludePadding", true)),
                               framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
TEST_SUITE_END() // GlobalPooling
TEST_SUITE_END() // FP32

TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, DFPoolMixedPrecisionFixture<half>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::SmallNoneUnitShapes(), PoolingLayerDatasetFP),
                                                                                                                       framework::dataset::make("DataType", DataType::F16)),
                                                                                                               pool_fp_mixed_precision_dataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f16);
}
FIXTURE_DATA_TEST_CASE(RunLarge, DFPoolMixedPrecisionFixture<half>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::LargeShapes(), PoolingLayerDatasetFP),
                                                                                                                     framework::dataset::make("DataType", DataType::F16)),
                                                                                                             pool_fp_mixed_precision_dataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f16);
}

TEST_SUITE(GlobalPooling)
FIXTURE_DATA_TEST_CASE(RunSmall, DynamicFusionGpuPool2dFixture<half>, framework::DatasetMode::ALL,
                       combine(combine(combine(combine(combine(combine(
                                                                   framework::dataset::make("InputShape", { TensorShape(27U, 13U, 2U),
                                                                                                            TensorShape(27U, 13U, 2U, 4U)
                                                                                                          }),
                                                                   framework::dataset::make("PoolingType", { PoolingType::AVG, PoolingType::MAX })),
                                                               framework::dataset::make("PoolingSize", { Size2D(27, 13) })),
                                                       framework::dataset::make("Pad", { Padding2D() })),
                                               framework::dataset::make("Stride", { Size2D(1, 1) })),
                                       framework::dataset::make("ExcludePadding", true)),
                               framework::dataset::make("DataType", DataType::F16)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f16);
}

FIXTURE_DATA_TEST_CASE(RunLarge, DynamicFusionGpuPool2dFixture<half>, framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(combine(combine(combine(
                                                                   framework::dataset::make("InputShape", { TensorShape(79U, 37U, 11U),
                                                                                                            TensorShape(79U, 37U, 11U, 4U)
                                                                                                          }),
                                                                   framework::dataset::make("PoolingType", { PoolingType::AVG, PoolingType::MAX })),
                                                               framework::dataset::make("PoolingSize", { Size2D(79, 37) })),
                                                       framework::dataset::make("Pad", { Padding2D() })),
                                               framework::dataset::make("Stride", { Size2D(1, 1) })),
                                       framework::dataset::make("ExcludePadding", true)),
                               framework::dataset::make("DataType", DataType::F16)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f16);
}
TEST_SUITE_END() // GlobalPooling
TEST_SUITE_END() // FP16
TEST_SUITE_END() // FLOAT

TEST_SUITE_END() // POOL2D
TEST_SUITE_END() // DYNAMIC_FUSION
TEST_SUITE_END() // CL
}
}
}
#endif // ACL_INTERNAL_TEST_CKW_IN_DF
