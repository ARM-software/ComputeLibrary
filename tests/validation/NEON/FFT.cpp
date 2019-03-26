/*
 * Copyright (c) 2019 ARM Limited.
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
#include "arm_compute/runtime/NEON/functions/NEFFT1D.h"
#include "arm_compute/runtime/NEON/functions/NEFFT2D.h"
#include "arm_compute/runtime/NEON/functions/NEFFTConvolutionLayer.h"
#include "arm_compute/runtime/Tensor.h"
#include "tests/NEON/Accessor.h"
#include "tests/datasets/SmallConvolutionLayerDataset.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/FFTFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
const auto data_types = framework::dataset::make("DataType", { DataType::F32 });
const auto shapes_1d  = framework::dataset::make("TensorShape", { TensorShape(2U, 2U, 3U), TensorShape(3U, 2U, 3U),
                                                                  TensorShape(4U, 2U, 3U), TensorShape(5U, 2U, 3U),
                                                                  TensorShape(7U, 2U, 3U), TensorShape(8U, 2U, 3U),
                                                                  TensorShape(9U, 2U, 3U), TensorShape(25U, 2U, 3U),
                                                                  TensorShape(49U, 2U, 3U), TensorShape(64U, 2U, 3U),
                                                                  TensorShape(16U, 2U, 3U), TensorShape(32U, 2U, 3U),
                                                                  TensorShape(96U, 2U, 2U)
                                                                });

const auto shapes_2d = framework::dataset::make("TensorShape", { TensorShape(2U, 2U, 3U), TensorShape(3U, 6U, 3U),
                                                                 TensorShape(4U, 5U, 3U), TensorShape(5U, 7U, 3U),
                                                                 TensorShape(7U, 25U, 3U), TensorShape(8U, 2U, 3U),
                                                                 TensorShape(9U, 16U, 3U), TensorShape(25U, 32U, 3U),
                                                                 TensorShape(192U, 128U, 2U)
                                                               });

const auto ActivationFunctionsSmallDataset = framework::dataset::make("ActivationInfo",
{
    ActivationLayerInfo(),
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, 0.5f)
});

RelativeTolerance<float> tolerance_f32(0.1f);   /**< Relative tolerance value for FP32 */
constexpr float          tolerance_num = 0.07f; /**< Tolerance number */

} // namespace
TEST_SUITE(NEON)
TEST_SUITE(FFT1D)

DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(shapes_1d, data_types),
               shape, data_type)
{
    // Create tensors
    Tensor src = create_tensor<Tensor>(shape, data_type, 2);
    Tensor dst = create_tensor<Tensor>(shape, data_type, 2);

    ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

    // Create and configure function
    NEFFT1D fft1d;
    fft1d.configure(&src, &dst, FFT1DInfo());

    // Validate valid region
    const ValidRegion valid_region = shape_to_valid_region(shape);
    validate(src.info()->valid_region(), valid_region);
    validate(dst.info()->valid_region(), valid_region);

    // Validate padding
    validate(src.info()->padding(), PaddingSize());
    validate(dst.info()->padding(), PaddingSize());
}

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(
        framework::dataset::make("InputInfo", { TensorInfo(TensorShape(32U, 13U, 2U), 2, DataType::F32), // Mismatching data types
                                                TensorInfo(TensorShape(32U, 13U, 2U), 2, DataType::F32), // Mismatching shapes
                                                TensorInfo(TensorShape(32U, 13U, 2U), 3, DataType::F32), // Invalid channels
                                                TensorInfo(TensorShape(32U, 13U, 2U), 2, DataType::F32), // Unsupported axis
                                                TensorInfo(TensorShape(11U, 13U, 2U), 2, DataType::F32), // Undecomposable FFT
                                                TensorInfo(TensorShape(25U, 13U, 2U), 2, DataType::F32),
        }),
        framework::dataset::make("OutputInfo",{ TensorInfo(TensorShape(32U, 13U, 2U), 2, DataType::F16),
                                                TensorInfo(TensorShape(16U, 13U, 2U), 2, DataType::F32),
                                                TensorInfo(TensorShape(32U, 13U, 2U), 2, DataType::F32),
                                                TensorInfo(TensorShape(32U, 13U, 2U), 2, DataType::F32),
                                                TensorInfo(TensorShape(11U, 13U, 2U), 2, DataType::F32),
                                                TensorInfo(TensorShape(25U, 13U, 2U), 2, DataType::F32),
        })),
        framework::dataset::make("Axis", { 0, 0, 0, 2, 0, 0 })),
        framework::dataset::make("Expected", { false, false, false, false, false, true })),
        input_info, output_info, axis, expected)
{
    FFT1DInfo desc;
    desc.axis = axis;
    const Status s = NEFFT1D::validate(&input_info.clone()->set_is_resizable(false), &output_info.clone()->set_is_resizable(false), desc);
    ARM_COMPUTE_EXPECT(bool(s) == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

template <typename T>
using NEFFT1DFixture = FFTValidationFixture<Tensor, Accessor, NEFFT1D, FFT1DInfo, T>;

TEST_SUITE(Float)
TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, NEFFT1DFixture<float>, framework::DatasetMode::ALL, combine(shapes_1d, framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f32, tolerance_num);
}
TEST_SUITE_END() // FP32
TEST_SUITE_END() // Float
TEST_SUITE_END() // FFT1D

TEST_SUITE(FFT2D)

DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(shapes_2d, data_types),
               shape, data_type)
{
    // Create tensors
    Tensor src = create_tensor<Tensor>(shape, data_type, 2);
    Tensor dst = create_tensor<Tensor>(shape, data_type, 2);

    ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

    // Create and configure function
    NEFFT2D fft2d;
    fft2d.configure(&src, &dst, FFT2DInfo());

    // Validate valid region
    const ValidRegion valid_region = shape_to_valid_region(shape);
    validate(src.info()->valid_region(), valid_region);
    validate(dst.info()->valid_region(), valid_region);

    // Validate padding
    validate(src.info()->padding(), PaddingSize());
    validate(dst.info()->padding(), PaddingSize());
}

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(
        framework::dataset::make("InputInfo", { TensorInfo(TensorShape(32U, 25U, 2U), 2, DataType::F32), // Mismatching data types
                                                TensorInfo(TensorShape(32U, 25U, 2U), 2, DataType::F32), // Mismatching shapes
                                                TensorInfo(TensorShape(32U, 25U, 2U), 3, DataType::F32), // Invalid channels
                                                TensorInfo(TensorShape(32U, 13U, 2U), 2, DataType::F32), // Undecomposable FFT
                                                TensorInfo(TensorShape(32U, 25U, 2U), 2, DataType::F32),
        }),
        framework::dataset::make("OutputInfo",{ TensorInfo(TensorShape(32U, 25U, 2U), 2, DataType::F16),
                                                TensorInfo(TensorShape(16U, 25U, 2U), 2, DataType::F32),
                                                TensorInfo(TensorShape(32U, 25U, 2U), 1, DataType::F32),
                                                TensorInfo(TensorShape(32U, 13U, 2U), 2, DataType::F32),
                                                TensorInfo(TensorShape(32U, 25U, 2U), 2, DataType::F32),
        })),
        framework::dataset::make("Expected", { false, false, false, false, true })),
               input_info, output_info, expected)
{
    const Status s = NEFFT2D::validate(&input_info.clone()->set_is_resizable(false), &output_info.clone()->set_is_resizable(false), FFT2DInfo());
    ARM_COMPUTE_EXPECT(bool(s) == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

template <typename T>
using NEFFT2DFixture = FFTValidationFixture<Tensor, Accessor, NEFFT2D, FFT2DInfo, T>;

TEST_SUITE(Float)
TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, NEFFT2DFixture<float>, framework::DatasetMode::ALL, combine(shapes_2d, framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f32, tolerance_num);
}
TEST_SUITE_END() // FP32
TEST_SUITE_END() // Float
TEST_SUITE_END() // FFT2D

TEST_SUITE(FFTConvolutionLayer)

template <typename T>
using NEFFTConvolutionLayerFixture = FFTConvolutionValidationFixture<Tensor, Accessor, NEFFTConvolutionLayer, T>;

TEST_SUITE(Float)
TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, NEFFTConvolutionLayerFixture<float>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::SmallFFTConvolutionLayerDataset(),
                                                                                                                 framework::dataset::make("DataType", DataType::F32)),
                                                                                                                 framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                                                                                                                 ActivationFunctionsSmallDataset))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f32, tolerance_num);
}
TEST_SUITE_END() // FP32
TEST_SUITE_END() // Float
TEST_SUITE_END() // FFTConvolutionLayer

TEST_SUITE_END() // NEON
} // namespace validation
} // namespace test
} // namespace arm_compute
