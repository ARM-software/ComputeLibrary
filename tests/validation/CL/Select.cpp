/*
 * Copyright (c) 2018-2019 ARM Limited.
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
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/CLTensorAllocator.h"
#include "arm_compute/runtime/CL/functions/CLSelect.h"
#include "tests/CL/CLAccessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/SelectFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
auto run_small_dataset = combine(datasets::SmallShapes(), framework::dataset::make("has_same_rank", { false, true }));
auto run_large_dataset = combine(datasets::LargeShapes(), framework::dataset::make("has_same_rank", { false, true }));

} // namespace
TEST_SUITE(CL)
TEST_SUITE(Select)

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(zip(
        framework::dataset::make("CInfo", { TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::S8), // Invalid condition datatype
                                            TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8), // Invalid output datatype
                                            TensorInfo(TensorShape(13U), 1, DataType::U8),          // Invalid c shape
                                            TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8), // Mismatching shapes
                                            TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                            TensorInfo(TensorShape(2U), 1, DataType::U8),
        }),
        framework::dataset::make("XInfo",{ TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                           TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                           TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                           TensorInfo(TensorShape(32U, 10U, 2U), 1, DataType::F32),
                                           TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),
                                           TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),
        })),
        framework::dataset::make("YInfo",{ TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                           TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                           TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                           TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),
                                           TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),
                                           TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),
        })),
        framework::dataset::make("OutputInfo",{ TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                                TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::S8),
                                                TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                                TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),
                                                TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),
                                                TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),
        })),
        framework::dataset::make("Expected", { false, false, false, false, true, true})),
        c_info, x_info, y_info, output_info, expected)
{
    Status s = CLSelect::validate(&c_info.clone()->set_is_resizable(false),
                                  &x_info.clone()->set_is_resizable(false),
                                  &y_info.clone()->set_is_resizable(false),
                                  &output_info.clone()->set_is_resizable(false));
    ARM_COMPUTE_EXPECT(bool(s) == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

template <typename T>
using CLSelectFixture = SelectValidationFixture<CLTensor, CLAccessor, CLSelect, T>;

TEST_SUITE(Float)
TEST_SUITE(F16)
DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, run_small_dataset,
               shape, same_rank)
{
    const DataType dt = DataType::F16;

    // Create tensors
    CLTensor ref_c = create_tensor<CLTensor>(detail::select_condition_shape(shape, same_rank), DataType::U8);
    CLTensor ref_x = create_tensor<CLTensor>(shape, dt);
    CLTensor ref_y = create_tensor<CLTensor>(shape, dt);
    CLTensor dst   = create_tensor<CLTensor>(shape, dt);

    // Create and Configure function
    CLSelect select;
    select.configure(&ref_c, &ref_x, &ref_y, &dst);

    // Validate valid region
    const ValidRegion valid_region = shape_to_valid_region(shape);
    validate(dst.info()->valid_region(), valid_region);

    // Validate padding
    const int         step    = 16 / arm_compute::data_size_from_type(dt);
    const PaddingSize padding = PaddingCalculator(shape.x(), step).required_padding();
    if(same_rank)
    {
        validate(ref_c.info()->padding(), padding);
    }
    validate(ref_x.info()->padding(), padding);
    validate(ref_y.info()->padding(), padding);
    validate(dst.info()->padding(), padding);
}

FIXTURE_DATA_TEST_CASE(RunSmall,
                       CLSelectFixture<half>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(run_small_dataset, framework::dataset::make("DataType", DataType::F16)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge,
                       CLSelectFixture<half>,
                       framework::DatasetMode::NIGHTLY,
                       combine(run_large_dataset, framework::dataset::make("DataType", DataType::F16)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END() // F16

TEST_SUITE(FP32)
DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, run_small_dataset,
               shape, same_rank)
{
    const DataType dt = DataType::F32;

    // Create tensors
    CLTensor ref_c = create_tensor<CLTensor>(detail::select_condition_shape(shape, same_rank), DataType::U8);
    CLTensor ref_x = create_tensor<CLTensor>(shape, dt);
    CLTensor ref_y = create_tensor<CLTensor>(shape, dt);
    CLTensor dst   = create_tensor<CLTensor>(shape, dt);

    // Create and Configure function
    CLSelect select;
    select.configure(&ref_c, &ref_x, &ref_y, &dst);

    // Validate valid region
    const ValidRegion valid_region = shape_to_valid_region(shape);
    validate(dst.info()->valid_region(), valid_region);

    // Validate padding
    const int         step    = 16 / arm_compute::data_size_from_type(dt);
    const PaddingSize padding = PaddingCalculator(shape.x(), step).required_padding();
    if(same_rank)
    {
        validate(ref_c.info()->padding(), padding);
    }
    validate(ref_x.info()->padding(), padding);
    validate(ref_y.info()->padding(), padding);
    validate(dst.info()->padding(), padding);
}

FIXTURE_DATA_TEST_CASE(RunSmall,
                       CLSelectFixture<float>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(run_small_dataset, framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge,
                       CLSelectFixture<float>,
                       framework::DatasetMode::NIGHTLY,
                       combine(run_large_dataset, framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END() // F32
TEST_SUITE_END() // Float

TEST_SUITE(Quantized)
TEST_SUITE(QASYMM8)
DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, run_small_dataset,
               shape, same_rank)
{
    const DataType dt = DataType::QASYMM8;

    // Create tensors
    CLTensor ref_c = create_tensor<CLTensor>(detail::select_condition_shape(shape, same_rank), DataType::U8);
    CLTensor ref_x = create_tensor<CLTensor>(shape, dt);
    CLTensor ref_y = create_tensor<CLTensor>(shape, dt);
    CLTensor dst   = create_tensor<CLTensor>(shape, dt);

    // Create and Configure function
    CLSelect select;
    select.configure(&ref_c, &ref_x, &ref_y, &dst);

    // Validate valid region
    const ValidRegion valid_region = shape_to_valid_region(shape);
    validate(dst.info()->valid_region(), valid_region);

    // Validate padding
    const int         step    = 16 / arm_compute::data_size_from_type(dt);
    const PaddingSize padding = PaddingCalculator(shape.x(), step).required_padding();
    if(same_rank)
    {
        validate(ref_c.info()->padding(), padding);
    }
    validate(ref_x.info()->padding(), padding);
    validate(ref_y.info()->padding(), padding);
    validate(dst.info()->padding(), padding);
}

FIXTURE_DATA_TEST_CASE(RunSmall,
                       CLSelectFixture<uint8_t>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(run_small_dataset, framework::dataset::make("DataType", DataType::QASYMM8)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge,
                       CLSelectFixture<uint8_t>,
                       framework::DatasetMode::NIGHTLY,
                       combine(run_large_dataset, framework::dataset::make("DataType", DataType::QASYMM8)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END() // QASYMM8
TEST_SUITE_END() // Quantized

TEST_SUITE_END() // Select
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute