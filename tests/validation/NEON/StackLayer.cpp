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
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/runtime/NEON/functions/NEStackLayer.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "tests/NEON/Accessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/StackLayerFixture.h"

#include <vector>

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
// *INDENT-OFF*
// clang-format off
/** Data types */
const auto data_types = framework::dataset::make("DataType", { DataType::QASYMM8, DataType::F16, DataType::F32 });

/** Num tensors values to test */
const auto n_values = framework::dataset::make("NumTensors", { 3, 4 });

/** Shapes 1D to test */
const auto shapes_1d_small = combine(datasets::Small1DShapes(), framework::dataset::make("Axis", -1, 2));

/** Shapes 2D to test */
const auto shapes_2d_small = combine(datasets::Small2DShapes(), framework::dataset::make("Axis", -2, 3));

/** Shapes 3D to test */
const auto shapes_3d_small = combine(datasets::Small3DShapes(), framework::dataset::make("Axis", -3, 4));

/** Shapes 4D to test */
const auto shapes_4d_small = combine(datasets::Small4DShapes(), framework::dataset::make("Axis", -4, 5));

/** Shapes 1D to test */
const auto shapes_1d_large = combine(datasets::Large1DShapes(), framework::dataset::make("Axis", -1, 2));

/** Shapes 2D to test */
const auto shapes_2d_large = combine(datasets::Large2DShapes(), framework::dataset::make("Axis", -2, 3));

/** Shapes 3D to test */
const auto shapes_3d_large = combine(datasets::Large3DShapes(), framework::dataset::make("Axis", -3, 4));

/** Shapes 4D to test */
const auto shapes_4d_large = combine(datasets::Large4DShapes(), framework::dataset::make("Axis", -4, 5));

/** Configuration test */
void validate_configuration(TensorShape shape_in, int axis, DataType data_type, int num_tensors)
{
    // Wrap around negative values
    const unsigned int axis_u = wrap_around(axis, static_cast<int>(shape_in.num_dimensions() + 1));

    const TensorShape shape_dst = compute_stack_shape(TensorInfo(shape_in, 1, data_type), axis_u, num_tensors);

    std::vector<Tensor>   tensors(num_tensors);
    std::vector<ITensor*> src(num_tensors);

    // Create vector of input tensors
    for(int i = 0; i < num_tensors; ++i)
    {
        tensors[i] = create_tensor<Tensor>(shape_in, data_type);
        src[i]     = &(tensors[i]);
        ARM_COMPUTE_EXPECT(src[i]->info()->is_resizable(), framework::LogLevel::ERRORS);
    }

    // Create tensors
    Tensor dst = create_tensor<Tensor>(shape_dst, data_type);

    ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

    // Create and configure function
    NEStackLayer stack;
    stack.configure(src, axis, &dst);
}
} // namespace

/** Fixture to use */
template<typename T>
using NEStackLayerFixture = StackLayerValidationFixture<Tensor, ITensor, Accessor, NEStackLayer, T>;

using namespace arm_compute::misc::shape_calculator;

TEST_SUITE(NEON)
TEST_SUITE(StackLayer)

DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(
                                                                      framework::dataset::make("InputInfo",
{
    std::vector<TensorInfo>{ TensorInfo(TensorShape(9U, 8U), 1, DataType::U8) },
    std::vector<TensorInfo>{ TensorInfo(TensorShape(1U, 2U), 1, DataType::U8) , TensorInfo(TensorShape(1U, 2U), 1, DataType::U8), TensorInfo(TensorShape(1U, 2U), 1, DataType::U8)}, 
    std::vector<TensorInfo>{ TensorInfo(TensorShape(2U, 3U), 1, DataType::S32) },
    std::vector<TensorInfo>{ TensorInfo(TensorShape(7U, 5U, 3U, 8U, 2U), 1, DataType::S32), TensorInfo(TensorShape(7U, 5U, 3U, 8U, 2U), 1, DataType::S32)}, 
    std::vector<TensorInfo>{ TensorInfo(TensorShape(9U, 8U), 1, DataType::S32) },
}),
framework::dataset::make("OutputInfo",
{
    TensorInfo(TensorShape(1U, 9U, 8U), 1, DataType::U8),   // Passes, stack 1 tensor on x axis
    TensorInfo(TensorShape(1U, 3U, 2U), 1, DataType::U8),   // Passes, stack 3 tensors on y axis
    TensorInfo(TensorShape(1U, 2U, 3U), 1, DataType::S32),  // fails axis <  (- input's rank)
    TensorInfo(TensorShape(3U, 7U, 5U), 1, DataType::S32),  // fails, input dimensions > 4
    TensorInfo(TensorShape(1U, 2U, 3U), 1, DataType::U8),   // fails mismatching data types
})),
framework::dataset::make("Axis", { -3, 1, -4, -3, 1 })),
framework::dataset::make("Expected", { true, true, false, false, false })),
input_info, output_info, axis, expected)
{
    std::vector<TensorInfo>    ti(input_info);
    std::vector<ITensorInfo *> vec(input_info.size());
    for(size_t j = 0; j < vec.size(); ++j)
    {
        vec[j] = &ti[j];
    }
    ARM_COMPUTE_EXPECT(bool(NEStackLayer::validate(vec, axis, &output_info)) == expected, framework::LogLevel::ERRORS);
}

TEST_SUITE(Shapes1D)

DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(combine(shapes_1d_small,
                                                                           data_types),
                                                                           n_values),
shape_in, axis, data_type, num_tensors)
{
    validate_configuration(shape_in, axis, data_type, num_tensors);
}

TEST_SUITE(S32)
FIXTURE_DATA_TEST_CASE(RunSmall, NEStackLayerFixture<int>, framework::DatasetMode::ALL,
                                                           combine(combine(shapes_1d_small,
                                                                           framework::dataset::make("DataType", { DataType::S32 })),
                                                                           n_values))
{
    // Validate output
    validate(Accessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge, NEStackLayerFixture<int>, framework::DatasetMode::NIGHTLY,
                                                           combine(combine(shapes_1d_large,
                                                                           framework::dataset::make("DataType", { DataType::S32 })),
                                                                           n_values))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // S32

TEST_SUITE(S16)
FIXTURE_DATA_TEST_CASE(RunSmall, NEStackLayerFixture<short>, framework::DatasetMode::ALL,
                                                           combine(combine(shapes_1d_small,
                                                                           framework::dataset::make("DataType", { DataType::S16 })),
                                                                           n_values))
{
    // Validate output
    validate(Accessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge, NEStackLayerFixture<short>, framework::DatasetMode::NIGHTLY,
                                                           combine(combine(shapes_1d_large,
                                                                           framework::dataset::make("DataType", { DataType::S16 })),
                                                                           n_values))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // S16

TEST_SUITE(S8)
FIXTURE_DATA_TEST_CASE(RunSmall, NEStackLayerFixture<char>, framework::DatasetMode::ALL,
                                                           combine(combine(shapes_1d_small,
                                                                           framework::dataset::make("DataType", { DataType::S8 })),
                                                                           n_values))
{
    // Validate output
    validate(Accessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge, NEStackLayerFixture<char>, framework::DatasetMode::NIGHTLY,
                                                           combine(combine(shapes_1d_large,
                                                                           framework::dataset::make("DataType", { DataType::S8 })),
                                                                           n_values))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // S8
TEST_SUITE_END() // Shapes1D

TEST_SUITE(Shapes2D)

DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(combine(shapes_2d_small,
                                                                           data_types),
                                                                           n_values),
shape_in, axis, data_type, num_tensors)
{
    validate_configuration(shape_in, axis, data_type, num_tensors);
}

TEST_SUITE(S32)
FIXTURE_DATA_TEST_CASE(RunSmall, NEStackLayerFixture<int>, framework::DatasetMode::ALL,
                                                           combine(combine(shapes_2d_small,
                                                                           framework::dataset::make("DataType", { DataType::S32 })),
                                                                           n_values))
{
    // Validate output
    validate(Accessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge, NEStackLayerFixture<int>, framework::DatasetMode::NIGHTLY,
                                                           combine(combine(shapes_2d_large,
                                                                           framework::dataset::make("DataType", { DataType::S32 })),
                                                                           n_values))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // S32

TEST_SUITE(S16)
FIXTURE_DATA_TEST_CASE(RunSmall, NEStackLayerFixture<short>, framework::DatasetMode::ALL,
                                                           combine(combine(shapes_2d_small,
                                                                           framework::dataset::make("DataType", { DataType::S16 })),
                                                                           n_values))
{
    // Validate output
    validate(Accessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge, NEStackLayerFixture<short>, framework::DatasetMode::NIGHTLY,
                                                           combine(combine(shapes_2d_large,
                                                                           framework::dataset::make("DataType", { DataType::S16 })),
                                                                           n_values))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // S16

TEST_SUITE(S8)
FIXTURE_DATA_TEST_CASE(RunSmall, NEStackLayerFixture<char>, framework::DatasetMode::ALL,
                                                           combine(combine(shapes_2d_small,
                                                                           framework::dataset::make("DataType", { DataType::S8 })),
                                                                           n_values))
{
    // Validate output
    validate(Accessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge, NEStackLayerFixture<char>, framework::DatasetMode::NIGHTLY,
                                                           combine(combine(shapes_2d_large,
                                                                           framework::dataset::make("DataType", { DataType::S8 })),
                                                                           n_values))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // S8
TEST_SUITE_END() // Shapes2D

TEST_SUITE(Shapes3D)
DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(combine(shapes_3d_small,
                                                                           data_types),
                                                                           n_values),
shape_in, axis, data_type, num_tensors)
{
    validate_configuration(shape_in, axis, data_type, num_tensors);
}

TEST_SUITE(S32)
FIXTURE_DATA_TEST_CASE(RunSmall, NEStackLayerFixture<int>, framework::DatasetMode::ALL,
                                                           combine(combine(shapes_3d_small,
                                                                           framework::dataset::make("DataType", { DataType::S32 })),
                                                                           n_values))
{
    // Validate output
    validate(Accessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge, NEStackLayerFixture<int>, framework::DatasetMode::NIGHTLY,
                                                           combine(combine(shapes_3d_large,
                                                                           framework::dataset::make("DataType", { DataType::S32 })),
                                                                           n_values))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // S32

TEST_SUITE(S16)
FIXTURE_DATA_TEST_CASE(RunSmall, NEStackLayerFixture<short>, framework::DatasetMode::ALL,
                                                           combine(combine(shapes_3d_small,
                                                                           framework::dataset::make("DataType", { DataType::S16 })),
                                                                           n_values))
{
    // Validate output
    validate(Accessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge, NEStackLayerFixture<short>, framework::DatasetMode::NIGHTLY,
                                                           combine(combine(shapes_3d_large,
                                                                           framework::dataset::make("DataType", { DataType::S16 })),
                                                                           n_values))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // S16

TEST_SUITE(S8)
FIXTURE_DATA_TEST_CASE(RunSmall, NEStackLayerFixture<char>, framework::DatasetMode::ALL,
                                                           combine(combine(shapes_3d_small,
                                                                           framework::dataset::make("DataType", { DataType::S8 })),
                                                                           n_values))
{
    // Validate output
    validate(Accessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge, NEStackLayerFixture<char>, framework::DatasetMode::NIGHTLY,
                                                           combine(combine(shapes_3d_large,
                                                                           framework::dataset::make("DataType", { DataType::S8 })),
                                                                           n_values))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // S8
TEST_SUITE_END() // Shapes3D

TEST_SUITE(Shapes4D)
DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(combine(shapes_4d_small,
                                                                           data_types),
                                                                           n_values),
shape_in, axis, data_type, num_tensors)
{
    validate_configuration(shape_in, axis, data_type, num_tensors);
}

TEST_SUITE(S32)
FIXTURE_DATA_TEST_CASE(RunSmall, NEStackLayerFixture<int>, framework::DatasetMode::ALL,
                                                           combine(combine(shapes_4d_small,
                                                                           framework::dataset::make("DataType", { DataType::S32 })),
                                                                           n_values))
{
    // Validate output
    validate(Accessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge, NEStackLayerFixture<int>, framework::DatasetMode::NIGHTLY,
                                                           combine(combine(shapes_4d_large,
                                                                           framework::dataset::make("DataType", { DataType::S32 })),
                                                                           n_values))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // S32

TEST_SUITE(S16)
FIXTURE_DATA_TEST_CASE(RunSmall, NEStackLayerFixture<short>, framework::DatasetMode::ALL,
                                                           combine(combine(shapes_4d_small,
                                                                           framework::dataset::make("DataType", { DataType::S16 })),
                                                                           n_values))
{
    // Validate output
    validate(Accessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge, NEStackLayerFixture<short>, framework::DatasetMode::NIGHTLY,
                                                           combine(combine(shapes_4d_large,
                                                                           framework::dataset::make("DataType", { DataType::S16 })),
                                                                           n_values))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // S16

TEST_SUITE(S8)
FIXTURE_DATA_TEST_CASE(RunSmall, NEStackLayerFixture<char>, framework::DatasetMode::ALL,
                                                           combine(combine(shapes_4d_small,
                                                                           framework::dataset::make("DataType", { DataType::S8 })),
                                                                           n_values))
{
    // Validate output
    validate(Accessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge, NEStackLayerFixture<char>, framework::DatasetMode::NIGHTLY,
                                                           combine(combine(shapes_4d_large,
                                                                           framework::dataset::make("DataType", { DataType::S8 })),
                                                                           n_values))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // S8
TEST_SUITE_END() // Shapes4D
TEST_SUITE_END() // StackLayer
TEST_SUITE_END() // NEON
} // namespace validation
} // namespace test
} // namespace arm_compute
