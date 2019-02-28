/*
 * Copyright (c) 2018 ARM Limited.
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
 * OUT OF OR IN CONCLCTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/CLTensorAllocator.h"
#include "arm_compute/runtime/CL/functions/CLCast.h"
#include "tests/CL/CLAccessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/ConvertPolicyDataset.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/CastFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
// Tolerance
constexpr AbsoluteTolerance<float> one_tolerance(1);
constexpr AbsoluteTolerance<float> zero_tolerance(0);

/** Input data sets **/
// U8
const auto CastU8toS8Dataset  = combine(framework::dataset::make("DataType", DataType::U8), framework::dataset::make("DataType", DataType::S8));
const auto CastU8toU16Dataset = combine(framework::dataset::make("DataType", DataType::U8), framework::dataset::make("DataType", DataType::U16));
const auto CastU8toS16Dataset = combine(framework::dataset::make("DataType", DataType::U8), framework::dataset::make("DataType", DataType::S16));
const auto CastU8toU32Dataset = combine(framework::dataset::make("DataType", DataType::U8), framework::dataset::make("DataType", DataType::U32));
const auto CastU8toS32Dataset = combine(framework::dataset::make("DataType", DataType::U8), framework::dataset::make("DataType", DataType::S32));
const auto CastU8toF16Dataset = combine(framework::dataset::make("DataType", DataType::U8), framework::dataset::make("DataType", DataType::F16));
const auto CastU8toF32Dataset = combine(framework::dataset::make("DataType", DataType::U8), framework::dataset::make("DataType", DataType::F32));

// S8
const auto CastS8toU8Dataset  = combine(framework::dataset::make("DataType", DataType::S8), framework::dataset::make("DataType", DataType::U8));
const auto CastS8toU16Dataset = combine(framework::dataset::make("DataType", DataType::S8), framework::dataset::make("DataType", DataType::U16));
const auto CastS8toS16Dataset = combine(framework::dataset::make("DataType", DataType::S8), framework::dataset::make("DataType", DataType::S16));
const auto CastS8toU32Dataset = combine(framework::dataset::make("DataType", DataType::S8), framework::dataset::make("DataType", DataType::U32));
const auto CastS8toS32Dataset = combine(framework::dataset::make("DataType", DataType::S8), framework::dataset::make("DataType", DataType::S32));
const auto CastS8toF16Dataset = combine(framework::dataset::make("DataType", DataType::S8), framework::dataset::make("DataType", DataType::F16));
const auto CastS8toF32Dataset = combine(framework::dataset::make("DataType", DataType::S8), framework::dataset::make("DataType", DataType::F32));

// U16
const auto CastU16toU8Dataset  = combine(framework::dataset::make("DataType", DataType::U16), framework::dataset::make("DataType", DataType::U8));
const auto CastU16toS8Dataset  = combine(framework::dataset::make("DataType", DataType::U16), framework::dataset::make("DataType", DataType::S8));
const auto CastU16toS16Dataset = combine(framework::dataset::make("DataType", DataType::U16), framework::dataset::make("DataType", DataType::S16));
const auto CastU16toU32Dataset = combine(framework::dataset::make("DataType", DataType::U16), framework::dataset::make("DataType", DataType::U32));
const auto CastU16toS32Dataset = combine(framework::dataset::make("DataType", DataType::U16), framework::dataset::make("DataType", DataType::S32));
const auto CastU16toF16Dataset = combine(framework::dataset::make("DataType", DataType::U16), framework::dataset::make("DataType", DataType::F16));
const auto CastU16toF32Dataset = combine(framework::dataset::make("DataType", DataType::U16), framework::dataset::make("DataType", DataType::F32));

// S16
const auto CastS16toU8Dataset  = combine(framework::dataset::make("DataType", DataType::S16), framework::dataset::make("DataType", DataType::U8));
const auto CastS16toS8Dataset  = combine(framework::dataset::make("DataType", DataType::S16), framework::dataset::make("DataType", DataType::S8));
const auto CastS16toU16Dataset = combine(framework::dataset::make("DataType", DataType::S16), framework::dataset::make("DataType", DataType::U16));
const auto CastS16toU32Dataset = combine(framework::dataset::make("DataType", DataType::S16), framework::dataset::make("DataType", DataType::U32));
const auto CastS16toS32Dataset = combine(framework::dataset::make("DataType", DataType::S16), framework::dataset::make("DataType", DataType::S32));
const auto CastS16toF16Dataset = combine(framework::dataset::make("DataType", DataType::S16), framework::dataset::make("DataType", DataType::F16));
const auto CastS16toF32Dataset = combine(framework::dataset::make("DataType", DataType::S16), framework::dataset::make("DataType", DataType::F32));

// U32
const auto CastU32toU8Dataset  = combine(framework::dataset::make("DataType", DataType::U32), framework::dataset::make("DataType", DataType::U8));
const auto CastU32toS8Dataset  = combine(framework::dataset::make("DataType", DataType::U32), framework::dataset::make("DataType", DataType::S8));
const auto CastU32toU16Dataset = combine(framework::dataset::make("DataType", DataType::U32), framework::dataset::make("DataType", DataType::U16));
const auto CastU32toS16Dataset = combine(framework::dataset::make("DataType", DataType::U32), framework::dataset::make("DataType", DataType::S16));
const auto CastU32toS32Dataset = combine(framework::dataset::make("DataType", DataType::U32), framework::dataset::make("DataType", DataType::S32));
const auto CastU32toF16Dataset = combine(framework::dataset::make("DataType", DataType::U32), framework::dataset::make("DataType", DataType::F16));
const auto CastU32toF32Dataset = combine(framework::dataset::make("DataType", DataType::U32), framework::dataset::make("DataType", DataType::F32));

// S32
const auto CastS32toU8Dataset  = combine(framework::dataset::make("DataType", DataType::S32), framework::dataset::make("DataType", DataType::U8));
const auto CastS32toS8Dataset  = combine(framework::dataset::make("DataType", DataType::S32), framework::dataset::make("DataType", DataType::S8));
const auto CastS32toU16Dataset = combine(framework::dataset::make("DataType", DataType::S32), framework::dataset::make("DataType", DataType::U16));
const auto CastS32toS16Dataset = combine(framework::dataset::make("DataType", DataType::S32), framework::dataset::make("DataType", DataType::S16));
const auto CastS32toU32Dataset = combine(framework::dataset::make("DataType", DataType::S32), framework::dataset::make("DataType", DataType::U32));
const auto CastS32toF16Dataset = combine(framework::dataset::make("DataType", DataType::S32), framework::dataset::make("DataType", DataType::F16));
const auto CastS32toF32Dataset = combine(framework::dataset::make("DataType", DataType::S32), framework::dataset::make("DataType", DataType::F32));

// F16
const auto CastF16toU8Dataset  = combine(framework::dataset::make("DataType", DataType::F16), framework::dataset::make("DataType", DataType::U8));
const auto CastF16toS8Dataset  = combine(framework::dataset::make("DataType", DataType::F16), framework::dataset::make("DataType", DataType::S8));
const auto CastF16toU16Dataset = combine(framework::dataset::make("DataType", DataType::F16), framework::dataset::make("DataType", DataType::U16));
const auto CastF16toS16Dataset = combine(framework::dataset::make("DataType", DataType::F16), framework::dataset::make("DataType", DataType::S16));
const auto CastF16toU32Dataset = combine(framework::dataset::make("DataType", DataType::F16), framework::dataset::make("DataType", DataType::U32));
const auto CastF16toS32Dataset = combine(framework::dataset::make("DataType", DataType::F16), framework::dataset::make("DataType", DataType::S32));
const auto CastF16toF32Dataset = combine(framework::dataset::make("DataType", DataType::F16), framework::dataset::make("DataType", DataType::F32));

// F32
const auto CastF32toU8Dataset  = combine(framework::dataset::make("DataType", DataType::F32), framework::dataset::make("DataType", DataType::U8));
const auto CastF32toS8Dataset  = combine(framework::dataset::make("DataType", DataType::F32), framework::dataset::make("DataType", DataType::S8));
const auto CastF32toU16Dataset = combine(framework::dataset::make("DataType", DataType::F32), framework::dataset::make("DataType", DataType::U16));
const auto CastF32toS16Dataset = combine(framework::dataset::make("DataType", DataType::F32), framework::dataset::make("DataType", DataType::S16));
const auto CastF32toU32Dataset = combine(framework::dataset::make("DataType", DataType::F32), framework::dataset::make("DataType", DataType::U32));
const auto CastF32toS32Dataset = combine(framework::dataset::make("DataType", DataType::F32), framework::dataset::make("DataType", DataType::S32));
const auto CastF32toF16Dataset = combine(framework::dataset::make("DataType", DataType::F32), framework::dataset::make("DataType", DataType::F16));
} // namespace

TEST_SUITE(CL)
TEST_SUITE(Cast)
template <typename T>
using CLCastToU8Fixture = CastValidationFixture<CLTensor, CLAccessor, CLCast, T, uint8_t>;
template <typename T>
using CLCastToS8Fixture = CastValidationFixture<CLTensor, CLAccessor, CLCast, T, int8_t>;
template <typename T>
using CLCastToU16Fixture = CastValidationFixture<CLTensor, CLAccessor, CLCast, T, uint16_t>;
template <typename T>
using CLCastToS16Fixture = CastValidationFixture<CLTensor, CLAccessor, CLCast, T, int16_t>;
template <typename T>
using CLCastToU32Fixture = CastValidationFixture<CLTensor, CLAccessor, CLCast, T, uint32_t>;
template <typename T>
using CLCastToS32Fixture = CastValidationFixture<CLTensor, CLAccessor, CLCast, T, int32_t>;
template <typename T>
using CLCastToF16Fixture = CastValidationFixture<CLTensor, CLAccessor, CLCast, T, half>;
template <typename T>
using CLCastToF32Fixture = CastValidationFixture<CLTensor, CLAccessor, CLCast, T, float>;

#define CAST_SUITE(NAME, idt, odt, type, dataset, tolerance)                                                                     \
    TEST_SUITE(NAME)                                                                                                             \
    DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(datasets::SmallShapes(), datasets::ConvertPolicies()),    \
                   shape, policy)                                                                                                \
    {                                                                                                                            \
        CLTensor src = create_tensor<CLTensor>(shape, idt, 1);                                                                   \
        CLTensor dst = create_tensor<CLTensor>(shape, odt, 1);                                                                   \
        \
        CLCast cast;                                                                                                             \
        cast.configure(&src, &dst, policy);                                                                                      \
        \
        const ValidRegion valid_region = shape_to_valid_region(shape);                                                           \
        validate(dst.info()->valid_region(), valid_region);                                                                      \
        \
        const PaddingSize padding = PaddingCalculator(shape.x(), 16).required_padding();                                         \
        validate(src.info()->padding(), padding);                                                                                \
        validate(dst.info()->padding(), padding);                                                                                \
    }                                                                                                                            \
    FIXTURE_DATA_TEST_CASE(RunSmall, type, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SmallShapes(), dataset), \
                                                                                      datasets::ConvertPolicies()))              \
    {                                                                                                                            \
        validate(CLAccessor(_target), _reference, tolerance);                                                                    \
    }                                                                                                                            \
    TEST_SUITE_END()

// U8
CAST_SUITE(U8_to_S8, DataType::U8, DataType::S8, CLCastToS8Fixture<uint8_t>, CastU8toS8Dataset, zero_tolerance)
CAST_SUITE(U8_to_U16, DataType::U8, DataType::U16, CLCastToU16Fixture<uint8_t>, CastU8toU16Dataset, zero_tolerance)
CAST_SUITE(U8_to_S16, DataType::U8, DataType::S16, CLCastToS16Fixture<uint8_t>, CastU8toS16Dataset, zero_tolerance)
CAST_SUITE(U8_to_U32, DataType::U8, DataType::U32, CLCastToU32Fixture<uint8_t>, CastU8toU32Dataset, zero_tolerance)
CAST_SUITE(U8_to_S32, DataType::U8, DataType::S32, CLCastToS32Fixture<uint8_t>, CastU8toS32Dataset, zero_tolerance)
CAST_SUITE(U8_to_F16, DataType::U8, DataType::F16, CLCastToF16Fixture<uint8_t>, CastU8toF16Dataset, zero_tolerance)
CAST_SUITE(U8_to_F32, DataType::U8, DataType::F32, CLCastToF32Fixture<uint8_t>, CastU8toF32Dataset, zero_tolerance)

// S8
CAST_SUITE(S8_to_U8, DataType::S8, DataType::U8, CLCastToU8Fixture<int8_t>, CastS8toU8Dataset, zero_tolerance)
CAST_SUITE(S8_to_U16, DataType::S8, DataType::U16, CLCastToU16Fixture<int8_t>, CastS8toU16Dataset, zero_tolerance)
CAST_SUITE(S8_to_S16, DataType::S8, DataType::S16, CLCastToS16Fixture<int8_t>, CastS8toS16Dataset, zero_tolerance)
CAST_SUITE(S8_to_U32, DataType::S8, DataType::U32, CLCastToU32Fixture<int8_t>, CastS8toU32Dataset, zero_tolerance)
CAST_SUITE(S8_to_S32, DataType::S8, DataType::S32, CLCastToS32Fixture<int8_t>, CastS8toS32Dataset, zero_tolerance)
CAST_SUITE(S8_to_F16, DataType::S8, DataType::F16, CLCastToF16Fixture<int8_t>, CastS8toF16Dataset, zero_tolerance)
CAST_SUITE(S8_to_F32, DataType::S8, DataType::F32, CLCastToF32Fixture<int8_t>, CastS8toF32Dataset, zero_tolerance)

// U16
CAST_SUITE(U16_to_U8, DataType::U16, DataType::U8, CLCastToU8Fixture<uint16_t>, CastU16toU8Dataset, zero_tolerance)
CAST_SUITE(U16_to_S8, DataType::U16, DataType::S8, CLCastToS8Fixture<uint16_t>, CastU16toS8Dataset, zero_tolerance)
CAST_SUITE(U16_to_S16, DataType::U16, DataType::S16, CLCastToS16Fixture<uint16_t>, CastU16toS16Dataset, zero_tolerance)
CAST_SUITE(U16_to_U32, DataType::U16, DataType::U32, CLCastToU32Fixture<uint16_t>, CastU16toU32Dataset, zero_tolerance)
CAST_SUITE(U16_to_S32, DataType::U16, DataType::S32, CLCastToS32Fixture<uint16_t>, CastU16toS32Dataset, zero_tolerance)
CAST_SUITE(U16_to_F16, DataType::U16, DataType::F16, CLCastToF16Fixture<uint16_t>, CastU16toF16Dataset, zero_tolerance)
CAST_SUITE(U16_to_F32, DataType::U16, DataType::F32, CLCastToF32Fixture<uint16_t>, CastU16toF32Dataset, zero_tolerance)

// S16
CAST_SUITE(S16_to_U8, DataType::S16, DataType::U8, CLCastToU8Fixture<int16_t>, CastS16toU8Dataset, zero_tolerance)
CAST_SUITE(S16_to_S8, DataType::S16, DataType::S8, CLCastToS8Fixture<int16_t>, CastS16toS8Dataset, zero_tolerance)
CAST_SUITE(S16_to_U16, DataType::S16, DataType::U16, CLCastToU16Fixture<int16_t>, CastS16toU16Dataset, zero_tolerance)
CAST_SUITE(S16_to_U32, DataType::S16, DataType::U32, CLCastToU32Fixture<int16_t>, CastS16toU32Dataset, zero_tolerance)
CAST_SUITE(S16_to_S32, DataType::S16, DataType::S32, CLCastToS32Fixture<int16_t>, CastS16toS32Dataset, zero_tolerance)
CAST_SUITE(S16_to_F16, DataType::S16, DataType::F16, CLCastToF16Fixture<int16_t>, CastS16toF16Dataset, zero_tolerance)
CAST_SUITE(S16_to_F32, DataType::S16, DataType::F32, CLCastToF32Fixture<int16_t>, CastS16toF32Dataset, zero_tolerance)

// U32
CAST_SUITE(U32_to_U8, DataType::U32, DataType::U8, CLCastToU8Fixture<uint32_t>, CastU32toU8Dataset, zero_tolerance)
CAST_SUITE(U32_to_S8, DataType::U32, DataType::S8, CLCastToS8Fixture<uint32_t>, CastU32toS8Dataset, zero_tolerance)
CAST_SUITE(U32_to_U16, DataType::U32, DataType::U16, CLCastToU16Fixture<uint32_t>, CastU32toU16Dataset, zero_tolerance)
CAST_SUITE(U32_to_S16, DataType::U32, DataType::S16, CLCastToS16Fixture<uint32_t>, CastU32toS16Dataset, zero_tolerance)
CAST_SUITE(U32_to_S32, DataType::U32, DataType::S32, CLCastToS32Fixture<uint32_t>, CastU32toS32Dataset, zero_tolerance)
CAST_SUITE(U32_to_F16, DataType::U32, DataType::F16, CLCastToF16Fixture<uint32_t>, CastU32toF16Dataset, zero_tolerance)
CAST_SUITE(U32_to_F32, DataType::U32, DataType::F32, CLCastToF32Fixture<uint32_t>, CastU32toF32Dataset, zero_tolerance)

// S32
CAST_SUITE(S32_to_U8, DataType::S32, DataType::U8, CLCastToU8Fixture<int32_t>, CastS32toU8Dataset, zero_tolerance)
CAST_SUITE(S32_to_S8, DataType::S32, DataType::S8, CLCastToS8Fixture<int32_t>, CastS32toS8Dataset, zero_tolerance)
CAST_SUITE(S32_to_U16, DataType::S32, DataType::U16, CLCastToU16Fixture<int32_t>, CastS32toU16Dataset, zero_tolerance)
CAST_SUITE(S32_to_S16, DataType::S32, DataType::S16, CLCastToS16Fixture<int32_t>, CastS32toS16Dataset, zero_tolerance)
CAST_SUITE(S32_to_U32, DataType::S32, DataType::U32, CLCastToU32Fixture<int32_t>, CastS32toU32Dataset, zero_tolerance)
CAST_SUITE(S32_to_F16, DataType::S32, DataType::F16, CLCastToF16Fixture<int32_t>, CastS32toF16Dataset, zero_tolerance)
CAST_SUITE(S32_to_F32, DataType::S32, DataType::F32, CLCastToF32Fixture<int32_t>, CastS32toF32Dataset, zero_tolerance)

// F16
CAST_SUITE(F16_to_U8, DataType::F16, DataType::U8, CLCastToU8Fixture<half>, CastF16toU8Dataset, one_tolerance)
CAST_SUITE(F16_to_S8, DataType::F16, DataType::S8, CLCastToS8Fixture<half>, CastF16toS8Dataset, one_tolerance)
CAST_SUITE(F16_to_U16, DataType::F16, DataType::U16, CLCastToU16Fixture<half>, CastF16toU16Dataset, one_tolerance)
CAST_SUITE(F16_to_S16, DataType::F16, DataType::S16, CLCastToS16Fixture<half>, CastF16toS16Dataset, one_tolerance)
CAST_SUITE(F16_to_U32, DataType::F16, DataType::U32, CLCastToU32Fixture<half>, CastF16toU32Dataset, one_tolerance)
CAST_SUITE(F16_to_S32, DataType::F16, DataType::S32, CLCastToS32Fixture<half>, CastF16toS32Dataset, one_tolerance)
CAST_SUITE(F16_to_F32, DataType::F16, DataType::F32, CLCastToF32Fixture<half>, CastF16toF32Dataset, zero_tolerance)

// F32
CAST_SUITE(F32_to_U8, DataType::F32, DataType::U8, CLCastToU8Fixture<float>, CastF32toU8Dataset, one_tolerance)
CAST_SUITE(F32_to_S8, DataType::F32, DataType::S8, CLCastToS8Fixture<float>, CastF32toS8Dataset, one_tolerance)
CAST_SUITE(F32_to_U16, DataType::F32, DataType::U16, CLCastToU16Fixture<float>, CastF32toU16Dataset, one_tolerance)
CAST_SUITE(F32_to_S16, DataType::F32, DataType::S16, CLCastToS16Fixture<float>, CastF32toS16Dataset, one_tolerance)
CAST_SUITE(F32_to_U32, DataType::F32, DataType::U32, CLCastToU32Fixture<float>, CastF32toU32Dataset, one_tolerance)
CAST_SUITE(F32_to_S32, DataType::F32, DataType::S32, CLCastToS32Fixture<float>, CastF32toS32Dataset, one_tolerance)
CAST_SUITE(F32_to_F16, DataType::F32, DataType::F16, CLCastToF16Fixture<float>, CastF32toF16Dataset, zero_tolerance)

TEST_SUITE_END() // Cast
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute
