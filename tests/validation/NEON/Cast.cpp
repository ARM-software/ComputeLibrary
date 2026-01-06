/*
 * Copyright (c) 2019-2026 Arm Limited.
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
#include "arm_compute/runtime/NEON/functions/NECast.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"

#include "src/common/cpuinfo/CpuIsaInfo.h"
#include "src/cpu/kernels/CpuCastKernel.h"
#include "tests/datasets/ConvertPolicyDataset.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/framework/Macros.h"
#include "tests/NEON/Accessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/validation/fixtures/CastFixture.h"
#include "tests/validation/Helpers.h"
#include "tests/validation/Validation.h"

#include <cstdint>
#include <vector>

namespace arm_compute
{
namespace test
{
namespace validation
{

using framework::dataset::make;

namespace
{
// Tolerance
constexpr AbsoluteTolerance<float> one_tolerance(1);
constexpr AbsoluteTolerance<float> zero_tolerance(0);

/*
 *This function ignores the scale and zeroPoint of quanized tensors,so QASYMM8 input is treated as uint8 values.
*/

/** Input data sets **/

// QASYMM8_SIGNED
const auto CastQASYMM8_SIGNEDtoS16Dataset =
    combine(make("DataType", DataType::QASYMM8_SIGNED), make("DataType", DataType::S16));
const auto CastQASYMM8_SIGNEDtoS32Dataset =
    combine(make("DataType", DataType::QASYMM8_SIGNED), make("DataType", DataType::S32));
const auto CastQASYMM8_SIGNEDtoF32Dataset =
    combine(make("DataType", DataType::QASYMM8_SIGNED), make("DataType", DataType::F32));
const auto CastQASYMM8_SIGNEDtoF16Dataset =
    combine(make("DataType", DataType::QASYMM8_SIGNED), make("DataType", DataType::F16));

// QASYMM8
const auto CastQASYMM8toF16Dataset = combine(make("DataType", DataType::QASYMM8), make("DataType", DataType::F16));
const auto CastQASYMM8toF32Dataset = combine(make("DataType", DataType::QASYMM8), make("DataType", DataType::F32));
const auto CastQASYMM8toS32Dataset = combine(make("DataType", DataType::QASYMM8), make("DataType", DataType::S32));

// U8
const auto CastU8toU16Dataset = combine(make("DataType", DataType::U8), make("DataType", DataType::U16));
const auto CastU8toS16Dataset = combine(make("DataType", DataType::U8), make("DataType", DataType::S16));
const auto CastU8toS32Dataset = combine(make("DataType", DataType::U8), make("DataType", DataType::S32));
const auto CastU8toF32Dataset = combine(make("DataType", DataType::U8), make("DataType", DataType::F32));

// U16
const auto CastU16toU8Dataset  = combine(make("DataType", DataType::U16), make("DataType", DataType::U8));
const auto CastU16toU32Dataset = combine(make("DataType", DataType::U16), make("DataType", DataType::U32));

// S16
const auto CastS16toQASYMM8_SIGNEDDataset =
    combine(make("DataType", DataType::S16), make("DataType", DataType::QASYMM8_SIGNED));
const auto CastS16toU8Dataset  = combine(make("DataType", DataType::S16), make("DataType", DataType::U8));
const auto CastS16toS32Dataset = combine(make("DataType", DataType::S16), make("DataType", DataType::S32));

//S32
const auto CastS32toF16Dataset     = combine(make("DataType", DataType::S32), make("DataType", DataType::F16));
const auto CastS32toU8Dataset      = combine(make("DataType", DataType::S32), make("DataType", DataType::U8));
const auto CastS32toF32Dataset     = combine(make("DataType", DataType::S32), make("DataType", DataType::F32));
const auto CastS32toQASYMM8Dataset = combine(make("DataType", DataType::S32), make("DataType", DataType::QASYMM8));
const auto CastS32toQASYMM8_SIGNEDDataset =
    combine(make("DataType", DataType::S32), make("DataType", DataType::QASYMM8_SIGNED));

// F16
const auto CastF16toF32Dataset     = combine(make("DataType", DataType::F16), make("DataType", DataType::F32));
const auto CastF16toS32Dataset     = combine(make("DataType", DataType::F16), make("DataType", DataType::S32));
const auto CastF16toQASYMM8Dataset = combine(make("DataType", DataType::F16), make("DataType", DataType::QASYMM8));
const auto CastF16toQASYMM8_SIGNEDDataset =
    combine(make("DataType", DataType::F16), make("DataType", DataType::QASYMM8_SIGNED));

// F32
const auto CastF32toU8Dataset      = combine(make("DataType", DataType::F32), make("DataType", DataType::U8));
const auto CastF32toF16Dataset     = combine(make("DataType", DataType::F32), make("DataType", DataType::F16));
const auto CastF32toS32Dataset     = combine(make("DataType", DataType::F32), make("DataType", DataType::S32));
const auto CastF32toQASYMM8Dataset = combine(make("DataType", DataType::F32), make("DataType", DataType::QASYMM8));
const auto CastF32toQASYMM8_SIGNEDDataset =
    combine(make("DataType", DataType::F32), make("DataType", DataType::QASYMM8_SIGNED));

// U64
const auto CastU64toF32Dataset = combine(make("DataType", DataType::U64), make("DataType", DataType::F32));

// S64
const auto CastS64toF32Dataset = combine(make("DataType", DataType::S64), make("DataType", DataType::F32));

template <typename T>
void validate_static_cast(const TensorShape &shape, DataType src_dtype, DataType dst_dtype)
{
    Tensor input  = create_tensor<Tensor>(shape, src_dtype, 1);
    Tensor output = create_tensor<Tensor>(shape, dst_dtype, 1);

    NECast cast;
    cast.configure(&input, &output, ConvertPolicy::SATURATE);
    input.allocator()->allocate();
    output.allocator()->allocate();

    library->fill_tensor_value(Accessor(input), 1.99f);
    cast.run();

    for (unsigned int i = 0; i < shape.x(); ++i)
    {
        const T ref    = 1;
        const T target = reinterpret_cast<T *>(output.buffer())[i];

        ARM_COMPUTE_EXPECT(ref == target, framework::LogLevel::ERRORS);
    }
}

} // namespace

TEST_SUITE(NEON)
TEST_SUITE(Cast)

// Validate casting truncates floats to integer instead of rounding
DATA_TEST_CASE(ValidateStaticCastBehavior,
               framework::DatasetMode::ALL,
               combine(make("InputDataType", {DataType::F32, DataType::F16}),
                       make("OutputDataType", {DataType::QASYMM8, DataType::QASYMM8_SIGNED, DataType::U8})),
               src_dtype,
               dst_dtype)
{
    const auto shape = TensorShape(18U); // > 16 for channel dim. to stress vector and leftover loops

    if (src_dtype == DataType::F32 || (src_dtype == DataType::F16 && cpu_supports_dtypes({DataType::F16})))
    {
        if (dst_dtype == DataType::QASYMM8_SIGNED)
        {
            validate_static_cast<int8_t>(shape, src_dtype, dst_dtype);
        }
        else
        {
            validate_static_cast<uint8_t>(shape, src_dtype, dst_dtype);
        }
    }
}

template <typename T>
using NECastToU8Fixture = CastValidationFixture<Tensor, Accessor, NECast, T, uint8_t>;
template <typename T>
using NECastToS8Fixture = CastValidationFixture<Tensor, Accessor, NECast, T, int8_t>;
template <typename T>
using NECastToU16Fixture = CastValidationFixture<Tensor, Accessor, NECast, T, uint16_t>;
template <typename T>
using NECastToS16Fixture = CastValidationFixture<Tensor, Accessor, NECast, T, int16_t>;
template <typename T>
using NECastToU32Fixture = CastValidationFixture<Tensor, Accessor, NECast, T, uint32_t>;
template <typename T>
using NECastToS32Fixture = CastValidationFixture<Tensor, Accessor, NECast, T, int32_t>;
template <typename T>
using NECastToU64Fixture = CastValidationFixture<Tensor, Accessor, NECast, T, uint64_t>;
template <typename T>
using NECastToS64Fixture = CastValidationFixture<Tensor, Accessor, NECast, T, int64_t>;
template <typename T>
using NECastToF16Fixture = CastValidationFixture<Tensor, Accessor, NECast, T, half>;
template <typename T>
using NECastToF32Fixture = CastValidationFixture<Tensor, Accessor, NECast, T, float>;
template <typename T>
using NECastToQASYMM8Fixture = CastValidationFixture<Tensor, Accessor, NECast, T, uint8_t>;
template <typename T>
using NECastToQASYMM8_SIGNEDFixture = CastValidationFixture<Tensor, Accessor, NECast, T, int8_t>;

#define CAST_SUITE(NAME, idt, odt, type, dataset, tolerance)                                           \
    TEST_SUITE(NAME)                                                                                   \
    FIXTURE_DATA_TEST_CASE(RunSmall, type, framework::DatasetMode::PRECOMMIT,                          \
                           combine(datasets::SmallShapes(), dataset, datasets::ConvertPolicies()))     \
    {                                                                                                  \
        if ((idt != DataType::F16 && odt != DataType::F16) || CPUInfo::get().has_fp16())               \
        {                                                                                              \
            validate(Accessor(_target), _reference, tolerance);                                        \
        }                                                                                              \
        else                                                                                           \
        {                                                                                              \
            ARM_COMPUTE_TEST_WARNING("Device does not support fp16 vector operations. Test SKIPPED."); \
            framework::ARM_COMPUTE_PRINT_WARNING();                                                    \
        }                                                                                              \
    }                                                                                                  \
    TEST_SUITE_END()

//QASYMM8_SIGNED
CAST_SUITE(QASYMM8_SIGNED_to_S16,
           DataType::QASYMM8_SIGNED,
           DataType::S16,
           NECastToS16Fixture<int8_t>,
           CastQASYMM8_SIGNEDtoS16Dataset,
           one_tolerance)
CAST_SUITE(QASYMM8_SIGNED_to_S32,
           DataType::QASYMM8_SIGNED,
           DataType::S32,
           NECastToS32Fixture<int8_t>,
           CastQASYMM8_SIGNEDtoS32Dataset,
           one_tolerance)
CAST_SUITE(QASYMM8_SIGNED_to_F32,
           DataType::QASYMM8_SIGNED,
           DataType::F32,
           NECastToF32Fixture<int8_t>,
           CastQASYMM8_SIGNEDtoF32Dataset,
           one_tolerance)
#ifdef ARM_COMPUTE_ENABLE_FP16
CAST_SUITE(QASYMM8_SIGNED_to_F16,
           DataType::QASYMM8_SIGNED,
           DataType::F16,
           NECastToF16Fixture<int8_t>,
           CastQASYMM8_SIGNEDtoF16Dataset,
           one_tolerance)
#endif //  ARM_COMPUTE_ENABLE_FP16

//QASYMM8
#ifdef ARM_COMPUTE_ENABLE_FP16
CAST_SUITE(QASYMM8_to_F16,
           DataType::QASYMM8,
           DataType::F16,
           NECastToF16Fixture<uint8_t>,
           CastQASYMM8toF16Dataset,
           one_tolerance)
#endif //  ARM_COMPUTE_ENABLE_FP16
CAST_SUITE(QASYMM8_to_F32,
           DataType::QASYMM8,
           DataType::F32,
           NECastToF32Fixture<uint8_t>,
           CastQASYMM8toF32Dataset,
           one_tolerance)
CAST_SUITE(QASYMM8_to_S32,
           DataType::QASYMM8,
           DataType::S32,
           NECastToS32Fixture<uint8_t>,
           CastQASYMM8toS32Dataset,
           one_tolerance)

// U8
CAST_SUITE(U8_to_U16, DataType::U8, DataType::U16, NECastToU16Fixture<uint8_t>, CastU8toU16Dataset, zero_tolerance)
CAST_SUITE(U8_to_S16, DataType::U8, DataType::S16, NECastToS16Fixture<uint8_t>, CastU8toS16Dataset, zero_tolerance)
CAST_SUITE(U8_to_S32, DataType::U8, DataType::S32, NECastToS32Fixture<uint8_t>, CastU8toS32Dataset, zero_tolerance)
CAST_SUITE(U8_to_F32, DataType::U8, DataType::F32, NECastToF32Fixture<uint8_t>, CastU8toF32Dataset, zero_tolerance)

// U16
CAST_SUITE(U16_to_U8, DataType::U16, DataType::U8, NECastToU8Fixture<uint16_t>, CastU16toU8Dataset, zero_tolerance)
CAST_SUITE(U16_to_U32, DataType::U16, DataType::U32, NECastToU32Fixture<uint16_t>, CastU16toU32Dataset, zero_tolerance)

// S16
CAST_SUITE(S16_to_QASYMM8_SIGNED,
           DataType::S16,
           DataType::QASYMM8_SIGNED,
           NECastToQASYMM8_SIGNEDFixture<int16_t>,
           CastS16toQASYMM8_SIGNEDDataset,
           zero_tolerance)
CAST_SUITE(S16_to_U8, DataType::S16, DataType::U8, NECastToU8Fixture<int16_t>, CastS16toU8Dataset, zero_tolerance)
CAST_SUITE(S16_to_S32, DataType::S16, DataType::S32, NECastToS32Fixture<int16_t>, CastS16toS32Dataset, zero_tolerance)

// S32
CAST_SUITE(S32_to_QASYMM8_SIGNED,
           DataType::S32,
           DataType::QASYMM8_SIGNED,
           NECastToQASYMM8_SIGNEDFixture<int32_t>,
           CastS32toQASYMM8_SIGNEDDataset,
           one_tolerance)
CAST_SUITE(S32_to_QASYMM8,
           DataType::S32,
           DataType::QASYMM8,
           NECastToQASYMM8Fixture<int32_t>,
           CastS32toQASYMM8Dataset,
           one_tolerance)
#ifdef ARM_COMPUTE_ENABLE_FP16
CAST_SUITE(S32_to_F16, DataType::S32, DataType::F16, NECastToF16Fixture<int32_t>, CastS32toF16Dataset, zero_tolerance)
#endif //  ARM_COMPUTE_ENABLE_FP16
CAST_SUITE(S32_to_F32, DataType::S32, DataType::F32, NECastToF32Fixture<int32_t>, CastS32toF32Dataset, one_tolerance)
CAST_SUITE(S32_to_U8, DataType::S32, DataType::U8, NECastToU8Fixture<int32_t>, CastS32toU8Dataset, one_tolerance)

// F16
#ifdef ARM_COMPUTE_ENABLE_FP16
CAST_SUITE(F16_to_QASYMM8_SIGNED,
           DataType::F16,
           DataType::QASYMM8_SIGNED,
           NECastToQASYMM8_SIGNEDFixture<half>,
           CastF16toQASYMM8_SIGNEDDataset,
           one_tolerance)
CAST_SUITE(F16_to_QASYMM8,
           DataType::F16,
           DataType::QASYMM8,
           NECastToQASYMM8Fixture<half>,
           CastF16toQASYMM8Dataset,
           one_tolerance)
CAST_SUITE(F16_to_F32, DataType::F16, DataType::F32, NECastToF32Fixture<half>, CastF16toF32Dataset, zero_tolerance)
CAST_SUITE(F16_to_S32, DataType::F16, DataType::S32, NECastToS32Fixture<half>, CastF16toS32Dataset, one_tolerance)
#endif //  ARM_COMPUTE_ENABLE_FP16

// F32
CAST_SUITE(F32_to_QASYMM8_SIGNED,
           DataType::F32,
           DataType::QASYMM8_SIGNED,
           NECastToQASYMM8_SIGNEDFixture<float>,
           CastF32toQASYMM8_SIGNEDDataset,
           one_tolerance)
CAST_SUITE(F32_to_QASYMM8,
           DataType::F32,
           DataType::QASYMM8,
           NECastToQASYMM8Fixture<float>,
           CastF32toQASYMM8Dataset,
           one_tolerance)
#ifdef ARM_COMPUTE_ENABLE_FP16
CAST_SUITE(F32_to_F16, DataType::F32, DataType::F16, NECastToF16Fixture<float>, CastF32toF16Dataset, zero_tolerance)
#endif //  ARM_COMPUTE_ENABLE_FP16
CAST_SUITE(F32_to_S32, DataType::F32, DataType::S32, NECastToS32Fixture<float>, CastF32toS32Dataset, one_tolerance)
CAST_SUITE(F32_to_U8, DataType::F32, DataType::U8, NECastToU8Fixture<float>, CastF32toU8Dataset, one_tolerance)

#ifdef __aarch64__
// S64
CAST_SUITE(S64_to_F32, DataType::S64, DataType::F32, NECastToF32Fixture<int64_t>, CastS64toF32Dataset, zero_tolerance)

// U64
CAST_SUITE(U64_to_F32, DataType::U64, DataType::F32, NECastToF32Fixture<uint64_t>, CastU64toF32Dataset, zero_tolerance)
#endif // __aarch64__

DATA_TEST_CASE(KernelSelectionDstFP16,
               framework::DatasetMode::ALL,
               combine(make("CpuExt", std::string("NEON")),
                       make("DataType",
                            {
                                DataType::F16,
                                DataType::U8,
                                DataType::S32,
                                DataType::QASYMM8,
                                DataType::QASYMM8_SIGNED,
                            })),
               cpu_ext,
               data_type)
{
    using namespace cpu::kernels;
    const CpuCastKernel::CastKernel *selected_impl;

    cpuinfo::CpuIsaInfo cpu_isa{};
    cpu_isa.neon = (cpu_ext == "NEON");
    cpu_isa.fp16 = true;

    selected_impl = CpuCastKernel::get_implementation(CastDataTypeISASelectorData{data_type, DataType::F16, cpu_isa},
                                                      cpu::KernelSelectionType::Preferred);
    ARM_COMPUTE_ERROR_ON_NULLPTR(selected_impl);

    std::string expected = lower_string(cpu_ext) + "_" + cpu_impl_dt(data_type) + "_cast";
    std::string actual   = selected_impl->name;

    ARM_COMPUTE_EXPECT_EQUAL(expected, actual, framework::LogLevel::ERRORS);
}

DATA_TEST_CASE(KernelSelectionSrcFP32,
               framework::DatasetMode::ALL,
               combine(make("CpuExt", std::string("NEON")),
                       make("DataType",
                            {
                                DataType::F16,
                            })),
               cpu_ext,
               data_type)
{
    using namespace cpu::kernels;

    cpuinfo::CpuIsaInfo cpu_isa{};
    cpu_isa.neon = (cpu_ext == "NEON");
    cpu_isa.fp16 = (data_type == DataType::F16);

    const auto *selected_impl = CpuCastKernel::get_implementation(
        CastDataTypeISASelectorData{DataType::F32, data_type, cpu_isa}, cpu::KernelSelectionType::Preferred);

    ARM_COMPUTE_ERROR_ON_NULLPTR(selected_impl);

    std::string expected = lower_string(cpu_ext) + "_fp32_to_" + cpu_impl_dt(data_type) + "_cast";
    std::string actual   = selected_impl->name;

    ARM_COMPUTE_EXPECT_EQUAL(expected, actual, framework::LogLevel::ERRORS);
}

TEST_SUITE_END() // Cast
TEST_SUITE_END() // Neon
} // namespace validation
} // namespace test
} // namespace arm_compute
