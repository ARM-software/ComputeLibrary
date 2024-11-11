/*
 * Copyright (c) 2017-2021, 2024 Arm Limited.
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
#include "arm_compute/runtime/experimental/operators/CpuDequantize.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "tests/NEON/Accessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/DatatypeDataset.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/CpuDequantizeFixture.h"


/*
 * Tests for arm_compute::experimental::op::CpuDequantize which is a shallow wrapper for arm_compute::cpu::CpuDequantize.
 * Any future testing to the functionalities of cpu::CpuDequantize will be tested in tests/NEON/DequantizationLayer.cpp given that experimental::op::CpuDequantize remain a shallow wrapper.
*/

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
        using   framework::dataset::make;
#ifdef ARM_COMPUTE_ENABLE_FP16
const auto data_types = framework::dataset::make("DataType", { DataType::F16, DataType::F32 });
#else  /* ARM_COMPUTE_ENABLE_FP16 */
const auto data_types = framework::dataset::make("DataType", { DataType::F32 });
#endif /* ARM_COMPUTE_ENABLE_FP16 */

const auto dataset_quant_f32 = combine(datasets::SmallShapes(),
                                        datasets::QuantizedTypes(),
                                             make("DataType", DataType::F32),
                                     make("DataLayout", { DataLayout::NCHW })
                                     );

const auto dataset_quant_asymm_signed_f32 = combine(datasets::SmallShapes(),
                                                                  make("QuantizedTypes", { DataType::QASYMM8_SIGNED }),
                                                          make("DataType", DataType::F32),
                                                  make("DataLayout", { DataLayout::NCHW })
                                                  );

const auto dataset_quant_per_channel_f32 = combine(datasets::SmallShapes(), datasets::QuantizedPerChannelTypes(),
                                                         make("DataType", DataType::F32),
                                                 make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })
);

const auto dataset_precommit_f32 = concat(concat(dataset_quant_f32, dataset_quant_per_channel_f32), dataset_quant_asymm_signed_f32);


} // namespace

TEST_SUITE(NEON)
TEST_SUITE(OPERATORS)
TEST_SUITE(CpuDequantize)


// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(
      make("InputInfo", { TensorInfo(TensorShape(16U, 16U, 16U, 5U), 1, DataType::F32),      // Wrong input data type
                                                TensorInfo(TensorShape(16U, 16U, 16U, 5U), 1, DataType::QASYMM8),  // Wrong output data type
                                                TensorInfo(TensorShape(16U, 16U, 2U, 5U), 1, DataType::QASYMM8),   // Missmatching shapes
                                                TensorInfo(TensorShape(17U, 16U, 16U, 5U), 1, DataType::QASYMM8),  // Valid
                                                TensorInfo(TensorShape(16U, 16U, 16U, 5U), 1, DataType::QASYMM8),  // Valid
                                                TensorInfo(TensorShape(16U, 16U, 16U, 5U), 1, DataType::QASYMM8_SIGNED),  // Valid
        }),
      make("OutputInfo",{ TensorInfo(TensorShape(16U, 16U, 16U, 5U), 1, DataType::F32),
                                                TensorInfo(TensorShape(16U, 16U, 16U, 5U), 1, DataType::U8),
                                                TensorInfo(TensorShape(16U, 16U, 16U, 5U), 1, DataType::F32),
                                                TensorInfo(TensorShape(17U, 16U, 16U, 5U), 1, DataType::F32),
                                                TensorInfo(TensorShape(16U, 16U, 16U, 5U), 1, DataType::F32),
                                                TensorInfo(TensorShape(16U, 16U, 16U, 5U), 1, DataType::F32),
        }),
      make("Expected", { false, false, false, true, true, true })),
        input_info, output_info, expected)
{
    ARM_COMPUTE_EXPECT(bool(arm_compute::experimental::op::CpuDequantize::validate(&input_info.clone()->set_is_resizable(false), &output_info.clone()->set_is_resizable(false))) == expected, framework::LogLevel::ERRORS);
}
// // clang-format on

using arm_compute::experimental::op::CpuDequantize;
template <typename T>
using CpuDequantizeFixture = CpuDequantizationValidationFixture<Tensor, Accessor, CpuDequantize, T>;


FIXTURE_DATA_TEST_CASE(SmokeTest, CpuDequantizeFixture<float>, framework::DatasetMode::ALL, dataset_precommit_f32)
{
    // Validate output
    validate(Accessor(_target), _reference);
}


TEST_SUITE_END() // CpuDequantize
TEST_SUITE_END() // OPERATORS
TEST_SUITE_END() // NENO
} // namespace validation
} // namespace test
} // namespace arm_compute
