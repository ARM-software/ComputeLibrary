/*
 * Copyright (c) 2020-2021 Arm Limited.
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
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "src/core/NEON/kernels/NEQLSTMLayerNormalizationKernel.h"
#include "tests/NEON/Accessor.h"
#include "tests/NEON/Helper.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Helpers.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/QLSTMLayerNormalizationFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
constexpr uint32_t vector_size_byte = 16;

using test::datasets::ShapeDataset;
using NEQLSTMLayerNormalization = NESynthetizeFunction<NEQLSTMLayerNormalizationKernel>;

template <uint32_t num_elements_per_iter, uint32_t num_batches, uint32_t num_iteration>
class QLSTMLayerNormShapeDataSet : public ShapeDataset
{
    static constexpr auto boundary_minus_one = num_elements_per_iter * num_iteration - 1;
    static constexpr auto boundary           = num_elements_per_iter * num_iteration;
    static constexpr auto boundary_plus_one  = num_elements_per_iter * num_iteration + 1;

public:
    QLSTMLayerNormShapeDataSet(std::string name)
        : ShapeDataset(name,
    {
        TensorShape{ boundary_minus_one, num_batches },
                     TensorShape{ boundary, num_batches },
                     TensorShape{ boundary_plus_one, num_batches }
    })
    {
    }
};

template <uint32_t num_elements_per_iter, uint32_t num_batches>
class QLSTMLayerNormShapeDataSet<num_elements_per_iter, num_batches, 0> : public ShapeDataset
{
public:
    QLSTMLayerNormShapeDataSet(std::string name)
        : ShapeDataset(name,
    {
        TensorShape{ 1, num_batches },
                     TensorShape{ 2, num_batches }
    })
    {
    }
};
} // namespace
TEST_SUITE(NEON)
TEST_SUITE(QLSTMLayerNormalization)

static const TensorShape correct_input_shape{ TensorShape(15U, 2U) };
static const TensorShape correct_weight_shape{ TensorShape(15U) };
static const TensorShape correct_bias_shape{ TensorShape(15U) };
static const TensorShape correct_output_shape{ correct_input_shape };
static const DataType    correct_input_dt{ DataType::QSYMM16 };
static const DataType    correct_weight_dt{ DataType::QSYMM16 };
static const DataType    correct_bias_dt{ DataType::S32 };
static const DataType    correct_output_dt{ correct_input_dt };
static const uint32_t    tensor_num_channel{ 1 };

// *INDENT-OFF*
// clang-format off

DATA_TEST_CASE(Validate, framework::DatasetMode::ALL,
    zip(zip(zip(
        framework::dataset::make("InputInfo", {
            TensorInfo(correct_input_shape, tensor_num_channel, DataType::F16), // input supports only QSYMM16
            TensorInfo(correct_input_shape, tensor_num_channel, correct_input_dt), // weight supports only QSYMM16
            TensorInfo(correct_input_shape, tensor_num_channel, correct_input_dt), // bias supports only S32
            TensorInfo(TensorShape(15U, 2U, 2U), tensor_num_channel, correct_input_dt), // input supports only up to 2D
            TensorInfo(correct_input_shape, tensor_num_channel, correct_input_dt), // weight supports only up to 1D
            TensorInfo(correct_input_shape, tensor_num_channel, correct_input_dt), // bias supports only up to 1D
            TensorInfo(correct_input_shape, tensor_num_channel, correct_input_dt), // input_shape[0] != weight_shape[0] should fail
            TensorInfo(correct_input_shape, tensor_num_channel, correct_input_dt), // weight_shape[0] != bias_shape[0] should fail
            TensorInfo(correct_input_shape, tensor_num_channel, correct_input_dt), // output shape mismatches with input shape
            TensorInfo(correct_input_shape, tensor_num_channel, correct_input_dt), // output data type mismatches with input data type
        }),
        framework::dataset::make("WeightInfo", {
            TensorInfo(correct_weight_shape, tensor_num_channel, correct_weight_dt),
            TensorInfo(correct_weight_shape, tensor_num_channel, DataType::F16),
            TensorInfo(correct_weight_shape, tensor_num_channel, correct_weight_dt),
            TensorInfo(correct_weight_shape, tensor_num_channel, correct_weight_dt),
            TensorInfo(TensorShape(15U, 2U), tensor_num_channel, correct_weight_dt),
            TensorInfo(correct_weight_shape, tensor_num_channel, correct_weight_dt),
            TensorInfo(TensorShape(14U), tensor_num_channel, correct_weight_dt),
            TensorInfo(correct_weight_shape, tensor_num_channel, correct_weight_dt),
            TensorInfo(correct_weight_shape, tensor_num_channel, correct_weight_dt),
            TensorInfo(correct_weight_shape, tensor_num_channel, correct_weight_dt),
        })
    ),
        framework::dataset::make("BiasInfo", {
            TensorInfo(correct_bias_shape, tensor_num_channel, correct_bias_dt),
            TensorInfo(correct_bias_shape, tensor_num_channel, correct_bias_dt),
            TensorInfo(correct_bias_shape, tensor_num_channel, DataType::QSYMM16),
            TensorInfo(correct_bias_shape, tensor_num_channel, correct_bias_dt),
            TensorInfo(correct_bias_shape, tensor_num_channel, correct_bias_dt),
            TensorInfo(TensorShape(15U, 2U), tensor_num_channel, correct_bias_dt),
            TensorInfo(correct_bias_shape, tensor_num_channel, correct_bias_dt),
            TensorInfo(TensorShape(14U), tensor_num_channel, correct_bias_dt),
            TensorInfo(correct_bias_shape, tensor_num_channel, correct_bias_dt),
            TensorInfo(correct_bias_shape, tensor_num_channel, correct_bias_dt),
        })
    ),
        framework::dataset::make("OutputInfo", {
            TensorInfo(correct_output_shape, tensor_num_channel, correct_output_dt),
            TensorInfo(correct_output_shape, tensor_num_channel, correct_output_dt),
            TensorInfo(correct_output_shape, tensor_num_channel, correct_output_dt),
            TensorInfo(correct_output_shape, tensor_num_channel, correct_output_dt),
            TensorInfo(correct_output_shape, tensor_num_channel, correct_output_dt),
            TensorInfo(correct_output_shape, tensor_num_channel, correct_output_dt),
            TensorInfo(correct_output_shape, tensor_num_channel, correct_output_dt),
            TensorInfo(correct_output_shape, tensor_num_channel, correct_output_dt),
            TensorInfo(TensorShape(15, 3), tensor_num_channel, correct_output_dt),
            TensorInfo(correct_output_shape, tensor_num_channel, DataType::S32),
        })
    ),
     input_info, weight_info, bias_info, output_info)
{
    const Status s = NEQLSTMLayerNormalization::validate(&input_info, &output_info, &weight_info, &bias_info);
    ARM_COMPUTE_EXPECT(!bool(s), framework::LogLevel::ERRORS);
}

// clang-format on
// *INDENT-ON*

template <typename T>
using NEQLSTMLayerNormalizationFixture = QLSTMLayerNormalizationValidationFixture<Tensor, Accessor, NEQLSTMLayerNormalization, T>;

TEST_SUITE(Quantized)
TEST_SUITE(QSYMM16)

/** Tests will be targetting
 * - Comparison between Neon kernel and the exact same but scalar version of reference kernel
 * - Input shapes of 1D and 2D with the first dimension covers boundary values of 128-bit vector size (0~3 iterations)
 * - Weight and bias 1D shape that have same size as that of input shapes
 * - Quantization scale is greater and smaller than one.
 * - Input values will be noted in fixture.
 *
 * What we can't test
 * - Since reference kernel uses the exact the same algorithm in the same quantized domain
 *   it is hard to fully test whether the algorithm accomplishes what it is supposed to.
 * - The algorithm has been sensitive to quantization scale but it is hard to fully test
 *   the sensitivity due to aforementioned reason.
 * - Again, it is hard to fully test corner values due to the exact same algorithm of the
 *   reference kernel and the Neon kernel.
 */

constexpr uint32_t qsymm16_per_vector = vector_size_byte / sizeof(int16_t);

#define QSYMM16_DATASET_ITER(num_input_batch, num_iter)                                                              \
    combine(combine(zip(zip(QLSTMLayerNormShapeDataSet<qsymm16_per_vector, num_input_batch, num_iter>("InputShape"), \
                            QLSTMLayerNormShapeDataSet<qsymm16_per_vector, 1, num_iter>("WeightShape")),             \
                        QLSTMLayerNormShapeDataSet<qsymm16_per_vector, 1, num_iter>("BiasShape")),                   \
                    framework::dataset::make("DataType", DataType::QSYMM16)),                                        \
            framework::dataset::make("WeightQuantizationInfo", { QuantizationInfo(1. / 8192), QuantizationInfo(8192) }))

#define QSYMM16_DATASET_1D \
    concat(concat(QSYMM16_DATASET_ITER(1, 0), QSYMM16_DATASET_ITER(1, 1)), QSYMM16_DATASET_ITER(1, 2))

#define QSYMM16_DATASET_2D \
    concat(concat(QSYMM16_DATASET_ITER(3, 0), QSYMM16_DATASET_ITER(3, 1)), QSYMM16_DATASET_ITER(3, 2))

FIXTURE_DATA_TEST_CASE(RandomValue1D, NEQLSTMLayerNormalizationFixture<int16_t>, framework::DatasetMode::ALL, QSYMM16_DATASET_1D)
{
    // Validate output
    validate(Accessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RandomValue2D, NEQLSTMLayerNormalizationFixture<int16_t>, framework::DatasetMode::ALL, QSYMM16_DATASET_2D)
{
    // Validate output
    validate(Accessor(_target), _reference);
}

#undef QSYMM16_DATASET_ITER
#undef QSYMM16_DATASET_2D
#undef QSYMM16_DATASET_1D

TEST_SUITE_END() // QSYMM16
TEST_SUITE_END() // Quantized
TEST_SUITE_END() // QLSTMLayerNormalization
TEST_SUITE_END() // Neon

} // namespace validation
} // namespace test
} // namespace arm_compute
