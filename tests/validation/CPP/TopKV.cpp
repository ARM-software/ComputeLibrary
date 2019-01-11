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
#include "arm_compute/runtime/CPP/functions/CPPTopKV.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "tests/NEON/Accessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/PermuteFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
template <typename U, typename T>
inline void fill_tensor(U &&tensor, const std::vector<T> &v)
{
    std::memcpy(tensor.data(), v.data(), sizeof(T) * v.size());
}
} // namespace

TEST_SUITE(CPP)
TEST_SUITE(TopKV)

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(zip(
        framework::dataset::make("PredictionsInfo", { TensorInfo(TensorShape(20, 10), 1, DataType::F32),
                                                TensorInfo(TensorShape(10, 20), 1, DataType::F16),  // Mismatching batch_size
                                                TensorInfo(TensorShape(20, 10), 1, DataType::S8), // Unsupported data type
                                                TensorInfo(TensorShape(10, 10, 10), 1, DataType::F32), // Wrong predictions dimensions
                                                TensorInfo(TensorShape(20, 10), 1, DataType::F32)}), // Wrong output dimension
        framework::dataset::make("TargetsInfo",{ TensorInfo(TensorShape(10), 1, DataType::U32),
                                                TensorInfo(TensorShape(10), 1, DataType::U32),
                                                TensorInfo(TensorShape(10), 1, DataType::U32),
                                                TensorInfo(TensorShape(10), 1, DataType::U32),
                                                TensorInfo(TensorShape(10), 1, DataType::U32)})),
        framework::dataset::make("OutputInfo",{ TensorInfo(TensorShape(10), 1, DataType::U8),
                                                TensorInfo(TensorShape(10), 1, DataType::U8),
                                                TensorInfo(TensorShape(10), 1, DataType::U8),
                                                TensorInfo(TensorShape(10), 1, DataType::U8),
                                                TensorInfo(TensorShape(1), 1, DataType::U8)})),

        framework::dataset::make("k",{ 0, 1, 2, 3, 4 })),
        framework::dataset::make("Expected", {true, false, false, false, false })),
        prediction_info, targets_info, output_info, k, expected)
{
    const Status status = CPPTopKV::validate(&prediction_info.clone()->set_is_resizable(false),&targets_info.clone()->set_is_resizable(false), &output_info.clone()->set_is_resizable(false), k);
    ARM_COMPUTE_EXPECT(bool(status) == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

TEST_CASE(Float, framework::DatasetMode::ALL)
{
    const unsigned int k = 5;

    Tensor predictions = create_tensor<Tensor>(TensorShape(10, 20), DataType::F32);
    Tensor targets     = create_tensor<Tensor>(TensorShape(20), DataType::U32);

    predictions.allocator()->allocate();
    targets.allocator()->allocate();

    // Fill the tensors with random pre-generated values
    fill_tensor(Accessor(predictions), std::vector<float>
    {
        0.8147, 0.6557, 0.4387, 0.7513, 0.3517, 0.1622, 0.1067, 0.8530, 0.7803, 0.5470,
        0.9058, 0.0357, 0.3816, 0.2551, 0.8308, 0.7943, 0.9619, 0.6221, 0.3897, 0.2963,
        0.1270, 0.8491, 0.7655, 0.5060, 0.5853, 0.3112, 0.0046, 0.3510, 0.2417, 0.7447,
        0.9134, 0.9340, 0.7952, 0.6991, 0.5497, 0.5285, 0.7749, 0.5132, 0.4039, 0.1890,
        0.6324, 0.6787, 0.1869, 0.8909, 0.9172, 0.1656, 0.8173, 0.4018, 0.0965, 0.6868,
        0.0975, 0.7577, 0.4898, 0.9593, 0.2858, 0.6020, 0.8687, 0.0760, 0.1320, 0.1835,
        0.2785, 0.7431, 0.4456, 0.5472, 0.7572, 0.2630, 0.0844, 0.2399, 0.9421, 0.3685,
        0.5469, 0.3922, 0.6463, 0.1386, 0.7537, 0.6541, 0.3998, 0.1233, 0.9561, 0.6256,
        0.9575, 0.6555, 0.7094, 0.1493, 0.3804, 0.6892, 0.2599, 0.1839, 0.5752, 0.7802,
        0.9649, 0.1712, 0.7547, 0.2575, 0.5678, 0.7482, 0.8001, 0.2400, 0.0598, 0.0811,
        0.1576, 0.7060, 0.2760, 0.8407, 0.0759, 0.4505, 0.4314, 0.4173, 0.2348, 0.9294,
        0.9706, 0.0318, 0.6797, 0.2543, 0.0540, 0.0838, 0.9106, 0.0497, 0.3532, 0.7757,
        0.9572, 0.2769, 0.6551, 0.8143, 0.5308, 0.2290, 0.1818, 0.9027, 0.8212, 0.4868,
        0.4854, 0.0462, 0.1626, 0.2435, 0.7792, 0.9133, 0.2638, 0.9448, 0.0154, 0.4359,
        0.8003, 0.0971, 0.1190, 0.9293, 0.9340, 0.1524, 0.1455, 0.4909, 0.0430, 0.4468,
        0.1419, 0.8235, 0.4984, 0.3500, 0.1299, 0.8258, 0.1361, 0.4893, 0.1690, 0.3063,
        0.4218, 0.6948, 0.9597, 0.1966, 0.5688, 0.5383, 0.8693, 0.3377, 0.6491, 0.5085,
        0.9157, 0.3171, 0.3404, 0.2511, 0.4694, 0.9961, 0.5797, 0.9001, 0.7317, 0.5108,
        0.7922, 0.9502, 0.5853, 0.6160, 0.0119, 0.0782, 0.5499, 0.3692, 0.6477, 0.8176,
        0.9595, 0.0344, 0.2238, 0.4733, 0.3371, 0.4427, 0.1450, 0.1112, 0.4509, 0.7948
    });

    fill_tensor(Accessor(targets), std::vector<int> { 1, 5, 7, 2, 8, 1, 2, 1, 2, 4, 3, 9, 4, 1, 9, 9, 4, 1, 2, 4 });

    // Determine the output through the CPP kernel
    Tensor   output;
    CPPTopKV topkv;
    topkv.configure(&predictions, &targets, &output, k);

    output.allocator()->allocate();

    // Run the kernel
    topkv.run();

    // Validate against the expected values
    SimpleTensor<uint8_t> expected_output(TensorShape(20), DataType::U8);
    fill_tensor(expected_output, std::vector<uint8_t> { 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0 });
    validate(Accessor(output), expected_output);
}

TEST_CASE(Quantized, framework::DatasetMode::ALL)
{
    const unsigned int k = 5;

    Tensor predictions = create_tensor<Tensor>(TensorShape(10, 20), DataType::QASYMM8, 1, QuantizationInfo());
    Tensor targets     = create_tensor<Tensor>(TensorShape(20), DataType::U32);

    predictions.allocator()->allocate();
    targets.allocator()->allocate();

    // Fill the tensors with random pre-generated values
    fill_tensor(Accessor(predictions), std::vector<uint8_t>
    {
        133, 235, 69, 118, 140, 179, 189, 203, 137, 157,
        242, 1, 196, 170, 166, 25, 102, 244, 24, 254,
        164, 119, 49, 198, 140, 135, 175, 84, 29, 136,
        246, 109, 74, 90, 185, 136, 181, 172, 35, 123,
        62, 118, 24, 170, 134, 221, 114, 113, 174, 206,
        174, 198, 148, 107, 255, 125, 6, 214, 127, 59,
        75, 83, 175, 216, 56, 101, 85, 197, 49, 128,
        172, 201, 140, 214, 28, 172, 109, 43, 127, 231,
        178, 121, 109, 66, 29, 190, 70, 221, 38, 148,
        18, 10, 165, 158, 17, 134, 51, 254, 15, 217,
        66, 46, 166, 150, 104, 90, 211, 132, 218, 190,
        58, 185, 174, 139, 115, 39, 111, 227, 144, 151,
        171, 122, 163, 223, 94, 151, 228, 151, 238, 64,
        217, 40, 242, 68, 196, 68, 101, 40, 179, 171,
        89, 88, 54, 82, 161, 12, 197, 52, 150, 22,
        200, 156, 182, 31, 198, 194, 102, 105, 209, 161,
        173, 50, 61, 241, 239, 63, 207, 192, 226, 170,
        2, 190, 31, 166, 250, 114, 194, 212, 254, 187,
        155, 63, 156, 123, 50, 177, 97, 203, 1, 229,
        100, 235, 116, 164, 36, 92, 56, 82, 222, 252
    });

    fill_tensor(Accessor(targets), std::vector<int> { 1, 5, 7, 2, 8, 1, 2, 1, 2, 4, 3, 9, 4, 1, 9, 9, 4, 1, 2, 4 });

    // Determine the output through the CPP kernel
    Tensor   output;
    CPPTopKV topkv;
    topkv.configure(&predictions, &targets, &output, k);

    output.allocator()->allocate();

    // Run the kernel
    topkv.run();

    // Validate against the expected values
    SimpleTensor<uint8_t> expected_output(TensorShape(20), DataType::U8);
    fill_tensor(expected_output, std::vector<uint8_t> { 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0 });
    validate(Accessor(output), expected_output);
}

TEST_SUITE_END() // TopKV
TEST_SUITE_END() // CPP
} // namespace validation
} // namespace test
} // namespace arm_compute
