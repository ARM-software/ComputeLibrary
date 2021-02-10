/*
 * Copyright (c) 2019-2021 Arm Limited.
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
#include "arm_compute/runtime/NEON/functions/NELSTMLayerQuantized.h"

#include "tests/NEON/Accessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/Utils.h"
#include "tests/datasets/LSTMLayerDataset.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"

#include <vector>

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
template <typename T>
inline void fill_tensor(Tensor &tensor, const std::vector<T> &v)
{
    // Import memory accounting for padding
    TensorShape t_shape = tensor.info()->tensor_shape();
    Window      window;
    window.use_tensor_dimensions(t_shape);
    Iterator out(&tensor, window);
    execute_window_loop(window, [&](const Coordinates & id)
    {
        *reinterpret_cast<T *>(out.ptr()) = v[coord2index(t_shape, id)];
    },
    out);
}

template <typename T>
inline void fill_tensor(SimpleTensor<T> &tensor, const std::vector<T> &v)
{
    std::memcpy(tensor.data(), v.data(), sizeof(T) * v.size());
}

/** Tolerance for quantized asymmetric operations */
#if defined(__aarch64__)
constexpr AbsoluteTolerance<int16_t> tolerance_qsymm16(0);
#else  // defined(__aarch64__)
constexpr AbsoluteTolerance<int16_t> tolerance_qsymm16(1);
#endif // defined(__aarch64__)

} // namespace

TEST_SUITE(NEON)
TEST_SUITE(LSTMLayerQuantized)

// *INDENT-OFF*
// clang-format off
TEST_SUITE(IntegrationTestCase)
TEST_SUITE(MultSmallerEq1)
TEST_CASE(RunSmall, framework::DatasetMode::PRECOMMIT)
{
    const int batch_size  = 2;
    const int input_size  = 2;
    const int output_size = 4;


    QuantizationInfo qasymm(1.f / 128.f, 128);
    QuantizationInfo qweights(1.f / 128.f, 128);
    QuantizationInfo qsymm_3(8.f / 32768.f, 0);
    QuantizationInfo qsymm_4(16.f / 32768.f, 0);

    TensorShape input_shape{ input_size, batch_size };
    TensorShape input_weights_shape{ input_size, output_size };
    TensorShape recurrent_weights_shape{ output_size, output_size };
    TensorShape output_shape{ output_size, batch_size};
    TensorShape bias_shape{ output_size };

    auto input_to_input_weights      = create_tensor<Tensor>(input_weights_shape, DataType::QASYMM8, 1, qweights);
    auto input_to_forget_weights     = create_tensor<Tensor>(input_weights_shape, DataType::QASYMM8, 1, qweights);
    auto input_to_cell_weights       = create_tensor<Tensor>(input_weights_shape, DataType::QASYMM8, 1, qweights);
    auto input_to_output_weights     = create_tensor<Tensor>(input_weights_shape, DataType::QASYMM8, 1, qweights);
    auto recurrent_to_input_weights  = create_tensor<Tensor>(recurrent_weights_shape, DataType::QASYMM8, 1, qweights);
    auto recurrent_to_forget_weights = create_tensor<Tensor>(recurrent_weights_shape, DataType::QASYMM8, 1, qweights);
    auto recurrent_to_cell_weights   = create_tensor<Tensor>(recurrent_weights_shape, DataType::QASYMM8, 1, qweights);
    auto recurrent_to_output_weights = create_tensor<Tensor>(recurrent_weights_shape, DataType::QASYMM8, 1, qweights);
    auto input_gate_bias             = create_tensor<Tensor>(bias_shape, DataType::S32);
    auto forget_gate_bias            = create_tensor<Tensor>(bias_shape, DataType::S32);
    auto cell_gate_bias              = create_tensor<Tensor>(bias_shape, DataType::S32);
    auto output_gate_bias            = create_tensor<Tensor>(bias_shape, DataType::S32);

    // LSTM input
    auto input = create_tensor<Tensor>(input_shape, DataType::QASYMM8, 1, qasymm);

    // LSTM output state
    auto output_state = create_tensor<Tensor>(output_shape, DataType::QASYMM8, 1, qasymm);

    // LSTM cell state
    auto cell_state = create_tensor<Tensor>(output_shape, DataType::QSYMM16, 1, qsymm_4);

    NELSTMLayerQuantized lstmq;

    lstmq.configure(&input, &input_to_input_weights, &input_to_forget_weights, &input_to_cell_weights, &input_to_output_weights,
                    &recurrent_to_input_weights, &recurrent_to_forget_weights, &recurrent_to_cell_weights, &recurrent_to_output_weights,
                    &input_gate_bias, &forget_gate_bias, &cell_gate_bias, &output_gate_bias, &cell_state, &output_state, &cell_state, &output_state);

    input.allocator()->allocate();
    input_to_input_weights.allocator()->allocate();
    input_to_forget_weights.allocator()->allocate();
    input_to_cell_weights.allocator()->allocate();
    input_to_output_weights.allocator()->allocate();
    recurrent_to_input_weights.allocator()->allocate();
    recurrent_to_forget_weights.allocator()->allocate();
    recurrent_to_cell_weights.allocator()->allocate();
    recurrent_to_output_weights.allocator()->allocate();
    input_gate_bias.allocator()->allocate();
    forget_gate_bias.allocator()->allocate();
    cell_gate_bias.allocator()->allocate();
    output_gate_bias.allocator()->allocate();
    cell_state.allocator()->allocate();
    output_state.allocator()->allocate();

    // Fill weights and biases
    fill_tensor(input_to_input_weights, std::vector<uint8_t>{ 47,  168,
                                                              66,  239,
                                                               6,   42,
                                                             237,  236 });

    fill_tensor(input_to_forget_weights, std::vector<uint8_t> { 204,  193,
                                                                148,  59,
                                                                113,  17,
                                                                 66, 197 });

    fill_tensor(input_to_cell_weights, std::vector<uint8_t> { 172,  101,
                                                              184, 209,
                                                              165,  82,
                                                              108, 209 });

    fill_tensor(input_to_output_weights, std::vector<uint8_t> { 203, 244,
                                                                219, 114,
                                                                130,  16,
                                                                163, 222 });

    fill_tensor(recurrent_to_input_weights, std::vector<uint8_t> { 162, 168,  7,  95,
                                                                    91, 155, 108, 216,
                                                                   255, 100,  48, 188,
                                                                    58,  37, 186, 147 });

    fill_tensor(recurrent_to_forget_weights, std::vector<uint8_t> {  46,  58,  47, 170,
                                                                    246,  96,  12,  99,
                                                                     68,  23, 186, 161,
                                                                    237, 164,  89,   6 });

    fill_tensor(recurrent_to_cell_weights, std::vector<uint8_t> { 234,  99,   71, 206,
                                                                  205, 159,   64, 253,
                                                                  191, 148,  116,   8,
                                                                  209, 136,   59, 138 });

    fill_tensor(recurrent_to_output_weights, std::vector<uint8_t> {  23, 241, 137, 36,
                                                                    206,   5, 227, 56,
                                                                    254, 176, 231, 47,
                                                                     18, 201, 161, 11 });

    fill_tensor(input_gate_bias, std::vector<int>  {-103038,   30525,  115255, -38154 });
    fill_tensor(forget_gate_bias, std::vector<int> { -23428,  126970,  116806,  46307 });
    fill_tensor(cell_gate_bias, std::vector<int>   { 128006,   69949,  -42808,  42568 });
    fill_tensor(output_gate_bias, std::vector<int> { -67066,  -53607,   47233,  7300  });

    SimpleTensor<uint8_t> expected_output(output_shape, DataType::QASYMM8, 1, qasymm);

    // Initialize state
    fill_tensor(output_state, std::vector<uint8_t> { 128, 128, 128, 128,
                                                     128, 128, 128, 128 });
    fill_tensor(cell_state, std::vector<int16_t> { 0, 0, 0, 0,
                                                   0, 0, 0, 0 });

    // First input
    fill_tensor(input, std::vector<uint8_t> { 106,  193,
                                              155,  150 });

    fill_tensor(expected_output, std::vector<uint8_t> { 128, 130,  36, 134,
                                                        128, 131,  35, 133 });

    lstmq.run();
    validate(Accessor(output_state), expected_output, tolerance_qsymm16);

    // Second input
    fill_tensor(expected_output, std::vector<uint8_t> { 128, 129, 12, 137,
                                                        128, 131, 10, 136 });
    lstmq.run();
    validate(Accessor(output_state), expected_output, tolerance_qsymm16);

    // Third input
    fill_tensor(expected_output, std::vector<uint8_t> { 128, 129, 8, 140,
                                                        128, 130, 6, 138 });
    lstmq.run();
    validate(Accessor(output_state), expected_output, tolerance_qsymm16);
}

TEST_CASE(RunLarge, framework::DatasetMode::PRECOMMIT)
{
    const int batch_size  = 16;
    const int input_size  = 8;
    const int output_size = 8;


    QuantizationInfo qasymm(1.f / 128.f, 128);
    QuantizationInfo qweights(1.f / 128.f, 128);
    QuantizationInfo qsymm_3(8.f / 32768.f, 0);
    QuantizationInfo qsymm_4(16.f / 32768.f, 0);

    TensorShape input_shape{ input_size, batch_size };
    TensorShape input_weights_shape{ input_size, output_size };
    TensorShape recurrent_weights_shape{ output_size, output_size };
    TensorShape output_shape{ output_size, batch_size};
    TensorShape bias_shape{ output_size };

    auto input_to_input_weights      = create_tensor<Tensor>(input_weights_shape, DataType::QASYMM8, 1, qweights);
    auto input_to_forget_weights     = create_tensor<Tensor>(input_weights_shape, DataType::QASYMM8, 1, qweights);
    auto input_to_cell_weights       = create_tensor<Tensor>(input_weights_shape, DataType::QASYMM8, 1, qweights);
    auto input_to_output_weights     = create_tensor<Tensor>(input_weights_shape, DataType::QASYMM8, 1, qweights);
    auto recurrent_to_input_weights  = create_tensor<Tensor>(recurrent_weights_shape, DataType::QASYMM8, 1, qweights);
    auto recurrent_to_forget_weights = create_tensor<Tensor>(recurrent_weights_shape, DataType::QASYMM8, 1, qweights);
    auto recurrent_to_cell_weights   = create_tensor<Tensor>(recurrent_weights_shape, DataType::QASYMM8, 1, qweights);
    auto recurrent_to_output_weights = create_tensor<Tensor>(recurrent_weights_shape, DataType::QASYMM8, 1, qweights);
    auto input_gate_bias             = create_tensor<Tensor>(bias_shape, DataType::S32);
    auto forget_gate_bias            = create_tensor<Tensor>(bias_shape, DataType::S32);
    auto cell_gate_bias              = create_tensor<Tensor>(bias_shape, DataType::S32);
    auto output_gate_bias            = create_tensor<Tensor>(bias_shape, DataType::S32);

    // LSTM input
    auto input = create_tensor<Tensor>(input_shape, DataType::QASYMM8, 1, qasymm);

    // LSTM output state
    auto output_state = create_tensor<Tensor>(output_shape, DataType::QASYMM8, 1, qasymm);

    // LSTM cell state
    auto cell_state = create_tensor<Tensor>(output_shape, DataType::QSYMM16, 1, qsymm_4);

    NELSTMLayerQuantized lstmq;

    lstmq.configure(&input, &input_to_input_weights, &input_to_forget_weights, &input_to_cell_weights, &input_to_output_weights,
                    &recurrent_to_input_weights, &recurrent_to_forget_weights, &recurrent_to_cell_weights, &recurrent_to_output_weights,
                    &input_gate_bias, &forget_gate_bias, &cell_gate_bias, &output_gate_bias, &cell_state, &output_state, &cell_state, &output_state);

    input.allocator()->allocate();
    input_to_input_weights.allocator()->allocate();
    input_to_forget_weights.allocator()->allocate();
    input_to_cell_weights.allocator()->allocate();
    input_to_output_weights.allocator()->allocate();
    recurrent_to_input_weights.allocator()->allocate();
    recurrent_to_forget_weights.allocator()->allocate();
    recurrent_to_cell_weights.allocator()->allocate();
    recurrent_to_output_weights.allocator()->allocate();
    input_gate_bias.allocator()->allocate();
    forget_gate_bias.allocator()->allocate();
    cell_gate_bias.allocator()->allocate();
    output_gate_bias.allocator()->allocate();
    cell_state.allocator()->allocate();
    output_state.allocator()->allocate();

    // Fill weights and biases
    fill_tensor(input_to_input_weights, std::vector<uint8_t>{ 141,  89, 200, 180,  46,  50,  87, 128,
                                                              149, 227, 177, 187, 212, 229,  54, 111,
                                                              131, 116,   3,  58, 196,  26, 131, 255,
                                                               22, 106, 216,  69, 239,  12, 232, 207,
                                                              184,  56, 236, 172,  28, 143, 161, 124,
                                                              255,  33, 197, 122,  47, 197,  26, 229,
                                                               91,  79,  11, 160,  26,  80, 100,  36,
                                                              248, 186,  97,  61, 125,  46,  14, 100, });

    fill_tensor(input_to_forget_weights, std::vector<uint8_t> { 237, 165, 141, 249,  72, 116, 36 , 115,
                                                                234, 213,  85,  84,  59,  62, 150, 246,
                                                                182, 102, 158, 214, 182, 183,  94,  11,
                                                                158, 192,  92, 189, 160, 219, 206, 249,
                                                                 88, 213, 193, 244, 151,  72, 129,  49,
                                                                239,  83, 106,   9, 169, 187, 125, 171,
                                                                 32, 141, 126,  92,  13,  36, 224, 150,
                                                                187, 250, 178, 169,  89, 214,  91, 173 });

    fill_tensor(input_to_cell_weights, std::vector<uint8_t> {  93, 103, 226, 139, 185, 252, 129, 171,
                                                              159,  32,  25, 175, 224, 183, 165,  35,
                                                              207,  69, 238, 228, 149, 214,  79,   6,
                                                                5,  66, 102,  14,  19, 111,  36, 143,
                                                               22,  85,  13,  78, 236, 121, 122,  77,
                                                              249,  39,  88,  12, 205, 143,  93, 240,
                                                              167,  89, 188,  50,  73,  69, 201, 251,
                                                               59,  32, 203, 184, 139, 191, 199,  74});

    fill_tensor(input_to_output_weights, std::vector<uint8_t> { 205,   7,  95, 104, 252, 143, 226,  73,
                                                                229, 114, 152, 171, 221, 153,  73, 229,
                                                                153, 165, 223, 239, 100,  38, 172, 211,
                                                                226, 133, 239, 207, 116, 230, 170, 100,
                                                                241,  95, 171, 124,  63, 115,  32, 127,
                                                                141, 239,  53, 193, 201,  53, 104, 178,
                                                                186, 212, 167, 107, 226, 230,  71, 213,
                                                                148, 217,  19, 248, 233, 195, 183, 156 });

    fill_tensor(recurrent_to_input_weights, std::vector<uint8_t> { 147, 112, 140, 103,   3, 255,  17,  49,
                                                                    84, 112, 144, 213, 138, 142, 112,  66,
                                                                   117,  30, 101,  35,  25, 132, 211, 229,
                                                                   183, 208, 102,  16,  38,  85, 101, 152,
                                                                   226,  83, 132,  22, 161, 110, 157, 129,
                                                                   184,  63, 168,  42, 220, 126, 209, 157,
                                                                     5,  88, 243,  83, 249,  19, 226, 209,
                                                                   173,  96, 185,  77, 146, 227, 238, 136 });


    fill_tensor(recurrent_to_forget_weights, std::vector<uint8_t> {  52, 132,  92, 200, 213,  32, 213,  37,
                                                                    116, 142, 116, 180,   4, 172, 158, 143,
                                                                    110,  40,  99,  28, 221, 153, 133,   2,
                                                                    247, 144, 198, 100,  20,  15, 221, 196,
                                                                    159, 178, 188, 151, 171,  15,  25, 217,
                                                                    178, 109, 110, 118, 128,  39, 232, 234,
                                                                    184, 214, 177,  13,  56,   6,  28, 252,
                                                                     89, 187, 242,  59, 146, 111, 132, 129});

    fill_tensor(recurrent_to_cell_weights, std::vector<uint8_t> {  70,  44, 137,  29,  36, 127,   1, 241,
                                                                   26, 241, 142, 114,  67, 181,  49,  57,
                                                                  131, 152, 175,  77,  23,  63,  37, 124,
                                                                  150, 113,  95, 103, 110, 201,  69,  97,
                                                                  196, 242,  62, 214,  66,  19,  45, 135,
                                                                   22, 168, 149, 104,  77, 101,  36,  68,
                                                                  170, 116, 222, 100, 109,   1, 154,  18,
                                                                  133, 215, 105,  93,  31,  57, 231, 112 });


    fill_tensor(recurrent_to_output_weights, std::vector<uint8_t> { 45 ,  181 ,  220 ,  219 ,   49  ,  63 ,   49  , 129,
                                                                     7 ,  166 ,  104 ,  114 ,   83  ,  40 ,    1  , 195,
                                                                   245 ,  142 ,   82 ,  232 ,  104  , 245 ,   82  , 196,
                                                                   111 ,   56 ,  156 ,    9 ,  141  , 240 ,  180  , 148,
                                                                   247 ,  198 ,  234 ,  137 ,   13  , 210 ,  161  , 192,
                                                                   196 ,   59 ,  233 ,  184 ,  142  , 187 ,  140  , 166,
                                                                     2 ,   95 ,  152 ,   46 ,   71  ,  46 ,  113  ,  32,
                                                                   175 ,  229 ,   86 ,   87 ,   62  ,  93 ,   74  , 130});

    fill_tensor(input_gate_bias, std::vector<int>  {  -40040, -106916,  -92315,  -79123,   45160, -17954,   50962, -63758 });
    fill_tensor(forget_gate_bias, std::vector<int> { -128514,    8463,  -57831,  116977,  106547, -28132, -124557,  44941 });
    fill_tensor(cell_gate_bias, std::vector<int>   { 88388  ,  123601, -116148,  -13022,   21619,  48926,   57523,  39332 });
    fill_tensor(output_gate_bias, std::vector<int> {  59485 ,  -33070,   21386, -100633, -115959, 125768,  -56407,  24897 });

    SimpleTensor<uint8_t> expected_output(output_shape, DataType::QASYMM8, 1, qasymm);

    // Initialize state
    fill_tensor(output_state, std::vector<uint8_t> { 128, 128, 128, 128, 128, 128, 128, 128,
                                                     128, 128, 128, 128, 128, 128, 128, 128,
                                                     128, 128, 128, 128, 128, 128, 128, 128,
                                                     128, 128, 128, 128, 128, 128, 128, 128,
                                                     128, 128, 128, 128, 128, 128, 128, 128,
                                                     128, 128, 128, 128, 128, 128, 128, 128,
                                                     128, 128, 128, 128, 128, 128, 128, 128,
                                                     128, 128, 128, 128, 128, 128, 128, 128,
                                                     128, 128, 128, 128, 128, 128, 128, 128,
                                                     128, 128, 128, 128, 128, 128, 128, 128,
                                                     128, 128, 128, 128, 128, 128, 128, 128,
                                                     128, 128, 128, 128, 128, 128, 128, 128,
                                                     128, 128, 128, 128, 128, 128, 128, 128,
                                                     128, 128, 128, 128, 128, 128, 128, 128,
                                                     128, 128, 128, 128, 128, 128, 128, 128,
                                                     128, 128, 128, 128, 128, 128, 128, 128 });

    fill_tensor(cell_state, std::vector<int16_t> { 0, 0, 0, 0, 0, 0, 0, 0,
                                                   0, 0, 0, 0, 0, 0, 0, 0,
                                                   0, 0, 0, 0, 0, 0, 0, 0,
                                                   0, 0, 0, 0, 0, 0, 0, 0,
                                                   0, 0, 0, 0, 0, 0, 0, 0,
                                                   0, 0, 0, 0, 0, 0, 0, 0,
                                                   0, 0, 0, 0, 0, 0, 0, 0,
                                                   0, 0, 0, 0, 0, 0, 0, 0,
                                                   0, 0, 0, 0, 0, 0, 0, 0,
                                                   0, 0, 0, 0, 0, 0, 0, 0,
                                                   0, 0, 0, 0, 0, 0, 0, 0,
                                                   0, 0, 0, 0, 0, 0, 0, 0,
                                                   0, 0, 0, 0, 0, 0, 0, 0,
                                                   0, 0, 0, 0, 0, 0, 0, 0,
                                                   0, 0, 0, 0, 0, 0, 0, 0,
                                                   0, 0, 0, 0, 0, 0, 0, 0});

    // First input
    fill_tensor(input, std::vector<uint8_t> { 247,  203, 159, 131, 182, 114, 207, 195,
                                              48 ,  61 , 154,  16,  80, 101, 116, 255,
                                              50 , 115 ,  45, 186,  75, 212,  98,  48,
                                              88 , 146 ,  24, 143, 218, 174, 203, 200,
                                             239 ,  16 ,  66, 136, 234,  54,  94,  51,
                                             101 , 128 , 220, 213, 164,  82, 137, 255,
                                              70 , 165 , 234, 220,  66,  35, 183, 206,
                                              39 ,  57 , 180, 202,  23, 172, 224, 109,
                                             102 , 215 , 186,  82, 215, 147,  85, 187,
                                              96 , 249 ,  59, 116, 150,  44, 167, 128,
                                              34 , 217 , 148, 193, 243,  38, 250, 208,
                                             112 , 130 , 208,  29,  16, 122,  20,  92,
                                              24 ,  72 , 104,  29, 150, 233, 151,  19,
                                             158 , 192 , 254,  70,  73, 142, 106, 152,
                                               3 ,  61 ,  24, 135, 212,   9,  80, 234,
                                             147 , 246 ,  83, 249,  49,  14,  68,  50});

    fill_tensor(expected_output, std::vector<uint8_t> {131, 128,  128,  128,  128,  180,  129,  133,
                                                       136, 128,  126,  128,  128,  173,  135,  130,
                                                       160, 128,  128,  128,  128,  138,  132,  129,
                                                       131, 128,  127,  128,  128,  169,  129,  131,
                                                       133, 128,  128,  128,  128,  182,  130,  129,
                                                       131, 128,  128,  128,  128,  163,  129,  130,
                                                       131, 128,  128,  128,  128,  149,  132,  129,
                                                       143, 128,  127,  128,  128,  150,  134,  131,
                                                       134, 128,  128,  128,  128,  167,  130,  130,
                                                       131, 128,  128,  128,  128,  152,  132,  129,
                                                       128, 128,  128,  128,  128,  169,  130,  130,
                                                       173, 128,  128,  128,  128,  148,  139,  130,
                                                       152, 128,  128,  128,  128,  168,  139,  132,
                                                       147, 128,  128,  128,  128,  161,  131,  132,
                                                       130, 128,  128,  128,  128,  159,  134,  128,
                                                       140, 128,  128,  128,  128,  133,  132,  128 });

    lstmq.run();
    validate(Accessor(output_state), expected_output, tolerance_qsymm16);

    // Second input
    fill_tensor(expected_output, std::vector<uint8_t> { 130,   128,   128,   128,   128,   205,   129,   137,
                                                        135,   128,   127,   128,   128,   190,   137,   132,
                                                        160,   128,   128,   128,   128,   142,   133,   131,
                                                        130,   128,   128,   128,   128,   185,   129,   133,
                                                        132,   128,   128,   128,   128,   198,   131,   130,
                                                        130,   128,   128,   128,   128,   178,   130,   131,
                                                        131,   128,   128,   128,   128,   158,   132,   131,
                                                        142,   128,   127,   128,   128,   158,   135,   134,
                                                        133,   128,   128,   128,   128,   178,   131,   132,
                                                        131,   128,   128,   128,   128,   160,   132,   130,
                                                        128,   128,   128,   128,   128,   190,   131,   131,
                                                        170,   128,   128,   128,   128,   157,   142,   131,
                                                        149,   128,   128,   128,   128,   178,   142,   135,
                                                        145,   128,   128,   128,   129,   173,   132,   135,
                                                        129,   128,   128,   128,   128,   171,   134,   129,
                                                        140,   128,   128,   128,   128,   135,   132,   129});
    lstmq.run();
    validate(Accessor(output_state), expected_output, tolerance_qsymm16);
}
TEST_SUITE_END() // MultSmallerEq1

TEST_SUITE(MultGreater1)
TEST_CASE(RunSmall, framework::DatasetMode::PRECOMMIT)
{
    //Input sequence length is 1
    const int batch_size  = 2;
    const int input_size  = 2;
    const int output_size = 4;

    QuantizationInfo qasymm(1.f / 128.f, 128);
    QuantizationInfo qweights(1.f / 16.f, 16);
    QuantizationInfo qsymm_3(8.f / 32768.f, 0);
    QuantizationInfo qsymm_4(16.f / 32768.f, 0);

    TensorShape input_shape{ input_size, batch_size };
    TensorShape input_weights_shape{ input_size, output_size };
    TensorShape recurrent_weights_shape{ output_size, output_size };
    TensorShape output_shape{ output_size, batch_size};
    TensorShape bias_shape{ output_size };

    auto input_to_input_weights      = create_tensor<Tensor>(input_weights_shape, DataType::QASYMM8, 1, qweights);
    auto input_to_forget_weights     = create_tensor<Tensor>(input_weights_shape, DataType::QASYMM8, 1, qweights);
    auto input_to_cell_weights       = create_tensor<Tensor>(input_weights_shape, DataType::QASYMM8, 1, qweights);
    auto input_to_output_weights     = create_tensor<Tensor>(input_weights_shape, DataType::QASYMM8, 1, qweights);
    auto recurrent_to_input_weights  = create_tensor<Tensor>(recurrent_weights_shape, DataType::QASYMM8, 1, qweights);
    auto recurrent_to_forget_weights = create_tensor<Tensor>(recurrent_weights_shape, DataType::QASYMM8, 1, qweights);
    auto recurrent_to_cell_weights   = create_tensor<Tensor>(recurrent_weights_shape, DataType::QASYMM8, 1, qweights);
    auto recurrent_to_output_weights = create_tensor<Tensor>(recurrent_weights_shape, DataType::QASYMM8, 1, qweights);
    auto input_gate_bias             = create_tensor<Tensor>(bias_shape, DataType::S32);
    auto forget_gate_bias            = create_tensor<Tensor>(bias_shape, DataType::S32);
    auto cell_gate_bias              = create_tensor<Tensor>(bias_shape, DataType::S32);
    auto output_gate_bias            = create_tensor<Tensor>(bias_shape, DataType::S32);

    // LSTM input
    auto input = create_tensor<Tensor>(input_shape, DataType::QASYMM8, 1, qasymm);

    // LSTM output state
    auto output_state = create_tensor<Tensor>(output_shape, DataType::QASYMM8, 1, qasymm);

    // LSTM cell state
    auto cell_state = create_tensor<Tensor>(output_shape, DataType::QSYMM16, 1, qsymm_4);

    NELSTMLayerQuantized lstmq;

    lstmq.configure(&input, &input_to_input_weights, &input_to_forget_weights, &input_to_cell_weights, &input_to_output_weights,
                    &recurrent_to_input_weights, &recurrent_to_forget_weights, &recurrent_to_cell_weights, &recurrent_to_output_weights,
                    &input_gate_bias, &forget_gate_bias, &cell_gate_bias, &output_gate_bias, &cell_state, &output_state, &cell_state, &output_state);

    input.allocator()->allocate();
    input_to_input_weights.allocator()->allocate();
    input_to_forget_weights.allocator()->allocate();
    input_to_cell_weights.allocator()->allocate();
    input_to_output_weights.allocator()->allocate();
    recurrent_to_input_weights.allocator()->allocate();
    recurrent_to_forget_weights.allocator()->allocate();
    recurrent_to_cell_weights.allocator()->allocate();
    recurrent_to_output_weights.allocator()->allocate();
    input_gate_bias.allocator()->allocate();
    forget_gate_bias.allocator()->allocate();
    cell_gate_bias.allocator()->allocate();
    output_gate_bias.allocator()->allocate();
    cell_state.allocator()->allocate();
    output_state.allocator()->allocate();

    // Fill weights and biases
    fill_tensor(input_to_input_weights, std::vector<uint8_t>{ 122,  130,
                                                              124,  134,
                                                               120,   122,
                                                             134,  134 });

    fill_tensor(input_to_forget_weights, std::vector<uint8_t> { 204,  193,
                                                                148,  59,
                                                                113,  17,
                                                                 66, 197 });

    fill_tensor(input_to_cell_weights, std::vector<uint8_t> { 172,  101,
                                                              184, 209,
                                                              165,  82,
                                                              108, 209 });

    fill_tensor(input_to_output_weights, std::vector<uint8_t> { 203, 244,
                                                                219, 114,
                                                                130,  16,
                                                                163, 222 });

    fill_tensor(recurrent_to_input_weights, std::vector<uint8_t> { 162, 168,  7,  95,
                                                                    91, 155, 108, 216,
                                                                   255, 100,  48, 188,
                                                                    58,  37, 186, 147 });

    fill_tensor(recurrent_to_forget_weights, std::vector<uint8_t> {  46,  58,  47, 170,
                                                                    246,  96,  12,  99,
                                                                     68,  23, 186, 161,
                                                                    237, 164,  89,   6 });

    fill_tensor(recurrent_to_cell_weights, std::vector<uint8_t> { 234,  99,   71, 206,
                                                                  205, 159,   64, 253,
                                                                  191, 148,  116,   8,
                                                                  209, 136,   59, 138 });

    fill_tensor(recurrent_to_output_weights, std::vector<uint8_t> {  23, 241, 137, 36,
                                                                    206,   5, 227, 56,
                                                                    254, 176, 231, 47,
                                                                     18, 201, 161, 11 });

    fill_tensor(input_gate_bias, std::vector<int>  {-103038,   30525,  115255, -38154 });
    fill_tensor(forget_gate_bias, std::vector<int> { -23428,  126970,  116806,  46307 });
    fill_tensor(cell_gate_bias, std::vector<int>   { 128006,   69949,  -42808,  42568 });
    fill_tensor(output_gate_bias, std::vector<int> { -67066,  -53607,   47233,  7300  });

    SimpleTensor<uint8_t> expected_output(output_shape, DataType::QASYMM8, 1, qasymm);

    // Initialize state
    fill_tensor(output_state, std::vector<uint8_t> { 128, 128, 128, 128,
                                                     128, 128, 128, 128 });
    fill_tensor(cell_state, std::vector<int16_t> { 0, 0, 0, 0,
                                                   0, 0, 0, 0 });

    // First input
    fill_tensor(input, std::vector<uint8_t> { 106,  193,
                                              155,  150 });

    fill_tensor(expected_output, std::vector<uint8_t> { 128, 128,  31, 128,
                                                        128, 128,  31, 128 });

    lstmq.run();
    validate(Accessor(output_state), expected_output);

    // Second input
    fill_tensor(expected_output, std::vector<uint8_t> { 128, 128, 5, 128,
                                                        128, 128, 5, 128 });
    lstmq.run();
    validate(Accessor(output_state), expected_output);

    // Third input
    fill_tensor(expected_output, std::vector<uint8_t> { 128, 128, 1, 128,
                                                        128, 128, 1, 128, });
    lstmq.run();
    validate(Accessor(output_state), expected_output);
}
TEST_SUITE_END() // MultGreater1
TEST_SUITE_END() // IntegrationTestCase
// clang-format on
// *INDENT-ON*

TEST_SUITE_END() // LSTMLayerQuantized
TEST_SUITE_END() // Neon
} // namespace validation
} // namespace test
} // namespace arm_compute
