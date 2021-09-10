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
#include "arm_compute/core/WindowIterator.h"
#include "arm_compute/core/utils/quantization/AsymmHelpers.h"
#include "arm_compute/runtime/NEON/NEFunctions.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "support/ToolchainSupport.h"
#include "utils/Utils.h"

#include <cstdlib>

using namespace arm_compute;
using namespace utils;

// Find min and max value in a float array
void find_min_max(int size, const float *data, float *min, float *max)
{
    *min = *max = data[0];
    for(int i = 0; i < size; i++)
    {
        const float val = data[i];
        *min            = std::min(*min, val);
        *max            = std::max(*max, val);
    }
}

// Return reasonable quantisation parameters to use for an array of floats
// based on min and max values
QuantizationInfo choose_quantization_params(float min, float max)
{
    // Extend the [min,max] interval to contain 0 so we can represent it exactly
    min = std::min(min, 0.f);
    max = std::max(max, 0.f);

    // Set the quantized min and max in float values
    const float qmin = 0;
    const float qmax = 255;

    // Determine the scale
    const float scale = (max - min) / (qmax - qmin);

    // Determine the zero-point; using affine equation val = (qval-zerop) * scale
    const float zero_point_real = qmin - min / scale;

    // But we need to nudge the zero_point to an integer (exact quantized value)
    std::uint8_t zero_point_nudged = 0;
    if(zero_point_real < qmin)
    {
        zero_point_nudged = qmin;
    }
    else if(zero_point_real > qmax)
    {
        zero_point_nudged = qmax;
    }
    else
    {
        zero_point_nudged = static_cast<std::uint8_t>(support::cpp11::round(zero_point_real));
    }

    QuantizationInfo qinfo = QuantizationInfo(scale, zero_point_nudged);
    return qinfo;
}

void quantize_values(int size, qasymm8_t *output, float *input, const QuantizationInfo qinfo)
{
    for(int i = 0; i < size; i++)
    {
        output[i] = quantize_qasymm8(input[i], qinfo);
    }
    std::cout << "\n";
}

int main(int argc, char **argv)
{
    Tensor src1;
    Tensor src2;
    Tensor dst0;
    Tensor q_src1;
    Tensor q_src2;
    Tensor q_dst0;
    Tensor q_res;
    Tensor q_res_output;
    size_t M             = 4;
    size_t N             = 4;
    size_t K             = 4;
    bool   default_input = true;

    // Parse args
    if(argc < 3) /* case default matrix sizes */
    {
        // Print help
        std::cout << "Usage: ./build/neon_gemm_qasymm8 M N K\n";
        std::cout << "Too few or no inputs provided. Using default M=4, N=4, K=4\n\n";
    }
    else /* case M N K arguments provided */
    {
        M             = strtol(argv[1], nullptr, 10);
        N             = strtol(argv[2], nullptr, 10);
        K             = strtol(argv[3], nullptr, 10);
        default_input = false;
    }

    /*** Floating point matrix multiplication ***/

    // Initialise input matrices
    NEGEMM fgemm{};

    src1.allocator()->init(TensorInfo(TensorShape(K, M), 1, DataType::F32));
    src2.allocator()->init(TensorInfo(TensorShape(N, K), 1, DataType::F32));
    dst0.allocator()->init(TensorInfo(TensorShape(N, M), 1, DataType::F32));
    fgemm.configure(&src1, &src2, nullptr, &dst0, 1, 0);

    // Allocate matrices
    src1.allocator()->allocate();
    src2.allocator()->allocate();
    dst0.allocator()->allocate();

    // Fill in tensors, by default fill in with known data - for easy testing
    auto *src1_ptr = reinterpret_cast<float *>(src1.buffer());
    auto *src2_ptr = reinterpret_cast<float *>(src2.buffer());
    auto *dst0_ptr = reinterpret_cast<float *>(dst0.buffer());

    // Fill in: one is the identity matrix, other is sequential values
    // src1: Identity matrix
    for(size_t i = 0; i < M * K; i++)
    {
        src1_ptr[i] = 0;
    }
    for(size_t i = 0; i < M; i++)
    {
        src1_ptr[i * K + i] = 1.0f;
    }

    // src2: Sequential values matrix
    for(size_t i = 0; i < K * N; i++)
    {
        src2_ptr[i] = i * 1.123f;
    }

    // Otherwise if M, N, K is given, fill in with random values
    if(!default_input)
    {
        fill_random_tensor(src1, 0.f, 1.f);
        fill_random_tensor(src2, 0.f, 1.f);
    }

    // Run single precision gemm and print result
    fgemm.run();

#if ARM_COMPUTE_DEBUG_ENABLED
    std::cout << "Result matrix:\n";
    src1.print(std::cout);
    src2.print(std::cout);
    dst0.print(std::cout);
#endif // ARM_COMPUTE_DEBUG_ENABLED

    /*** Quantised asymmetric 8bit matrix  multiplication ***/

    // Start by finding the quantisation parameters for each set of values
    float src1_min;
    float src1_max;
    float src2_min;
    float src2_max;
    float dst0_min;
    float dst0_max;

    find_min_max(M * K, src1_ptr, &src1_min, &src1_max);
    find_min_max(K * N, src2_ptr, &src2_min, &src2_max);
    find_min_max(M * N, dst0_ptr, &dst0_min, &dst0_max);

    const QuantizationInfo src1_qinfo = choose_quantization_params(src1_min, src1_max);
    const QuantizationInfo src2_qinfo = choose_quantization_params(src2_min, src2_max);
    const QuantizationInfo dst0_qinfo = choose_quantization_params(dst0_min, dst0_max);

    std::cout << "Matrix 1: min=" << src1_min << ", max=" << src1_max << ", ";
    std::cout << "QuantisationInfo(" << src1_qinfo.scale()[0] << ", " << src1_qinfo.offset()[0] << ")\n";
    std::cout << "Matrix 2: min=" << src2_min << ", max=" << src2_max << ", ";
    std::cout << "QuantisationInfo(" << src2_qinfo.scale()[0] << ", " << src2_qinfo.offset()[0] << ")\n";
    std::cout << "Result  : min=" << dst0_min << ", max=" << dst0_max << ", ";
    std::cout << "QuantisationInfo(" << dst0_qinfo.scale()[0] << ", " << dst0_qinfo.offset()[0] << ")\n";

    // We now have the quantisation info and can configure the quantised tensors
    q_src1.allocator()->init(TensorInfo(TensorShape(K, M), 1, DataType::QASYMM8, src1_qinfo));
    q_src2.allocator()->init(TensorInfo(TensorShape(N, K), 1, DataType::QASYMM8, src2_qinfo));
    q_dst0.allocator()->init(TensorInfo(TensorShape(N, M), 1, DataType::QASYMM8, dst0_qinfo));

    // In this approach we use the QuantizationLayer construct to perform quantization
    NEQuantizationLayer q1;
    NEQuantizationLayer q2;
    NEQuantizationLayer q3;
    q1.configure(&src1, &q_src1);
    q2.configure(&src2, &q_src2);
    q3.configure(&dst0, &q_dst0);

    // Configure low precision gemm and initialise result tensor (pre-output)
    NEGEMMLowpMatrixMultiplyCore qgemm;
    q_res.allocator()->init(TensorInfo(TensorShape(N, M), 1, DataType::S32));
    qgemm.configure(&q_src1, &q_src2, nullptr, &q_res);

    // Configure output stage after computing shift and multiplier parameters
    NEGEMMLowpOutputStage gemmlowp_output_stage;
    int                   output_multiplier;
    int                   output_shift;
    float                 multiplier = (src1_qinfo.uniform().scale * src2_qinfo.uniform().scale) / dst0_qinfo.uniform().scale;
    quantization::calculate_quantized_multiplier_less_than_one(multiplier, &output_multiplier, &output_shift);
    std::cout << "(q_multiplier, q_shift) = (" << output_multiplier << ", " << output_shift << ")\n\n";

    GEMMLowpOutputStageInfo info;
    info.type                = GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT;
    info.gemmlowp_multiplier = output_multiplier;
    info.gemmlowp_shift      = output_shift;
    info.gemmlowp_offset     = dst0_qinfo.uniform().offset;
    info.output_data_type    = DataType::QASYMM8;
    q_res_output.info()->set_data_type(DataType::QASYMM8);
    q_res_output.info()->set_num_channels(1);
    gemmlowp_output_stage.configure(&q_res, nullptr, &q_res_output, info);

    // Allocate all tensors
    q_src1.allocator()->allocate();
    q_src2.allocator()->allocate();
    q_dst0.allocator()->allocate();
    q_res.allocator()->allocate();
    q_res_output.allocator()->allocate();

    // Run quantization layers (quantizes values of each tensor)
    q1.run();
    q2.run();
    q3.run();
    // Run low precision matrix multiply kernel
    qgemm.run();
    // Run output stage kernel
    gemmlowp_output_stage.run();
    std::cout << "\nTest Passed\n";

#if ARM_COMPUTE_DEBUG_ENABLED
    // Print quantized source matrices
    q_src1.print(std::cout);
    q_src2.print(std::cout);
    // Print result matrix in int32 form - before output stage processing
    std::cout << "Lowp GEMM output (int32):\n";
    q_res.print(std::cout);
    // Print QASYMM8 (quantized) matrix
    std::cout << "Output pipeline result matrix:\n";
    q_res_output.print(std::cout);

    // Expected result
    std::cout << "Expected result:\n";
    q_dst0.print(std::cout);
#endif // ARM_COMPUTE_DEBUG_ENABLED
}
