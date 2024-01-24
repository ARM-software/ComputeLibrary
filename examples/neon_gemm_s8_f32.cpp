/*
 * Copyright (c) 2020-2021, 2024 Arm Limited.
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
#include "arm_compute/core/utils/quantization/AsymmHelpers.h"
#include "arm_compute/core/WindowIterator.h"
#include "arm_compute/runtime/NEON/NEFunctions.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"

#include "support/ToolchainSupport.h"
#include "utils/Utils.h"

#include <cstdlib>

using namespace arm_compute;
using namespace utils;

QuantizationInfo dynamic_qinfo(QuantizationInfo qinfo)
{
    return QuantizationInfo(qinfo.scale(), qinfo.offset(), true);
}
void set_qinfo_dynamic(Tensor &t)
{
    t.info()->set_quantization_info(dynamic_qinfo(t.info()->quantization_info()));
}

void quantize(Tensor &qt, const Tensor &t, float min, float max)
{
    DataType dt = DataType::QASYMM8_SIGNED;

    // Determine the scale
    const float scale = (max - min) / 256.0f;

    // Determine the zero-point; using affine equation val = (qval-zerop) * scale
    const float zero_point = -128.0f - min / scale;

    QuantizationInfo qinfo(scale, (int32_t)round(zero_point), true);

    // We now have the quantisation info and can configure the quantised tensor
    qt.allocator()->init(TensorInfo(t.info()->tensor_shape(), 1, dt, qinfo));
    qt.allocator()->allocate();
    NEQuantizationLayer quantization;
    quantization.configure(&t, &qt);
    quantization.run();
}

void invert_qinfo_offset(Tensor &t)
{
    QuantizationInfo qinfo = t.info()->quantization_info();
    t.info()->set_quantization_info(QuantizationInfo(qinfo.scale()[0], -qinfo.offset()[0], qinfo.is_dynamic()));
}

void print_quantization_info(const Tensor &t, const std::string &name_prefix)
{
    QuantizationInfo qinfo = t.info()->quantization_info();
    std::cout << name_prefix << "_qinfo="
              << "QuantizationInfo(" << qinfo.scale()[0] << ", " << qinfo.offset()[0] << ")\n";
}

int main(int argc, char **argv)
{
    size_t M = 4;
    size_t N = 4;
    size_t K = 4;

    // Parse args
    if (argc < 3) /* case default matrix sizes */
    {
        // Print help
        std::cout << "Usage: ./build/neon_gemm_qasymm8 M N K\n";
        std::cout << "Too few or no inputs provided. Using default M=4, N=4, K=4\n\n";
    }
    else /* case M N K arguments provided */
    {
        M = strtol(argv[1], nullptr, 10);
        N = strtol(argv[2], nullptr, 10);
        K = strtol(argv[3], nullptr, 10);
    }

    /*** Floating point matrix multiplication ***/

    // Initialise input matrices
    NEGEMM fgemm{};

    Tensor src1;
    Tensor src2;
    Tensor dst;
    src1.allocator()->init(TensorInfo(TensorShape(K, M), 1, DataType::F32));
    src2.allocator()->init(TensorInfo(TensorShape(N, K), 1, DataType::F32));
    dst.allocator()->init(TensorInfo(TensorShape(N, M), 1, DataType::F32));
    fgemm.configure(&src1, &src2, nullptr, &dst, 1, 0);

    // Allocate matrices
    src1.allocator()->allocate();
    src2.allocator()->allocate();
    dst.allocator()->allocate();

    float min1 = 0.0f;
    float max1 = 1.0f;
    fill_random_tensor(src1, 0, min1, max1);

    float min2 = -1.0f;
    float max2 = 2.0f;
    fill_random_tensor(src2, 1, min2, max2);

    // Run single precision gemm and print result
    fgemm.run();

#if ARM_COMPUTE_DEBUG_ENABLED
    std::cout << "# F32 GEMM result:\n";
    std::cout << "src1=[ \n";
    src1.print(std::cout);
    std::cout << "] \n";
    std::cout << "src2=[ \n";
    src2.print(std::cout);
    std::cout << "] \n";
    std::cout << "dst=[ \n";
    dst.print(std::cout);
    std::cout << "] \n";
#endif // ARM_COMPUTE_DEBUG_ENABLED

    Tensor q_src1;
    quantize(q_src1, src1, min1, max1);
    print_quantization_info(q_src1, "src1");
    q_src1.info()->set_are_values_constant(false);

    // NEGEMMLowpMatrixMultiplyCore adopts the opposite convention for the offset
    // compared to NEQuantizeLayer
    invert_qinfo_offset(q_src1);

    Tensor q_src2;
    quantize(q_src2, src2, min2, max2);
    print_quantization_info(q_src2, "src2");
    q_src2.info()->set_are_values_constant(false);

    // NEGEMMLowpMatrixMultiplyCore adopts the opposite convention for the offset
    // compared to NEQuantizeLayer
    invert_qinfo_offset(q_src2);

    // q_dst will be Dequantized to F32 so it doesn't need a QuantizationInfo
    Tensor q_dst;
    q_dst.allocator()->init(TensorInfo(TensorShape(N, M), 1, DataType::F32));

    // Configure low precision gemm and initialise result tensor (pre-output)
    NEGEMMLowpMatrixMultiplyCore qgemm;
    qgemm.configure(&q_src1, &q_src2, nullptr, &q_dst);

    q_dst.allocator()->allocate();

    // Run low precision matrix multiply kernel
    qgemm.run();

#if ARM_COMPUTE_DEBUG_ENABLED
    // Print quantized source matrices
    std::cout << "q_src1=[ \n";
    q_src1.print(std::cout);
    std::cout << "] \n";
    std::cout << "q_src2=[ \n";
    q_src2.print(std::cout);
    std::cout << "] \n";
    std::cout << "# Lowp GEMM output (FP32):\n";
    std::cout << "q_dst=[ \n";
    q_dst.print(std::cout);
    std::cout << "] \n";

    // Expected result
    std::cout << "# Expected result:\n";
    std::cout << "dst=[ \n";
    dst.print(std::cout);
    std::cout << "] \n";
#endif // ARM_COMPUTE_DEBUG_ENABLED

    // Rerun to test the ability to modify the Tensor contents and QuantizationInfo (dynamic quantization)
    min1 = -1.0f;
    max1 = 1.0f;
    fill_random_tensor(src1, 2, min1, max1);

#if ARM_COMPUTE_DEBUG_ENABLED
    std::cout << "# Refilled src1\n";
    std::cout << "src1=[ \n";
    src1.print(std::cout);
    std::cout << "] \n";
    std::cout << "src2=[ \n";
    src2.print(std::cout);
    std::cout << "] \n";
#endif // ARM_COMPUTE_DEBUG_ENABLED

    fgemm.run();

    quantize(q_src1, src1, min1, max1);
    set_qinfo_dynamic(q_src1);
    print_quantization_info(q_src1, "src1");

    // NEGEMMLowpMatrixMultiplyCore adopts the opposite convention for the offset
    // compared to NEQuantizeLayer
    invert_qinfo_offset(q_src1);

    qgemm.run();

#if ARM_COMPUTE_DEBUG_ENABLED
    // Print quantized source matrices
    std::cout << "q_src1=[ \n";
    q_src1.print(std::cout);
    std::cout << "] \n";
    std::cout << "q_src2=[ \n";
    q_src2.print(std::cout);
    std::cout << "] \n";
    std::cout << "# Lowp GEMM output (FP32):\n";
    std::cout << "q_dst=[ \n";
    q_dst.print(std::cout);
    std::cout << "] \n";

    // Expected result
    std::cout << "# Expected result:\n";
    std::cout << "dst=[ \n";
    dst.print(std::cout);
    std::cout << "] \n";
#endif // ARM_COMPUTE_DEBUG_ENABLED
}
