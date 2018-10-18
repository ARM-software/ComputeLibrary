/*
 * Copyright (c) 2017-2018 ARM Limited.
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
#ifndef ARM_COMPUTE_CL /* Needed by Utils.cpp to handle OpenCL exceptions properly */
#error "This example needs to be built with -DARM_COMPUTE_CL"
#endif /* ARM_COMPUTE_CL */

#include "arm_compute/core/Types.h"
#include "arm_compute/core/utils/quantization/AsymmHelpers.h"
#include "arm_compute/runtime/CL/CLFunctions.h"
#include "arm_compute/runtime/CL/CLScheduler.h"

#include "tests/AssetsLibrary.h"
#include "tests/CL/CLAccessor.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/SimpleTensor.h"
#include "tests/validation/Validation.h"
#include "tests/validation/reference/GEMM.h"
#include "tests/validation/reference/GEMMLowp.h"

#include "ValidateExample.h"

#include "utils/Utils.h"

#include <cstdlib>

using namespace arm_compute;
using namespace utils;
using namespace arm_compute::test;
using namespace arm_compute::test::validation;

constexpr float                     abs_tolerance_f32(0.0001f); /**< F32 Absolute tolerance value for comparing reference's output against implementation's output for
                                                               * floating point data types in case using relative tolerance fails because of small values */
RelativeTolerance<float>            tolerance_f32(0.001f);      /**< F32 Tolerance value for comparing reference's output against implementation's output for floating point data types */
RelativeTolerance<half_float::half> tolerance_f16(half(0.2));   /**< F16 Tolerance value for comparing reference's output against implementation's output for floating point data types */
constexpr float                     tolerance_num_f16 = 0.02f;  /**< F16 Tolerance number */

class CLGEMMValidateExample : public ValidateExample
{
public:
    bool do_setup(int argc, char **argv) override
    {
        //TODO(antbar01): Update to use command line interface ?
        CLScheduler::get().default_init();
        if(argc == 2)
        {
            size_t dt = strtol(argv[1], nullptr, 10);
            switch(dt)
            {
                case 1:
                {
                    data_type = DataType::F16;
                    std::cout << "Usage: " << argv[0] << "1 M N K [alpha = 1.0f] [beta = 0.0f]\n";
                    std::cout << "Using default values: Datatype=FP16 M=7, N=3, K=5, alpha=1.0f and beta=0.0f\n";
                    break;
                }
                case 2:
                {
                    data_type = DataType::QASYMM8;
                    std::cout << "Usage: " << argv[0] << "2 M N K [scale_src0 = 0.1f] [offset_scr0 = f] [scale_scr1 = 0.1f] [offset_scr1 = 10] [scale_dst = 0.1f] [offset_dst = 10] [bias = 1]\n";
                    std::cout <<
                              "Using default values: Datatype=QASYMM8 M=7, N=3, K=5, scale_src0 =(1.0f/255), offset_src0 = 10, scale_src1 =(1.0f/255), offset_src1 = 10, scale_dst =(1.0f/255), offset_dst = 10, bias=1\n\n";
                    break;
                }
                case 0:
                default:
                {
                    data_type = DataType::F32;
                    std::cout << "Usage: " << argv[0] << "0 M N K [alpha = 1.0f] [beta = 0.0f]\n";
                    std::cout << "Using default values: Datatype=FP32 M=7, N=3, K=5, alpha=1.0f and beta=0.0f\n";
                }
            }
        }
        else if(argc < 5)
        {
            // Print help
            std::cout << "Usage with datatype = FP32    : " << argv[0] << "0 M N K [alpha = 1.0f] [beta = 0.0f]\n";
            std::cout << "           datatype = FP16    : " << argv[0] << "1 M N K [alpha = 1.0f] [beta = 0.0f]\n";
            std::cout << "           datatype = QASYMM8 : " << argv[0] << "2 M N K [scale_src0 = 0.1f] [offset_scr0 = f] [scale_scr1 = 0.1f] [offset_scr1 = 10] [scale_dst = 0.1f] [offset_dst = 10] [bias = 1]\n";
            std::cout << "Too few or no arguments provided.\n";
            std::cout << "Using default values: Datatype=FP32 M=7, N=3, K=5, alpha=1.0f and beta=0.0f\n";
        }
        else
        {
            size_t dt = strtol(argv[1], nullptr, 10);
            switch(dt)
            {
                case 1:
                {
                    data_type = DataType::F16;
                    break;
                }
                case 2:
                {
                    data_type = DataType::QASYMM8;
                    break;
                }
                case 0:
                default:
                    data_type = DataType::F32;
            }
            M = strtol(argv[2], nullptr, 10);
            N = strtol(argv[3], nullptr, 10);
            K = strtol(argv[4], nullptr, 10);
        }

        switch(data_type)
        {
            case DataType::F16:
            case DataType::F32:
            {
                if(argc > 5)
                {
                    alpha = strtof(argv[5], nullptr);
                    if(argc > 6)
                    {
                        beta = strtof(argv[6], nullptr);
                    }
                }
                break;
            }
            case DataType::QASYMM8:
            {
                if(argc > 5)
                {
                    scale_src0 = strtof(argv[5], nullptr);
                    if(argc > 6)
                    {
                        offset_src0 = strtol(argv[6], nullptr, 10);
                        if(argc > 7)
                        {
                            scale_src1 = strtof(argv[7], nullptr);
                            if(argc > 8)
                            {
                                offset_src1 = strtol(argv[8], nullptr, 10);
                                if(argc > 9)
                                {
                                    scale_dst = strtof(argv[9], nullptr);
                                    if(argc > 10)
                                    {
                                        offset_dst = strtol(argv[10], nullptr, 10);
                                        if(argc > 11)
                                        {
                                            add_bias = (strtol(argv[11], nullptr, 10) == 1);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                float multiplier = scale_src0 * scale_src1 / scale_dst;
                quantization::calculate_quantized_multiplier_less_than_one(multiplier, &dst_multiplier, &dst_shift);
                break;
            }
            default:
                break;
        }

        src0.allocator()->init(TensorInfo(TensorShape(K, M), 1, data_type));
        src1.allocator()->init(TensorInfo(TensorShape(N, K), 1, data_type));
        src2.allocator()->init(TensorInfo(TensorShape(N, M), 1, data_type));
        init_sgemm_output(dst, src0, src1, data_type);

        // Configure function
        if(data_type == DataType::QASYMM8)
        {
            src0.info()->set_quantization_info(QuantizationInfo(scale_src0, offset_src0));
            src1.info()->set_quantization_info(QuantizationInfo(scale_src1, offset_src1));
            dst.info()->set_quantization_info(QuantizationInfo(scale_dst, offset_dst));
            biases.allocator()->init(TensorInfo(TensorShape(N), 1, DataType::S32));
            init_sgemm_output(tmp_dst, src0, src1, DataType::S32);

            // Configure GEMMlowp matrix multiply function
            mm_gemmlowp.configure(&src0, &src1, nullptr, &tmp_dst);

            // Configure GEMMlowp output stage
            mm_gemmlowp_output_stage.configure(&tmp_dst, add_bias ? &biases : nullptr, &dst, dst_multiplier, dst_shift, offset_dst);
            tmp_dst.allocator()->allocate();
            biases.allocator()->allocate();
            fill(CLAccessor(biases), 3);
        }
        else
        {
            // Configure matrix multiply function
            mm_gemm.configure(&src0, &src1, &src2, &dst, alpha, beta);
        }

        // Allocate all the tensors
        src0.allocator()->allocate();
        src1.allocator()->allocate();
        dst.allocator()->allocate();
        src2.allocator()->allocate();

        fill(CLAccessor(src0), 0);
        fill(CLAccessor(src1), 1);
        fill(CLAccessor(src2), 2);

        return true;
    }

    void print_parameters(framework::Printer &printer) override
    {
        printer.print_entry("Datatype", string_from_data_type(data_type));
        printer.print_entry("M", support::cpp11::to_string(M));
        printer.print_entry("N", support::cpp11::to_string(N));
        printer.print_entry("K", support::cpp11::to_string(K));
        if(data_type == DataType::QASYMM8)
        {
            printer.print_entry("Scale_Src0", support::cpp11::to_string(scale_src0));
            printer.print_entry("Offset_Src0", support::cpp11::to_string(offset_src0));
            printer.print_entry("Scale_Scr1", support::cpp11::to_string(scale_src1));
            printer.print_entry("Offset_Src1", support::cpp11::to_string(offset_src1));
            printer.print_entry("Scale_Dst", support::cpp11::to_string(scale_dst));
            printer.print_entry("Offset_Dst", support::cpp11::to_string(offset_dst));
            printer.print_entry("Bias", support::cpp11::to_string(add_bias));
        }
        else
        {
            printer.print_entry("Alpha", support::cpp11::to_string(alpha));
            printer.print_entry("Beta", support::cpp11::to_string(beta));
        }
    }

    void do_validate() override
    {
        switch(data_type)
        {
            case DataType::F16:
            {
                SimpleTensor<half> ref_src0 = { TensorShape(K, M), data_type, 1 };
                SimpleTensor<half> ref_src1 = { TensorShape(N, K), data_type, 1 };
                SimpleTensor<half> ref_src2 = { TensorShape(N, M), data_type, 1 };

                fill(ref_src0, 0);
                fill(ref_src1, 1);
                fill(ref_src2, 2);

                SimpleTensor<half> ref_dst = reference::gemm<half>(ref_src0, ref_src1, ref_src2, alpha, beta);
                validate(CLAccessor(dst), ref_dst, tolerance_f16, tolerance_num_f16);
                break;
            }
            case DataType::F32:
            {
                SimpleTensor<float> ref_src0 = { TensorShape(K, M), data_type, 1 };
                SimpleTensor<float> ref_src1 = { TensorShape(N, K), data_type, 1 };
                SimpleTensor<float> ref_src2 = { TensorShape(N, M), data_type, 1 };

                fill(ref_src0, 0);
                fill(ref_src1, 1);
                fill(ref_src2, 2);

                SimpleTensor<float> ref_dst = reference::gemm<float>(ref_src0, ref_src1, ref_src2, alpha, beta);
                validate(CLAccessor(dst), ref_dst, tolerance_f32, 0.f, abs_tolerance_f32);
                break;
            }
            case DataType::QASYMM8:
            {
                SimpleTensor<uint8_t> ref_src0{ TensorShape(K, M), data_type, 1 };
                SimpleTensor<uint8_t> ref_src1{ TensorShape(N, K), data_type, 1 };
                SimpleTensor<uint8_t> ref_dst;

                // Fill reference
                fill(ref_src0, 0);
                fill(ref_src1, 1);

                SimpleTensor<int32_t> ref_tmp_dst = reference::gemmlowp_matrix_multiply_core<int32_t, uint8_t>(ref_src0, ref_src1, TensorShape(N, M), offset_src0, offset_src1);

                if(add_bias)
                {
                    SimpleTensor<int32_t> biases{ TensorShape(N), DataType::S32, 1 };
                    // Fill bias
                    fill(biases, 3);
                    ref_dst = reference::gemmlowp_quantize_down_int32_to_uint8_scale_by_fixedpoint<int32_t>(ref_tmp_dst, biases, dst_multiplier, dst_shift, offset_dst);
                }
                else
                {
                    ref_dst = reference::gemmlowp_quantize_down_int32_to_uint8_scale_by_fixedpoint<int32_t>(ref_tmp_dst, dst_multiplier, dst_shift, offset_dst);
                }
                validate(CLAccessor(dst), ref_dst);
                break;
            }
            default:
                break;
        }
    }
    void do_run() override
    {
        // Execute the function
        if(data_type == DataType::QASYMM8)
        {
            // Run gemmlowp
            mm_gemmlowp.run();
            // Run output stage
            mm_gemmlowp_output_stage.run();
        }
        else
        {
            // Run gemm
            mm_gemm.run();
        }

        // Make sure all the OpenCL jobs are done executing:
        CLScheduler::get().sync();
    }

private:
    template <typename U>
    void fill(U &&tensor, int i)
    {
        switch(tensor.data_type())
        {
            case DataType::F16:
            case DataType::F32:
            {
                std::uniform_real_distribution<> distribution(-1.0f, 1.0f);
                library->fill(tensor, distribution, i);
                break;
            }
            case DataType::S32:
            case DataType::QASYMM8:
            {
                std::uniform_int_distribution<> distribution(-6000, 6000);
                library->fill(tensor, distribution, i);
                break;
            }
            default:
                library->fill_tensor_uniform(tensor, i);
        }
    }

    CLTensor src0{}, src1{}, src2{}, dst{};
    CLTensor tmp_dst{}, biases{};

    CLGEMM                                              mm_gemm{};
    CLGEMMLowpMatrixMultiplyCore                        mm_gemmlowp{};
    CLGEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPoint mm_gemmlowp_output_stage{};

    size_t   M{ 7 }, N{ 3 }, K{ 5 };
    DataType data_type{ DataType::F32 };
    float    alpha{ 1.0 }, beta{ 0.0 };
    int      offset_src0{ 10 }, offset_src1{ 10 }, offset_dst{ 10 };
    float    scale_src0{ 1.0f / 255 }, scale_src1{ 1.0f / 255 }, scale_dst{ 1.0f / 255 };
    int32_t  dst_multiplier{ 0 }, dst_shift{ 0 };
    bool     add_bias{ true };
};

/** Main program for gemm test
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments ( [optional] datatype, [optional] M, [optional] N, [optional] K, [optional] scale_src0, [optional] offset_src0, [optional] scale_src1, [optional] offset_src1, [optional] scale_dst, [optional] offset_dst, [optional] bias, [optional] alpha, [optional] beta )
 *
 */
int main(int argc, char **argv)
{
    return utils::run_example<CLGEMMValidateExample>(argc, argv);
}
