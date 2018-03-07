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
#include "arm_compute/runtime/CL/CLFunctions.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "arm_compute/runtime/CL/CLTuner.h"

#include "tests/AssetsLibrary.h"
#include "tests/CL/CLAccessor.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/SimpleTensor.h"
#include "tests/validation/Validation.h"
#include "tests/validation/reference/GEMM.h"

#include "ValidateExample.h"

#include "utils/Utils.h"

#include <cstdlib>

using namespace arm_compute;
using namespace utils;
using namespace arm_compute::test;
using namespace arm_compute::test::validation;

RelativeTolerance<float> tolerance_f32(0.001f); /**< Tolerance value for comparing reference's output against implementation's output for floating point data types */
constexpr float          tolerance_num = 0.02f; /**< Tolerance number */

class CLSGEMMValidateExample : public ValidateExample
{
public:
    void do_setup(int argc, char **argv) override
    {
        alpha = 1.0f;
        beta  = 0.0f;

        CLScheduler::get().default_init(&tuner);
        if(argc < 3)
        {
            // Print help
            std::cout << "Usage: " << argv[0] << " M N K [alpha = 1.0f] [beta = 0.0f]\n\n";
            std::cout << "Too few or no input_matrices provided. Using M=7, N=3, K=5, alpha=1.0f and beta=0.0f\n\n";
        }
        else
        {
            M = strtol(argv[1], nullptr, 10);
            N = strtol(argv[2], nullptr, 10);
            K = strtol(argv[3], nullptr, 10);
        }

        src0.allocator()->init(TensorInfo(TensorShape(K, M), 1, DataType::F32));
        src1.allocator()->init(TensorInfo(TensorShape(N, K), 1, DataType::F32));
        src2.allocator()->init(TensorInfo(TensorShape(N, M), 1, DataType::F32));

        if(argc > 4)
        {
            alpha = strtof(argv[4], nullptr);

            if(argc > 5)
            {
                beta = strtof(argv[5], nullptr);
            }
        }

        init_sgemm_output(dst, src0, src1, DataType::F32);

        // Configure function
        sgemm.configure(&src0, &src1, (src2.info()->total_size() > 0) ? &src2 : nullptr, &dst, alpha, beta);

        // Allocate all the images
        src0.allocator()->allocate();
        src1.allocator()->allocate();
        dst.allocator()->allocate();

        // Fill the input images with either the data provided or random data
        src2.allocator()->allocate();

        fill(CLAccessor(src0), 0);
        fill(CLAccessor(src1), 1);
        fill(CLAccessor(src2), 2);
    }
    void print_parameters(framework::Printer &printer) override
    {
        printer.print_entry("M", support::cpp11::to_string(M));
        printer.print_entry("N", support::cpp11::to_string(N));
        printer.print_entry("K", support::cpp11::to_string(K));
    }
    void do_validate() override
    {
        SimpleTensor<float> ref_src0 = { TensorShape(K, M), DataType::F32, 1 };
        SimpleTensor<float> ref_src1 = { TensorShape(N, K), DataType::F32, 1 };
        SimpleTensor<float> ref_src2 = { TensorShape(N, M), DataType::F32, 1 };

        fill(ref_src0, 0);
        fill(ref_src1, 1);
        fill(ref_src2, 2);

        SimpleTensor<float> ref_dst = reference::gemm<float>(ref_src0, ref_src1, ref_src2, alpha, beta);
        validate(CLAccessor(dst), ref_dst, tolerance_f32, tolerance_num);
    }
    void do_run() override
    {
        // Execute the function
        sgemm.run();

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
            default:
                library->fill_tensor_uniform(tensor, i);
        }
    }
    size_t   M{ 7 }, N{ 3 }, K{ 5 };
    CLTensor src0{}, src1{}, src2{}, dst{};
    CLGEMM   sgemm{};
    CLTuner  tuner{};
    float    alpha{}, beta{};
};

/** Main program for sgemm test
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments ( [optional] Matrix A, [optional] Matrix B, [optional] Matrix C, [optional] alpha, [optional] beta )
 */
int main(int argc, char **argv)
{
    return utils::run_example<CLSGEMMValidateExample>(argc, argv);
}
