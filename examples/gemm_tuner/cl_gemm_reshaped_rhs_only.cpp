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
#ifndef ARM_COMPUTE_CL /* Needed by Utils.cpp to handle OpenCL exceptions properly */
#error "This example needs to be built with -DARM_COMPUTE_CL"
#endif /* ARM_COMPUTE_CL */

#include "arm_compute/core/CL/kernels/CLGEMMMatrixMultiplyReshapedOnlyRHSKernel.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/KernelDescriptors.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/runtime/CL/CLFunctions.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "arm_compute/runtime/CL/CLTuner.h"
#include "tests/CL/Helper.h"
#include "utils/Utils.h"

#include <cstdlib>

using namespace arm_compute;
using namespace utils;
using namespace arm_compute::misc::shape_calculator;

namespace
{
/** Structure holding all the common gemm example parameters */
struct CommonGemmExampleParams
{
    size_t M{ 100 };
    size_t N{ 100 };
    size_t K{ 50 };
    size_t B{ 1 };
};

/** Formatted output of the CommonGemmExampleParams type
 *
 * @param[out] os            Output stream.
 * @param[in]  common_params Common parameters to output
 *
 * @return Modified output stream.
 */
::std::ostream &operator<<(::std::ostream &os, const CommonGemmExampleParams &common_params)
{
    os << "M : " << common_params.M << std::endl;
    os << "N : " << common_params.N << std::endl;
    os << "K : " << common_params.K << std::endl;
    os << "B : " << common_params.B << std::endl;
    return os;
}

/** Structure holding all tunable gemm configs specific to this example/strategy */
struct GemmConfigs
{
    size_t m0{ 4 };
    size_t n0{ 4 };
    size_t k0{ 4 };
    size_t h0{ 1 };
    bool   interleave_rhs{ true };
    bool   transpose_rhs{ true };
};

/** Formatted output of the GemmConfigs type
 *
 * @param[out] os      Output stream.
 * @param[in]  configs Tunable configurations to output
 *
 * @return Modified output stream.
 */
::std::ostream &operator<<(::std::ostream &os, const GemmConfigs &configs)
{
    std::string false_str = std::string("false");
    std::string true_str  = std::string("true");

    os << "m0 : " << configs.m0 << std::endl;
    os << "n0 : " << configs.n0 << std::endl;
    os << "k0 : " << configs.k0 << std::endl;
    os << "h0 : " << configs.h0 << std::endl;
    os << "interleave_rhs : " << (configs.interleave_rhs ? true_str : false_str) << std::endl;
    os << "transpose_rhs : " << (configs.transpose_rhs ? true_str : false_str) << std::endl;
    return os;
}
} // namespace
// Create function for CLGEMMMatrixMultiplyReshapedOnlyRHSKernel
using CLGEMMMatrixMultiplyReshapedOnlyRHS = test::CLSynthetizeFunction<CLGEMMMatrixMultiplyReshapedOnlyRHSKernel>;

class CLGEMMMatrixMultiplyReshapedOnlyRHSExample : public Example
{
public:
    bool do_setup(int argc, char **argv) override
    {
        // Default parameters
        const DataType            data_type = DataType::F32;
        const float               alpha     = 1.0f;
        const float               beta      = 0.0f;
        const ActivationLayerInfo act_info  = ActivationLayerInfo();
        CommonGemmExampleParams   params;
        GemmConfigs               configs;
        if(argc < 9 || argc > 11)
        {
            // Print help
            // Use default parameters
            std::cerr << "Usage: ./build/cl_gemm_reshaped_rhs_only M N K B m0 n0 k0 h0 [interleave_rhs = 1] [transpose_rhs = 1]\n\n";
            std::cerr << "Falling back to default parameters and configs" << std::endl;
        }
        else
        {
            // Set parameters from command line arguments
            params.M   = strtol(argv[1], nullptr, 10);
            params.N   = strtol(argv[2], nullptr, 10);
            params.K   = strtol(argv[3], nullptr, 10);
            params.B   = strtol(argv[4], nullptr, 10);
            configs.m0 = strtol(argv[5], nullptr, 10);
            configs.n0 = strtol(argv[6], nullptr, 10);
            configs.k0 = strtol(argv[7], nullptr, 10);
            configs.h0 = strtol(argv[8], nullptr, 10);
            if(argc > 9)
            {
                configs.interleave_rhs = strtol(argv[9], nullptr, 10) == 1;
            }
            if(argc > 10)
            {
                configs.transpose_rhs = strtol(argv[10], nullptr, 10) == 1;
            }
        }
        std::cerr << "Gemm parameters:" << std::endl;
        std::cerr << params << std::endl;
        std::cerr << "Gemm configurations:" << std::endl;
        std::cerr << configs << std::endl;

        CLScheduler::get().default_init(&tuner);

        lhs.allocator()->init(TensorInfo(TensorShape(params.K, params.M, params.B), 1, data_type));
        rhs.allocator()->init(TensorInfo(TensorShape(params.N, params.K, params.B), 1, data_type));
        bias.allocator()->init(TensorInfo(TensorShape(params.N, params.M, params.B), 1, data_type));

        init_sgemm_output(dst, lhs, rhs, data_type);

        GEMMLHSMatrixInfo lhs_info;
        lhs_info.m0 = configs.m0;
        lhs_info.k0 = configs.k0;

        GEMMRHSMatrixInfo rhs_info;
        rhs_info.n0         = configs.n0;
        rhs_info.k0         = configs.k0;
        rhs_info.h0         = configs.h0;
        rhs_info.interleave = configs.interleave_rhs;
        rhs_info.transpose  = configs.transpose_rhs;

        GEMMKernelInfo kernel_info;
        kernel_info.m                       = params.M;
        kernel_info.n                       = params.N;
        kernel_info.k                       = params.K;
        kernel_info.depth_output_gemm3d     = 0;
        kernel_info.reinterpret_input_as_3d = false;
        kernel_info.broadcast_bias          = true;
        kernel_info.activation_info         = act_info;

        // Initialise rhs_reshaped tensor info
        auto_init_if_empty(*rhs_reshaped.info(), rhs.info()->clone()->set_tensor_shape(compute_rhs_reshaped_shape(*rhs.info(), rhs_info)));

        // Configure function
        gemm.configure(&lhs, &rhs_reshaped, &bias, &dst, alpha, beta, lhs_info, rhs_info, kernel_info);

        // Allocate tensors
        lhs.allocator()->allocate();
        rhs.allocator()->allocate();
        rhs_reshaped.allocator()->allocate();
        bias.allocator()->allocate();
        dst.allocator()->allocate();

        return true;
    }
    void do_run() override
    {
        // Execute the function
        gemm.run();

        // Make sure all the OpenCL jobs are done executing:
        CLScheduler::get().sync();
    }

    void do_teardown() override
    {
    }

private:
    CLTensor                            lhs{};
    CLTensor                            rhs{};
    CLTensor                            rhs_reshaped{};
    CLTensor                            bias{};
    CLTensor                            dst{};
    CLTuner                             tuner{};
    CLGEMMMatrixMultiplyReshapedOnlyRHS gemm{};
};

/** Main program for gemm reshaped rhs only test
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments ( M, N, K, B, m0, n0, k0, h0, [optional] interleave_rhs, [optional] transpose_rhs )
 */
int main(int argc, char **argv)
{
    return utils::run_example<CLGEMMMatrixMultiplyReshapedOnlyRHSExample>(argc, argv);
}
