/*
 * Copyright (c) 2019-2021, 2023 Arm Limited.
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

#include "CommonGemmExampleOptions.h"
#include "GemmTunerHelpers.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/KernelDescriptors.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "arm_compute/runtime/CL/CLTuner.h"
#include "src/gpu/cl/kernels/ClGemmMatrixMultiplyReshapedOnlyRhsKernel.h"
#include "tests/CL/Helper.h"
#include "utils/Utils.h"
#include "utils/command_line/CommandLineOptions.h"
#include "utils/command_line/CommandLineParser.h"

#include <cstdlib>

using namespace arm_compute;
using namespace arm_compute::opencl::kernels;
using namespace utils;
using namespace arm_compute::misc::shape_calculator;
using namespace gemm_tuner;

namespace
{
/** Structure holding all tunable gemm configs specific to this example/strategy */
struct GemmConfigs
{
    size_t m0{ 4 };                        /**< Number of rows processed by the matrix multiplication */
    size_t n0{ 4 };                        /**< Number of columns processed by the matrix multiplication */
    size_t k0{ 4 };                        /**< Number of partial accumulations performed by the matrix multiplication */
    size_t h0{ 1 };                        /**< Number of horizontal blocks of size (k0xn0) stored on the same output row */
    bool   interleave_rhs{ true };         /**< Interleave rhs matrix */
    bool   transpose_rhs{ true };          /**< Transpose rhs matrix */
    bool   export_to_cl_image_rhs{ true }; /**< Export rhs matrix to cl_image.*/
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
    os << "export_to_cl_image_rhs : " << (configs.export_to_cl_image_rhs ? true_str : false_str) << std::endl;
    return os;
}

/** Command line options for gemm configs */
class GemmConfigOptions
{
public:
    /** Constructor
     *
     * @param[in,out] parser A parser on which "parse()" hasn't been called yet.
     */
    GemmConfigOptions(CommandLineParser &parser)
        : m0(parser.add_positional_option<SimpleOption<size_t>>("m0", 4)),
          n0(parser.add_positional_option<SimpleOption<size_t>>("n0", 4)),
          k0(parser.add_positional_option<SimpleOption<size_t>>("k0", 4)),
          h0(parser.add_positional_option<SimpleOption<size_t>>("h0", 1)),
          interleave_rhs(parser.add_positional_option<SimpleOption<size_t>>("interleave_rhs", 1)),
          transpose_rhs(parser.add_positional_option<SimpleOption<size_t>>("transpose_rhs", 1)),
          export_to_cl_image_rhs(parser.add_positional_option<SimpleOption<size_t>>("export_to_cl_image_rhs", 1))
    {
        m0->set_help("Number of rows processed by the matrix multiplication");
        n0->set_help("Number of columns processed by the matrix multiplication");
        k0->set_help("Number of partial accumulations performed by the matrix multiplication");
        h0->set_help("Number of horizontal blocks of size (k0xn0) stored on the same output row");
        interleave_rhs->set_help("Interleave rhs matrix (1) / Do not interleave rhs matrix (0)");
        transpose_rhs->set_help("Transpose rhs matrix (1) / Do not transpose rhs matrix (0)");
        export_to_cl_image_rhs->set_help("Export rhs matrix to cl_image (1) / Do not export rhs matrix to cl_image (0)");
    }
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    GemmConfigOptions(const GemmConfigOptions &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    GemmConfigOptions &operator=(const GemmConfigOptions &) = delete;
    /** Allow instances of this class to be moved */
    GemmConfigOptions(GemmConfigOptions &&) = default;
    /** Allow instances of this class to be moved */
    GemmConfigOptions &operator=(GemmConfigOptions &&) = default;
    /** Default destructor */
    ~GemmConfigOptions() = default;

    SimpleOption<size_t> *m0;                     /**< Number of rows processed by the matrix multiplication option */
    SimpleOption<size_t> *n0;                     /**< Number of columns processed by the matrix multiplication option */
    SimpleOption<size_t> *k0;                     /**< Number of partial accumulations performed by the matrix multiplication option */
    SimpleOption<size_t> *h0;                     /**< Number of horizontal blocks of size (k0xn0) stored on the same output row option */
    SimpleOption<size_t> *interleave_rhs;         /**< Interleave rhs matrix option (1 enable; 0 disable) */
    SimpleOption<size_t> *transpose_rhs;          /**< Transpose rhs matrix option (1 enable; 0 disable) */
    SimpleOption<size_t> *export_to_cl_image_rhs; /**< Export rhs matrix to cl_image.*/
};

/** Consumes the gemm configuration options and creates a structure containing all information
 *
 * @param[in] options Options to consume
 *
 * @return Structure containing the gemm configurations
 */
GemmConfigs consume_gemm_configs(const GemmConfigOptions &options)
{
    GemmConfigs configs;
    configs.m0                     = options.m0->value();
    configs.n0                     = options.n0->value();
    configs.k0                     = options.k0->value();
    configs.h0                     = options.h0->value();
    configs.interleave_rhs         = options.interleave_rhs->value() != 0;
    configs.transpose_rhs          = options.transpose_rhs->value() != 0;
    configs.export_to_cl_image_rhs = options.export_to_cl_image_rhs->value() != 0;
    return configs;
}

} // namespace
// Create function for ClGemmMatrixMultiplyReshapedOnlyRhsKernel
using CLGEMMMatrixMultiplyReshapedOnlyRHS = test::CLSynthetizeOperator<ClGemmMatrixMultiplyReshapedOnlyRhsKernel>;

class CLGEMMMatrixMultiplyReshapedOnlyRHSExample : public Example
{
public:
    bool do_setup(int argc, char **argv) override
    {
        // Default parameters
        const float               alpha    = 1.0f;
        const float               beta     = 0.0f;
        const ActivationLayerInfo act_info = ActivationLayerInfo();
        CommonGemmExampleParams   params;
        GemmConfigs               configs;

        // Set up command line parser and options
        CommandLineParser        parser;
        CommonGemmExampleOptions param_options(parser);
        GemmConfigOptions        config_options(parser);

        // Parse command line options
        parser.parse(argc, argv);
        if(param_options.help->is_set() && param_options.help->value())
        {
            // Print help message
            parser.print_help(argv[0]);
            return false;
        }
        if(!parser.validate())
        {
            // Invalid arguments. Use default parameters and configs
            std::cerr << "Invalid arguments." << std::endl;
            parser.print_help(argv[0]);
            std::cerr << "Falling back to default parameters and configs" << std::endl;
        }
        else
        {
            // Get parameters and configs from command-line options
            params  = consume_common_gemm_example_parameters(param_options);
            configs = consume_gemm_configs(config_options);
        }

        // Print gemm parameters and configurations
        std::cout << "Gemm parameters:" << std::endl;
        std::cout << params << std::endl;
        std::cout << "Gemm configurations:" << std::endl;
        std::cout << configs << std::endl;

        tuner.set_tuner_mode(params.tuner_mode);

        CLScheduler::get().default_init(&tuner);

        lhs.allocator()->init(TensorInfo(TensorShape(params.K, params.M, params.B), 1, params.data_type));
        rhs.allocator()->init(TensorInfo(TensorShape(params.N, params.K, params.B), 1, params.data_type));
        bias.allocator()->init(TensorInfo(TensorShape(params.N, 1, params.B), 1, params.data_type));

        GEMMLHSMatrixInfo lhs_info;
        lhs_info.m0 = configs.m0;
        lhs_info.k0 = configs.k0;

        GEMMRHSMatrixInfo rhs_info;
        rhs_info.n0                 = configs.n0;
        rhs_info.k0                 = configs.k0;
        rhs_info.h0                 = configs.h0;
        rhs_info.interleave         = configs.interleave_rhs;
        rhs_info.transpose          = configs.transpose_rhs;
        rhs_info.export_to_cl_image = configs.export_to_cl_image_rhs;

        GEMMKernelInfo kernel_info;
        kernel_info.m                       = params.M;
        kernel_info.n                       = params.N;
        kernel_info.k                       = params.K;
        kernel_info.depth_output_gemm3d     = 0;
        kernel_info.reinterpret_input_as_3d = false;
        kernel_info.broadcast_bias          = true;
        kernel_info.activation_info         = act_info;

        if(rhs_info.h0 == 0)
        {
            rhs_info.h0 = std::max(kernel_info.n / rhs_info.n0, 1U);
        }

        // Initialise rhs_reshaped tensor info
        rhs_reshaped.allocator()->init(TensorInfo(compute_rhs_reshaped_shape(*rhs.info(), rhs_info), 1, params.data_type));

        if(rhs_info.export_to_cl_image)
        {
            if(!examples::gemm_tuner_helpers::update_padding_for_cl_image(rhs_reshaped.info()))
            {
                std::cerr << "cl_image is not supported on the device, disable export_to_cl_image" << std::endl;
                return false;
            }
        }

        // Validate argments
        Status status{};
        status = gemm.validate(lhs.info(), rhs_reshaped.info(), bias.info(), dst.info(), alpha, beta, lhs_info, rhs_info, kernel_info);
        if(!status)
        {
            // Unsupported arguments
            std::cerr << "Unsupported arguments." << std::endl;
            std::cerr << "Check documentation for supported/unsupported combinations" << std::endl;
            return false;
        }

        // Configure function
        gemm.configure(lhs.info(), rhs_reshaped.info(), bias.info(), dst.info(), alpha, beta, lhs_info, rhs_info, kernel_info);

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
        ITensorPack gemm_pack({ { ACL_SRC_0, &lhs },
            { ACL_SRC_1, &rhs_reshaped },
            { ACL_SRC_2, &bias },
            { ACL_DST, &dst }
        });
        gemm.run(gemm_pack);

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
 * @param[in] argv Arguments ( [optional] M, [optional] N, [optional] K, [optional] B, [optional] m0, [optional] n0, [optional] k0, [optional] h0, [optional] interleave_rhs, [optional] transpose_rhs, [optional] export_to_cl_image)
 */
int main(int argc, char **argv)
{
    return run_example<CLGEMMMatrixMultiplyReshapedOnlyRHSExample>(argc, argv);
}
