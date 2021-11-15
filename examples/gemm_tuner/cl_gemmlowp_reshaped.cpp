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
#ifndef ARM_COMPUTE_CL /* Needed by Utils.cpp to handle OpenCL exceptions properly */
#error "This example needs to be built with -DARM_COMPUTE_CL"
#endif /* ARM_COMPUTE_CL */

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/KernelDescriptors.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "arm_compute/runtime/CL/CLTuner.h"
#include "examples/gemm_tuner/CommonGemmExampleOptions.h"
#include "examples/gemm_tuner/GemmTunerHelpers.h"
#include "src/gpu/cl/kernels/ClGemmLowpMatrixMultiplyReshapedKernel.h"
#include "src/gpu/cl/kernels/ClGemmReshapeLhsMatrixKernel.h"
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
    size_t m0{ 4 };                /**< Number of rows processed by the matrix multiplication */
    size_t n0{ 4 };                /**< Number of columns processed by the matrix multiplication */
    size_t k0{ 4 };                /**< Number of partial accumulations performed by the matrix multiplication */
    size_t v0{ 1 };                /**< Number of vertical blocks of size (m0xk0) stored on the same output row */
    size_t h0{ 1 };                /**< Number of horizontal blocks of size (k0xn0) stored on the same output row */
    bool   interleave_lhs{ true }; /**< Interleave lhs matrix */
    bool   transpose_lhs{ true };  /**< Transpose lhs matrix. */
    bool   interleave_rhs{ true }; /**< Interleave rhs matrix */
    bool   transpose_rhs{ true };  /**< Transpose rhs matrix. */
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
    os << "v0 : " << configs.v0 << std::endl;
    os << "h0 : " << configs.h0 << std::endl;
    os << "interleave_lhs : " << (configs.interleave_lhs ? true_str : false_str) << std::endl;
    os << "transpose_lhs : " << (configs.transpose_lhs ? true_str : false_str) << std::endl;
    os << "interleave_rhs : " << (configs.interleave_rhs ? true_str : false_str) << std::endl;
    os << "transpose_rhs : " << (configs.transpose_rhs ? true_str : false_str) << std::endl;
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
          v0(parser.add_positional_option<SimpleOption<size_t>>("v0", 1)),
          h0(parser.add_positional_option<SimpleOption<size_t>>("h0", 1)),
          interleave_lhs(parser.add_positional_option<SimpleOption<size_t>>("interleave_lhs", 1)),
          interleave_rhs(parser.add_positional_option<SimpleOption<size_t>>("interleave_rhs", 1)),
          transpose_rhs(parser.add_positional_option<SimpleOption<size_t>>("transpose_rhs", 1))
    {
        m0->set_help("Number of rows processed by the matrix multiplication");
        n0->set_help("Number of columns processed by the matrix multiplication");
        k0->set_help("Number of partial accumulations performed by the matrix multiplication");
        v0->set_help("Number of vertical blocks of size (m0xk0) stored on the same output row");
        h0->set_help("Number of horizontal blocks of size (k0xn0) stored on the same output row");
        interleave_lhs->set_help("Interleave lhs matrix (1) / Do not interleave lhs matrix (0)");
        interleave_rhs->set_help("Interleave rhs matrix (1) / Do not interleave rhs matrix (0)");
        // FIXME: Currently we only support 2 variants of the gemm reshaped kernels in which transpose_lhs and
        // transpose_rhs are the opposites of each other. In the future we may extend the kernels to include the other
        // 2 variants (both transposed and none transposed)
        transpose_rhs->set_help("Transpose rhs matrix but not lhs matrix (1) / Do not transpose rhs matrix but do transpose lhs matrix (0)");
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

    SimpleOption<size_t> *m0;             /**< Number of rows processed by the matrix multiplication option */
    SimpleOption<size_t> *n0;             /**< Number of columns processed by the matrix multiplication option */
    SimpleOption<size_t> *k0;             /**< Number of partial accumulations performed by the matrix multiplication option */
    SimpleOption<size_t> *v0;             /**< Number of vertical blocks of size (m0xk0) stored on the same output row option */
    SimpleOption<size_t> *h0;             /**< Number of horizontal blocks of size (k0xn0) stored on the same output row option */
    SimpleOption<size_t> *interleave_lhs; /**< Interleave lhs matrix option (1 enable; 0 disable) */
    SimpleOption<size_t> *interleave_rhs; /**< Interleave rhs matrix option (1 enable; 0 disable) */
    // FIXME: Currently we only support 2 variants of the gemm reshaped kernels in which transpose_lhs and
    // transpose_rhs are the opposites of each other. In the future we may extend the kernels to include the other
    // 2 variants (both transposed and none transposed)
    SimpleOption<size_t> *transpose_rhs; /**< Transpose rhs matrix option (1 enable; 0 disable). Also set the lhs matrix transpose option to the opposite. */
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
    configs.m0             = options.m0->value();
    configs.n0             = options.n0->value();
    configs.k0             = options.k0->value();
    configs.v0             = options.v0->value();
    configs.h0             = options.h0->value();
    configs.interleave_lhs = options.interleave_lhs->value() != 0;
    // FIXME: Currently we only support 2 variants of the gemm reshaped kernels in which transpose_lhs and
    // transpose_rhs are the opposites of each other. In the future we may extend the kernels to include the other
    // 2 variants (both transposed and none transposed)
    configs.transpose_lhs  = options.transpose_rhs->value() == 0;
    configs.interleave_rhs = options.interleave_rhs->value() != 0;
    configs.transpose_rhs  = options.transpose_rhs->value() != 0;
    return configs;
}

} // namespace

using ClGemmReshapeLHSMatrix           = test::CLSynthetizeOperator<ClGemmReshapeLhsMatrixKernel>;
using ClGemmLowpMatrixMultiplyReshaped = test::CLSynthetizeOperator<ClGemmLowpMatrixMultiplyReshapedKernel>;

class CLGEMMLowpMatrixMultiplyReshapedExample : public Example
{
public:
    bool do_setup(int argc, char **argv) override
    {
        // Default parameters
        CommonGemmExampleParams params;
        GemmConfigs             configs;

        // Parse command line options
        CommandLineParser        parser;
        CommonGemmExampleOptions param_options(parser, DataType::QASYMM8);
        GemmConfigOptions        config_options(parser);

        parser.parse(argc, argv);
        if(param_options.help->is_set() && param_options.help->value())
        {
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
            params  = consume_common_gemm_example_parameters(param_options);
            configs = consume_gemm_configs(config_options);
        }

        std::cout << "Gemm parameters:" << std::endl;
        std::cout << params << std::endl;
        std::cout << "Gemm configurations:" << std::endl;
        std::cout << configs << std::endl;

        tuner.set_tuner_mode(params.tuner_mode);

        CLScheduler::get().default_init(&tuner);

        lhs.allocator()->init(TensorInfo(TensorShape(params.K, params.M, params.B), 1, params.data_type));
        rhs.allocator()->init(TensorInfo(TensorShape(params.N, params.K, params.B), 1, params.data_type));

        // Set arbitrary quantization information
        const QuantizationInfo q_info
        {
            0.012, 3
        };
        lhs.info()->set_quantization_info(q_info);
        rhs.info()->set_quantization_info(q_info);
        dst.info()->set_quantization_info(q_info);

        GEMMLHSMatrixInfo lhs_info;
        lhs_info.m0         = configs.m0;
        lhs_info.k0         = configs.k0;
        lhs_info.v0         = configs.v0;
        lhs_info.interleave = configs.interleave_lhs;
        lhs_info.transpose  = configs.transpose_lhs;

        GEMMRHSMatrixInfo rhs_info;
        rhs_info.n0                 = configs.n0;
        rhs_info.k0                 = configs.k0;
        rhs_info.h0                 = configs.h0;
        rhs_info.interleave         = configs.interleave_rhs;
        rhs_info.transpose          = configs.transpose_rhs;
        rhs_info.export_to_cl_image = false; // CL image not supported for quantized cases yet

        lhs_reshaped.allocator()->init(TensorInfo(compute_lhs_reshaped_shape(*lhs.info(), lhs_info), 1, params.data_type));
        rhs_reshaped.allocator()->init(TensorInfo(compute_rhs_reshaped_shape(*rhs.info(), rhs_info), 1, params.data_type));
        lhs_reshaped.info()->set_quantization_info(q_info);
        rhs_reshaped.info()->set_quantization_info(q_info);

        if(rhs_info.export_to_cl_image)
        {
            if(!examples::gemm_tuner_helpers::update_padding_for_cl_image(rhs_reshaped.info()))
            {
                std::cerr << "cl_image is not supported on the device, disable export_to_cl_image" << std::endl;
                return false;
            }
        }

        GEMMReshapeInfo gemm_info
        {
            static_cast<int>(params.M),
            static_cast<int>(params.N),
            static_cast<int>(params.K),
            static_cast<int>(configs.h0),
            static_cast<int>(configs.v0),
            0,
            false,
            true
        };

        // Validate argments
        if(!reshape_lhs.validate(lhs.info(), lhs_reshaped.info(), lhs_info, gemm_info.reinterpret_input_as_3d()))
        {
            std::cerr << "Invalid arguments for ClGemmReshapeLHSMatrixKernel." << std::endl;
            return false;
        }

        if(!gemm.validate(lhs_reshaped.info(), rhs_reshaped.info(), dst.info(), lhs_info, rhs_info, gemm_info))
        {
            std::cerr << "Invalid arguments for ClGemmLowpMatrixMultiplyReshapedKernel." << std::endl;
            return false;
        }

        // Configure functions
        reshape_lhs.configure(lhs.info(), lhs_reshaped.info(), lhs_info);

        gemm.configure(lhs_reshaped.info(), rhs_reshaped.info(), dst.info(), lhs_info, rhs_info, gemm_info);

        // Allocate tensors
        lhs.allocator()->allocate();
        rhs.allocator()->allocate();
        lhs_reshaped.allocator()->allocate();
        rhs_reshaped.allocator()->allocate();
        dst.allocator()->allocate();

        return true;
    }
    void do_run() override
    {
        ITensorPack reshape_lsh_pack({ { ACL_SRC, &lhs }, { ACL_DST, &lhs_reshaped } });
        reshape_lhs.run(reshape_lsh_pack);

        ITensorPack gemm_pack({ { ACL_SRC_0, &lhs_reshaped }, { ACL_SRC_1, &rhs_reshaped }, { ACL_DST, &dst } });
        gemm.run(gemm_pack);

        // Make sure all the OpenCL jobs are done executing:
        CLScheduler::get().sync();
    }

    void do_teardown() override
    {
    }

private:
    CLTensor                         lhs{};
    CLTensor                         rhs{};
    CLTensor                         lhs_reshaped{};
    CLTensor                         rhs_reshaped{};
    CLTensor                         dst{};
    CLTuner                          tuner{};
    ClGemmReshapeLHSMatrix           reshape_lhs{};
    ClGemmLowpMatrixMultiplyReshaped gemm{};
};

/** Main test program for gemmlowp reshaped
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments ( [optional] M, [optional] N, [optional] K, [optional] B, [optional] m0, [optional] n0, [optional] k0, [optional] v0, [optional] h0, [optional] interleave_lhs, [optional] interleave_rhs, [optional] transpose_rhs )
 */
int main(int argc, char **argv)
{
    return run_example<CLGEMMLowpMatrixMultiplyReshapedExample>(argc, argv);
}
