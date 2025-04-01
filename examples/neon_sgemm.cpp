/*
 * Copyright (c) 2018-2019, 2025 Arm Limited.
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
#include "arm_compute/function_info/GEMMInfo.h"
#include "arm_compute/runtime/NEON/NEFunctions.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"

#include "utils/command_line/CommandLineParser.h"
#include "utils/command_line/SimpleOption.h"
#include "utils/command_line/ToggleOption.h"
#include "utils/Utils.h"

#include <cstdlib>

using namespace arm_compute;
using namespace utils;

static bool file_exists(const std::string &filename)
{
    std::ifstream file(filename);
    return file.good();
}

static bool equal_float(const float a, const float b)
{
    constexpr float tolerance = 1e-6;
    return std::fabs(a - b) <= tolerance;
}

class NESGEMMExample : public Example
{
public:
    bool do_setup(int argc, char **argv) override
    {
        NPYLoader npy0;
        NPYLoader npy1;
        NPYLoader npy2;
        alpha           = 1.0f;
        beta            = 0.0f;
        is_dynamic      = false;
        is_constant_b_c = false;
        is_bias_present = false;

        utils::CommandLineParser parser;

        auto help_opt = parser.add_option<utils::ToggleOption>("help");
        help_opt->set_help("Print help message and exit");

        auto src0_opt = parser.add_option<utils::SimpleOption<std::string>>("src0");
        src0_opt->set_help("File name with NPY data for src0");

        auto src1_opt = parser.add_option<utils::SimpleOption<std::string>>("src1");
        src1_opt->set_help("File name with NPY data for src1");

        auto src2_opt = parser.add_option<utils::SimpleOption<std::string>>("src2");
        src2_opt->set_help("File name with NPY data for src2");

        auto m_opt = parser.add_option<utils::SimpleOption<int>>("m");
        m_opt->set_help("M shape. This cannot be set together with src0/src1/src2");
        auto n_opt = parser.add_option<utils::SimpleOption<int>>("n");
        n_opt->set_help("N shape. This cannot be set together with src0/src1/src2");
        auto k_opt = parser.add_option<utils::SimpleOption<int>>("k");
        k_opt->set_help("K shape. This cannot be set together with src0/src1/src2");

        auto alpha_opt = parser.add_option<utils::SimpleOption<float>>("alpha", 1.0f);
        alpha_opt->set_help("Alpha value. Default = 1.0");

        auto beta_opt = parser.add_option<utils::SimpleOption<float>>("beta", 0.0f);
        beta_opt->set_help("Beta value. Default = 0.0");

        auto constant_b_c_opt = parser.add_option<utils::ToggleOption>("constant_b_c", false);
        constant_b_c_opt->set_help("Whether B and C should be treated as constant data. Default = false");

        auto mode_opt = parser.add_option<utils::SimpleOption<std::string>>("mode", "static");
        mode_opt->set_help("GEMM mode. Allowed values: static, dynamic. Default value: static");

        parser.parse(argc, argv);

        if (help_opt->is_set() && help_opt->value())
        {
            parser.print_help(argv[0]);
            return false;
        }
        const bool shapes_set = m_opt->is_set() && n_opt->is_set() && k_opt->is_set();
        const bool files_set  = src0_opt->is_set() && src1_opt->is_set() && src2_opt->is_set();

        if (shapes_set && files_set)
        {
            std::cout << "M,N,K cannot be set together with src0/src1/src2." << std::endl;
            parser.print_help(argv[0]);
            return false;
        }

        alpha           = alpha_opt->value();
        beta            = beta_opt->value();
        is_constant_b_c = constant_b_c_opt->is_set() && constant_b_c_opt->value();
        is_bias_present = !equal_float(beta, 0.0f);

        if (mode_opt->value() == "dynamic")
        {
            is_dynamic = true;
        }
        else if (mode_opt->value() != "static")
        {
            std::cout << "Invalid mode: " << mode_opt->value() << ". Allowed values: static, dynamic." << std::endl;
            parser.print_help(argv[0]);
            return false;
        }

        if (is_dynamic && (!equal_float(alpha, 1.0f) || !equal_float(beta, 1.0f)))
        {
            std::cout << "Dynamic shape tensors are only supported when 'alpha' and 'beta' equal to 1.0" << std::endl;
            parser.print_help(argv[0]);
            return false;
        }

        if (files_set)
        {
            if (!file_exists(src0_opt->value()) || !file_exists(src1_opt->value()) ||
                (is_bias_present && !file_exists(src2_opt->value())))
            {
                std::cout << "Some of provided files cannot be open: " << src0_opt->value() << ", " << src1_opt->value()
                          << ((is_bias_present) ? (", " + src2_opt->value()) : "") << std::endl;
                return false;
            }
        }

        if (files_set)
        {
            npy0.open(src0_opt->value());
            npy0.init_tensor(src0, DataType::F32);
            npy1.open(src1_opt->value());
            npy1.init_tensor(src1, DataType::F32);

            if (is_bias_present)
            {
                npy2.open(src2_opt->value());
                npy2.init_tensor(src2, DataType::F32);
            }
        }
        else
        {
            size_t M = 7;
            size_t N = 3;
            size_t K = 5;

            if (shapes_set)
            {
                M = m_opt->value();
                N = n_opt->value();
                K = k_opt->value();
            }
            else
            {
                std::cout << "Shapes are invalid or not provided. Using M=" << M << ", N=" << N << ", K=" << K << "."
                          << std::endl;
            }

            src0.allocator()->init(TensorInfo(TensorShape(K, M), 1, DataType::F32));
            src1.allocator()->init(TensorInfo(TensorShape(N, K), 1, DataType::F32));
            if (is_bias_present)
            {
                const auto bias_shape = (is_dynamic) ? TensorShape(N) : TensorShape(N, M);
                src2.allocator()->init(TensorInfo(bias_shape, 1, DataType::F32));
            }
        }

        init_sgemm_output(dst, src0, src1, DataType::F32);
        auto src0_shape = src0.info()->tensor_shape();
        auto src1_shape = src1.info()->tensor_shape();
        auto src2_shape = src2.info()->tensor_shape();
        auto dst_shape  = dst.info()->tensor_shape();

        if (is_dynamic)
        {
            src0.info()->set_tensor_shape(TensorShape()).set_dynamic(true);
            if (!is_constant_b_c)
            {
                src1.info()->set_tensor_shape(TensorShape()).set_dynamic(true);
                src2.info()->set_tensor_shape(TensorShape()).set_dynamic(true);
            }
            dst.info()->set_tensor_shape(TensorShape()).set_dynamic(true);
        }
        src1.info()->set_are_values_constant(is_constant_b_c);
        src2.info()->set_are_values_constant(is_constant_b_c);

        // Configure function
        GEMMInfo gemm_info = {false, false, is_constant_b_c};
        sgemm.configure(&src0, &src1, (is_bias_present) ? &src2 : nullptr, &dst, alpha, beta, gemm_info);

        if (is_dynamic)
        {
            src0.info()->set_tensor_shape(src0_shape);
            if (!is_constant_b_c)
            {
                src1.info()->set_tensor_shape(src1_shape);
                src2.info()->set_tensor_shape(src2_shape);
            }
            dst.info()->set_tensor_shape(dst_shape);
        }

        // Allocate all the images
        src0.allocator()->allocate();
        src1.allocator()->allocate();
        if (is_bias_present)
        {
            src2.allocator()->allocate();
        }
        dst.allocator()->allocate();

        // Fill the input images with either the data provided or random data
        if (npy0.is_open())
        {
            npy0.fill_tensor(src0);
            npy1.fill_tensor(src1);

            output_filename = "sgemm_out.npy";
            is_fortran      = npy0.is_fortran();

            if (npy2.is_open())
            {
                npy2.fill_tensor(src2);
            }
        }
        else
        {
            fill_random_tensor(src0, -1.f, 1.f);
            fill_random_tensor(src1, -1.f, 1.f);
            if (is_bias_present)
            {
                fill_random_tensor(src2, -1.f, 1.f);
            }
        }

        // Dummy run for CLTuner
        sgemm.run();

        return true;
    }
    void do_run() override
    {
        // Execute the function
        sgemm.run();
    }
    void do_teardown() override
    {
        if (!output_filename.empty()) /* Save to .npy file */
        {
            save_to_npy(dst, output_filename, is_fortran);
        }
    }

private:
    Tensor      src0{}, src1{}, src2{}, dst{};
    NEGEMM      sgemm{};
    float       alpha{}, beta{};
    bool        is_fortran{};
    std::string output_filename{};
    bool        is_dynamic{};
    bool        is_constant_b_c{};
    bool        is_bias_present{};
};

/** Main program for sgemm test
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments ( [optional] Matrix A, [optional] Matrix B, [optional] Matrix C, [optional] alpha, [optional] beta )
 */
int main(int argc, char **argv)
{
    return utils::run_example<NESGEMMExample>(argc, argv);
}
