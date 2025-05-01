/*
 * Copyright (c) 2025 Arm Limited.
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
#include "arm_compute/runtime/NEON/NEFunctions.h"

#include "utils/Utils.h"

using namespace arm_compute;
using namespace utils;

class NEMatMulExample : public Example
{
public:
    bool do_setup(int argc, char **argv) override
    {
        size_t m = 4096;
        size_t n = 4096;
        size_t k = 128;

        if (argc == 4)
        {
            try
            {
                m = std::stoul(argv[1]);
                n = std::stoul(argv[2]);
                k = std::stoul(argv[3]);
            }
            catch (const std::exception &e)
            {
                ARM_COMPUTE_ERROR(e.what());
                return false;
            }
        }
        else if (argc != 1)
        {
            ARM_COMPUTE_ERROR("Invalid number of arguments. Usage:\n"
                              "<M> <N> <K>\n");
            return false;
        }

        const TensorInfo a_info{TensorShape{k, m}, 1, DataType::F32, DataLayout::NHWC};
        const TensorInfo b_info{TensorShape{n, k}, 1, DataType::F32, DataLayout::NHWC};
        const TensorInfo output_info{TensorShape{n, m}, 1, DataType::F32, DataLayout::NHWC};

        a.allocator()->init(a_info);
        b.allocator()->init(b_info);
        output.allocator()->init(output_info);

        a.info()->set_are_values_constant(false);
        b.info()->set_are_values_constant(false);
        output.info()->set_are_values_constant(false);

        const MatMulInfo        info;
        const CpuMatMulSettings settings;

        auto status = NEMatMul::validate(a.info(), b.info(), output.info(), info, settings);
        if (status.error_code() != ErrorCode::OK)
        {
            ARM_COMPUTE_ERROR(status.error_description().c_str());
            return false;
        }

        matmul.configure(&a, &b, &output, info, settings);
        a.allocator()->allocate();
        b.allocator()->allocate();
        output.allocator()->allocate();

        // Fill with fixed values
        const std::vector<float> values_a(m * k, 2.2f);
        const std::vector<float> values_b(n * k, 3.5f);
        fill_tensor_vector(a, values_a);
        fill_tensor_vector(b, values_b);

        return true;
    }

    void do_run() override
    {
        matmul.run();
    }

private:
    NEMatMul matmul{};
    Tensor   a{}, b{}, output{};
};

/** Main program for MatMul test
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments (M, N, K)
 */
int main(int argc, char **argv)
{
    return utils::run_example<NEMatMulExample>(argc, argv);
}
