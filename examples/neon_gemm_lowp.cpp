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

class NEGEMMLowpMatrixMultiplyCoreExample : public Example
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

        const QuantizationInfo qinfo{1.0f, 64};
        TensorInfo             a_info{TensorShape{k, m}, 1, DataType::QASYMM8, qinfo};
        TensorInfo             b_info{TensorShape{n, k}, 1, DataType::QASYMM8, qinfo};
        const TensorInfo       output_info{TensorShape{n, m}, 1, DataType::S32, DataLayout::NHWC};

        a_info.set_data_layout(DataLayout::NHWC);
        b_info.set_data_layout(DataLayout::NHWC);

        a.allocator()->init(a_info);
        b.allocator()->init(b_info);
        output.allocator()->init(output_info);

        a.info()->set_are_values_constant(false);
        b.info()->set_are_values_constant(false);
        output.info()->set_are_values_constant(false);

        auto status = NEGEMMLowpMatrixMultiplyCore::validate(a.info(), b.info(), nullptr, output.info());
        if (status.error_code() != ErrorCode::OK)
        {
            ARM_COMPUTE_ERROR(status.error_description().c_str());
            return false;
        }

        lpgemm.configure(&a, &b, nullptr, &output);
        a.allocator()->allocate();
        b.allocator()->allocate();
        output.allocator()->allocate();

        // Fill with fixed values
        fill_tensor_value(a, 65);
        fill_tensor_value(b, 63);

        return true;
    }

    void do_run() override
    {
        lpgemm.run();
    }

private:
    NEGEMMLowpMatrixMultiplyCore lpgemm{};
    Tensor                       a{}, b{}, output{};
};

/** Main program for GEMMLowpMatrixMultiplyCore test
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments (M, N, K)
 */
int main(int argc, char **argv)
{
    return utils::run_example<NEGEMMLowpMatrixMultiplyCoreExample>(argc, argv);
}
