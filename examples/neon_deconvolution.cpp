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
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/runtime/NEON/NEFunctions.h"

#include "utils/Utils.h"

using namespace arm_compute;
using namespace utils;

class NEDeconvolutionExample : public Example
{
public:
    bool do_setup(int argc, char **argv) override
    {
        if (argc == 12)
        {
            try
            {
                size_t   input_x         = std::stoul(argv[1]);
                size_t   input_y         = std::stoul(argv[2]);
                size_t   input_z         = std::stoul(argv[3]);
                size_t   kernel_size_x   = std::stoul(argv[4]);
                size_t   kernel_size_y   = std::stoul(argv[5]);
                size_t   output_channels = std::stoul(argv[6]);
                uint32_t stride_x        = static_cast<uint32_t>(std::stoul(argv[7]));
                uint32_t stride_y        = static_cast<uint32_t>(std::stoul(argv[8]));
                uint32_t pad_x           = static_cast<uint32_t>(std::stoul(argv[9]));
                uint32_t pad_y           = static_cast<uint32_t>(std::stoul(argv[10]));
                bool     fast_math       = std::stoul(argv[11]);

                TensorShape   input_shape{input_z, input_x, input_y};
                TensorInfo    input_info{input_shape, 1, DataType::F16, DataLayout::NHWC};
                TensorShape   weights_shape{input_z, kernel_size_x, kernel_size_y, output_channels};
                TensorInfo    weights_info{weights_shape, 1, DataType::F16, DataLayout::NHWC};
                PadStrideInfo ps_info{stride_x, stride_y, pad_x, pad_y, DimensionRoundingType::FLOOR};
                auto out_dim = deconvolution_output_dimensions(input_x, input_y, kernel_size_x, kernel_size_y, ps_info);
                TensorShape output_shape = arm_compute::misc::shape_calculator::compute_deconvolution_output_shape(
                    out_dim, input_info, weights_info);
                TensorInfo output_info{output_shape, 1, DataType::F16, DataLayout::NHWC};

                input.allocator()->init(input_info);
                weights.allocator()->init(weights_info);
                output.allocator()->init(output_info);

                auto status = NEDeconvolutionLayer::validate(input.info(), weights.info(), nullptr, output.info(),
                                                             ps_info, fast_math);
                if (status.error_code() != ErrorCode::OK)
                {
                    ARM_COMPUTE_ERROR(status.error_description().c_str());
                    return false;
                }

                deconv.configure(&input, &weights, nullptr, &output, ps_info, fast_math);
                input.allocator()->allocate();
                weights.allocator()->allocate();
                output.allocator()->allocate();

                return true;
            }
            catch (const std::exception &e)
            {
                ARM_COMPUTE_ERROR(e.what());
                return false;
            }
        }
        else
        {
            ARM_COMPUTE_ERROR(
                "Invalid number of arguments. Usage:\n"
                "<input_width> <input_height> <input_channels> <kernel_size_x> <kernel_size_y> <output_channels> "
                "<stride_x> <stride_y> <pad_x> <pad_y> <fast_math (0/1)>\n");
            return false;
        }
        return false;
    }

    void do_run() override
    {
        deconv.run();
    }

private:
    NEDeconvolutionLayer deconv{};
    Tensor               input{}, weights{}, output{};
};

/** Main program for deconvolution test
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments (input_width, input_height, input_channels, kernel_size_x, kernel_size_y, output_channels, stride_x, stride_y pad_x, pad_y, fast_math)
 */
int main(int argc, char **argv)
{
    return utils::run_example<NEDeconvolutionExample>(argc, argv);
}
