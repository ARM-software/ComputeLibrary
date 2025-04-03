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

class NEConvolutionExample : public Example
{
public:
    bool do_setup(int argc, char **argv) override
    {
        TensorShape   input_shape{32, 256, 256};
        TensorShape   weights_shape{32, 4, 4, 4};
        TensorShape   output_shape{4, 127, 127};
        TensorInfo    input_info{input_shape, 1, DataType::F16, DataLayout::NHWC};
        TensorInfo    weights_info{weights_shape, 1, DataType::F16, DataLayout::NHWC};
        TensorInfo    output_info{output_shape, 1, DataType::F16, DataLayout::NHWC};
        PadStrideInfo ps_info{1, 1, 0, 0, DimensionRoundingType::FLOOR};

        if (argc == 11)
        {
            try
            {
                const size_t   input_x         = std::stoul(argv[1]);
                const size_t   input_y         = std::stoul(argv[2]);
                const size_t   input_z         = std::stoul(argv[3]);
                const size_t   kernel_size_x   = std::stoul(argv[4]);
                const size_t   kernel_size_y   = std::stoul(argv[5]);
                const size_t   output_channels = std::stoul(argv[6]);
                const uint32_t stride_x        = static_cast<uint32_t>(std::stoul(argv[7]));
                const uint32_t stride_y        = static_cast<uint32_t>(std::stoul(argv[8]));
                const uint32_t pad_x           = static_cast<uint32_t>(std::stoul(argv[9]));
                const uint32_t pad_y           = static_cast<uint32_t>(std::stoul(argv[10]));

                input_shape   = TensorShape{input_z, input_x, input_y};
                input_info    = TensorInfo{input_shape, 1, DataType::F16, DataLayout::NHWC};
                weights_shape = TensorShape{input_z, kernel_size_x, kernel_size_y, output_channels};
                weights_info  = TensorInfo{weights_shape, 1, DataType::F16, DataLayout::NHWC};
                ps_info       = PadStrideInfo{stride_x, stride_y, pad_x, pad_y, DimensionRoundingType::FLOOR};
                output_shape  = arm_compute::misc::shape_calculator::compute_deep_convolution_shape(
                     input_info, weights_info, ps_info);
                output_info = TensorInfo{output_shape, 1, DataType::F16, DataLayout::NHWC};
            }
            catch (const std::exception &e)
            {
                ARM_COMPUTE_ERROR(e.what());
                return false;
            }
        }
        else if (argc != 1)
        {
            ARM_COMPUTE_ERROR(
                "Invalid number of arguments. Usage:\n"
                "<input_width> <input_height> <input_channels> <kernel_size_x> <kernel_size_y> <output_channels> "
                "<stride_x> <stride_y> <pad_x> <pad_y>\n");
            return false;
        }

        input.allocator()->init(input_info);
        weights.allocator()->init(weights_info);
        output.allocator()->init(output_info);

        auto status = NEConvolutionLayer::validate(input.info(), weights.info(), nullptr, output.info(), ps_info);
        if (status.error_code() != ErrorCode::OK)
        {
            ARM_COMPUTE_ERROR(status.error_description().c_str());
            return false;
        }

        conv.configure(&input, &weights, nullptr, &output, ps_info);
        input.allocator()->allocate();
        weights.allocator()->allocate();
        output.allocator()->allocate();

        return true;
    }

    void do_run() override
    {
        conv.run();
    }

private:
    NEConvolutionLayer conv{};
    Tensor             input{}, weights{}, output{};
};

/** Main program for convolution test
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments (input_width, input_height, input_channels, kernel_size_x, kernel_size_y, output_channels, stride_x, stride_y pad_x, pad_y)
 */
int main(int argc, char **argv)
{
    return utils::run_example<NEConvolutionExample>(argc, argv);
}
