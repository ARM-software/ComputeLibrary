/*
 * Copyright (c) 2016, 2018 ARM Limited.
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
#include "arm_compute/runtime/NEON/NEFunctions.h"

#include "arm_compute/core/Types.h"
#include "utils/Utils.h"

using namespace arm_compute;
using namespace utils;

void main_neon_scale(int argc, char **argv)
{
    PPMLoader ppm;
    Image     src, dst;

    if(argc < 2)
    {
        // Print help
        std::cout << "Usage: ./build/neon_scale[input_image.ppm]\n\n";
        std::cout << "No input_image provided, creating a dummy 640x480 image\n";
        // Create an empty grayscale 640x480 image
        src.allocator()->init(TensorInfo(640, 480, Format::U8));
    }
    else
    {
        ppm.open(argv[1]);
        ppm.init_image(src, Format::U8);
    }

    constexpr int scale_factor = 2;

    TensorInfo dst_tensor_info(src.info()->dimension(0) / scale_factor, src.info()->dimension(1) / scale_factor, Format::U8);

    // Configure the destination image
    dst.allocator()->init(dst_tensor_info);

    // Create and initialize a Scale function object:
    NEScale scale;
    scale.configure(&src, &dst, InterpolationPolicy::NEAREST_NEIGHBOR, BorderMode::UNDEFINED);

    // Allocate all the images
    src.allocator()->allocate();
    dst.allocator()->allocate();
    // Fill the input image with the content of the PPM image if a filename was provided:
    if(ppm.is_open())
    {
        ppm.fill_image(src);
    }

    // Run the scale operation:
    scale.run();

    // Save the result to file:
    if(ppm.is_open())
    {
        const std::string output_filename = std::string(argv[1]) + "_out.ppm";
        save_to_ppm(dst, output_filename);
    }
}

/** Main program for convolution test
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments ( [optional] Path to PPM image to process )
 */
int main(int argc, char **argv)
{
    return utils::run_example(argc, argv, main_neon_scale);
}
