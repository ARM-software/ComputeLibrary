/*
 * Copyright (c) 2017, 2018 ARM Limited.
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

class NEONCartoonEffectExample : public Example
{
public:
    void do_setup(int argc, char **argv) override
    {
        // Open PPM file
        PPMLoader ppm;

        if(argc < 2)
        {
            // Print help
            std::cout << "Usage: ./build/neon_cartoon_effect [input_image.ppm]\n\n";
            std::cout << "No input_image provided, creating a dummy 640x480 image\n";
            // Create an empty grayscale 640x480 image
            src_img.allocator()->init(TensorInfo(640, 480, Format::U8));
        }
        else
        {
            ppm.open(argv[1]);
            ppm.init_image(src_img, Format::U8);
        }

        // Initialize just the dimensions and format of the images:
        gaus5x5_img.allocator()->init(*src_img.info());
        canny_edge_img.allocator()->init(*src_img.info());
        dst_img.allocator()->init(*src_img.info());

        // Configure the functions to call
        gaus5x5.configure(&src_img, &gaus5x5_img, BorderMode::REPLICATE);
        canny_edge.configure(&src_img, &canny_edge_img, 100, 80, 3, 1, BorderMode::REPLICATE);
        sub.configure(&gaus5x5_img, &canny_edge_img, &dst_img, ConvertPolicy::SATURATE);

        // Now that the padding requirements are known we can allocate the images:
        src_img.allocator()->allocate();
        dst_img.allocator()->allocate();
        gaus5x5_img.allocator()->allocate();
        canny_edge_img.allocator()->allocate();

        // Fill the input image with the content of the PPM image if a filename was provided:
        if(ppm.is_open())
        {
            ppm.fill_image(src_img);
            output_filename = std::string(argv[1]) + "_out.ppm";
        }
    }

    void do_run() override
    {
        // Execute the functions:
        gaus5x5.run();
        canny_edge.run();
        sub.run();
    }

    void do_teardown() override
    {
        // Save the result to file:
        if(!output_filename.empty())
        {
            save_to_ppm(dst_img, output_filename); // save_to_ppm maps and unmaps the image to store as PPM
        }
    }

private:
    Image                   src_img{}, dst_img{}, gaus5x5_img{}, canny_edge_img{};
    NEGaussian5x5           gaus5x5{};
    NECannyEdge             canny_edge{};
    NEArithmeticSubtraction sub{};
    std::string             output_filename{};
};

/** Main program for cartoon effect test
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments ( [optional] Path to PPM image to process )
 */
int main(int argc, char **argv)
{
    return utils::run_example<NEONCartoonEffectExample>(argc, argv);
}
