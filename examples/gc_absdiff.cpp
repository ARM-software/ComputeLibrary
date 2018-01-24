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

#ifndef ARM_COMPUTE_GC /* Needed by Utils.cpp to handle OpenGL ES exceptions properly */
#error "This example needs to be built with -DARM_COMPUTE_GC"
#endif /* ARM_COMPUTE_GC */

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/GLES_COMPUTE/GCFunctions.h"
#include "arm_compute/runtime/GLES_COMPUTE/GCScheduler.h"
#include "utils/Utils.h"

using namespace arm_compute;
using namespace utils;

class GCAbsDiffExample : public Example
{
public:
    void do_setup(int argc, char **argv) override
    {
        PPMLoader ppm1, ppm2;

        GCScheduler::get().default_init();
        if(argc < 2)
        {
            // Print help
            std::cout << "Usage: " << argv[0] << " [input0_image.ppm] [input1_image.ppm] \n\n";
            std::cout << "No input_image provided, creating two dummy 640x480 images\n";
            // Create two empty grayscale 640x480 images
            src1.allocator()->init(TensorInfo(640, 480, Format::U8));
            src2.allocator()->init(TensorInfo(640, 480, Format::U8));
        }
        else if(argc < 3)
        {
            // Print help
            std::cout << "Usage: " << argv[0] << " [input0_image.ppm] [input1_image.ppm] \n\n";
            std::cout << "Only one input_image provided, creating a dummy 640x480 image\n";
            ppm1.open(argv[1]);
            ppm1.init_image(src1, Format::U8);
            // Create an empty grayscale 640x480 image
            src2.allocator()->init(TensorInfo(640, 480, Format::U8));
        }
        else
        {
            ppm1.open(argv[1]);
            ppm1.init_image(src1, Format::U8);
            ppm2.open(argv[2]);
            ppm2.init_image(src2, Format::U8);
        }

        // Configure the temporary and destination images
        dst.allocator()->init(*src1.info());

        absdiff.configure(&src1, &src2, &dst);

        // Allocate all the images
        src1.allocator()->allocate();
        src2.allocator()->allocate();
        dst.allocator()->allocate();

        // Fill the input image with the content of the PPM image if a filename was provided:
        if(ppm1.is_open())
        {
            ppm1.fill_image(src1);
            output_filename = std::string(argv[1]) + "_out.ppm";
        }
        if(ppm2.is_open())
        {
            ppm2.fill_image(src2);
        }
    }
    void do_run() override
    {
        // Execute the functions:
        absdiff.run();
    }
    void do_teardown() override
    {
        // Save the result to file:
        if(!output_filename.empty())
        {
            // save_to_ppm maps and unmaps the image to store as PPM
            // The GCTensor::map call inside the save_to_ppm will block until all pending operations on that image have completed
            save_to_ppm(dst, output_filename);
        }
    }

private:
    GCImage              src1{}, src2{}, dst{};
    GCAbsoluteDifference absdiff{};
    std::string          output_filename{};
};

/** Main program for absdiff test
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments ( [optional] Path to the first PPM image to process, [optional] Path the the second PPM image to process )
 */
int main(int argc, char **argv)
{
    return utils::run_example<GCAbsDiffExample>(argc, argv);
}
