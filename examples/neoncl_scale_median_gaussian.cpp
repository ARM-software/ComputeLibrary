/*
 * Copyright (c) 2016-2021 Arm Limited.
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

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "arm_compute/runtime/CL/functions/CLGaussian5x5.h"
#include "arm_compute/runtime/CL/functions/CLScale.h"
#include "arm_compute/runtime/NEON/NEFunctions.h"
#include "utils/ImageLoader.h"
#include "utils/Utils.h"

using namespace arm_compute;
using namespace utils;

/** Example demonstrating how to use both CL and Neon functions in the same pipeline
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments ( [optional] Path to PPM image to process )
 */
class NEONCLScaleMedianGaussianExample : public Example
{
public:
    bool do_setup(int argc, char **argv) override
    {
        /** [Neon / OpenCL Interop] */
        PPMLoader ppm;

        CLScheduler::get().default_init();

        if(argc < 2)
        {
            // Print help
            std::cout << "Usage: ./build/cl_convolution [input_image.ppm]\n\n";
            std::cout << "No input_image provided, creating a dummy 640x480 image\n";
            // Create an empty grayscale 640x480 image
            src.allocator()->init(TensorInfo(640, 480, Format::U8));
        }
        else
        {
            ppm.open(argv[1]);
            ppm.init_image(src, Format::U8);
        }

        TensorInfo scale_median_info(TensorInfo(src.info()->dimension(0) / 2, src.info()->dimension(1) / 2, Format::U8));

        // Configure the temporary and destination images
        scale_median.allocator()->init(scale_median_info);
        median_gauss.allocator()->init(scale_median_info);
        dst.allocator()->init(scale_median_info);

        scale.configure(&src, &scale_median, ScaleKernelInfo{ InterpolationPolicy::NEAREST_NEIGHBOR, BorderMode::REPLICATE });
        median.configure(&scale_median, &median_gauss, BorderMode::REPLICATE);
        gauss.configure(&median_gauss, &dst, BorderMode::REPLICATE);

        // Allocate all the images
        src.allocator()->allocate();
        scale_median.allocator()->allocate();
        median_gauss.allocator()->allocate();
        dst.allocator()->allocate();

        // Fill the input image with the content of the PPM image if a filename was provided:
        if(ppm.is_open())
        {
            ppm.fill_image(src);
            const std::string output_filename = std::string(argv[1]) + "_out.ppm";
        }
        /** [Neon / OpenCL Interop] */

        return true;
    }
    void do_run() override
    {
        // Enqueue and flush the OpenCL kernel:
        scale.run();

        // Do a blocking map of the input and output buffers of the Neon function:
        scale_median.map();
        median_gauss.map();

        // Run the Neon function:
        median.run();

        // Unmap the output buffer before it's used again by OpenCL:
        scale_median.unmap();
        median_gauss.unmap();

        // Run the final OpenCL function:
        gauss.run();

        // Make sure all the OpenCL jobs are done executing:
        CLScheduler::get().sync();
    }
    void do_teardown() override
    {
        // Save the result to file:
        if(!output_filename.empty())
        {
            save_to_ppm(dst, output_filename); // save_to_ppm maps and unmaps the image to store as PPM
        }
    }

private:
    CLImage       src{}, scale_median{}, median_gauss{}, dst{};
    CLScale       scale{};
    NEMedian3x3   median{};
    CLGaussian5x5 gauss{};
    std::string   output_filename{};
};

/** Main program for neon/cl scale median gaussian test
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments ( [optional] Path to PPM image to process )
 */
int main(int argc, char **argv)
{
    return utils::run_example<NEONCLScaleMedianGaussianExample>(argc, argv);
}
