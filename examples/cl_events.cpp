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
#ifndef ARM_COMPUTE_CL /* Needed by Utils.cpp to handle OpenCL exceptions properly */
#error "This example needs to be built with -DARM_COMPUTE_CL"
#endif /* ARM_COMPUTE_CL */

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/CLFunctions.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "utils/Utils.h"

using namespace arm_compute;
using namespace utils;

class CLEventsExample : public Example
{
public:
    void do_setup(int argc, char **argv) override
    {
        /** [OpenCL events] **/
        PPMLoader     ppm;
        constexpr int scale_factor = 2;

        CLScheduler::get().default_init();

        if(argc < 2)
        {
            // Print help
            std::cout << "Usage: ./build/cl_events [input_image.ppm]\n\n";
            std::cout << "No input_image provided, creating a dummy 640x480 image\n";
            // Create an empty grayscale 640x480 image
            src.allocator()->init(TensorInfo(640, 480, Format::U8));
        }
        else
        {
            ppm.open(argv[1]);
            ppm.init_image(src, Format::U8);
        }

        TensorInfo dst_info(src.info()->dimension(0) / scale_factor, src.info()->dimension(1) / scale_factor, Format::U8);

        // Configure the temporary and destination images
        dst.allocator()->init(dst_info);
        tmp_scale_median.allocator()->init(dst_info);
        tmp_median_gauss.allocator()->init(dst_info);

        //Configure the functions:
        scale.configure(&src, &tmp_scale_median, InterpolationPolicy::NEAREST_NEIGHBOR, BorderMode::REPLICATE);
        median.configure(&tmp_scale_median, &tmp_median_gauss, BorderMode::REPLICATE);
        gauss.configure(&tmp_median_gauss, &dst, BorderMode::REPLICATE);

        // Allocate all the images
        src.allocator()->allocate();
        dst.allocator()->allocate();
        tmp_scale_median.allocator()->allocate();
        tmp_median_gauss.allocator()->allocate();

        // Fill the input image with the content of the PPM image if a filename was provided:
        if(ppm.is_open())
        {
            ppm.fill_image(src);
            output_filename = std::string(argv[1]) + "_out.ppm";
        }
        /** [OpenCL events] **/
    }
    void do_run() override
    {
        // Enqueue and flush the scale OpenCL kernel:
        scale.run();
        // Create a synchronisation event between scale and median:
        cl::Event scale_event = CLScheduler::get().enqueue_sync_event();
        // Enqueue and flush the median OpenCL kernel:
        median.run();
        // Enqueue and flush the Gaussian OpenCL kernel:
        gauss.run();

        //Make sure all the OpenCL jobs are done executing:
        scale_event.wait();        // Block until Scale is done executing (Median3x3 and Gaussian5x5 might still be running)
        CLScheduler::get().sync(); // Block until Gaussian5x5 is done executing
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
    CLImage       src{}, tmp_scale_median{}, tmp_median_gauss{}, dst{};
    CLScale       scale{};
    CLMedian3x3   median{};
    CLGaussian5x5 gauss{};
    std::string   output_filename{};
};

/** Main program for convolution test
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments ( [optional] Path to PPM image to process )
 */
int main(int argc, char **argv)
{
    return utils::run_example<CLEventsExample>(argc, argv);
}
