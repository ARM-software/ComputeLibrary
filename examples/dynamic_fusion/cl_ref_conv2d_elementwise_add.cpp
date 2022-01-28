/*
 * Copyright (c) 2022 Arm Limited.
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
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/CLTuner.h"
#include "arm_compute/runtime/CL/functions/CLDirectConvolutionLayer.h"
#include "arm_compute/runtime/CL/functions/CLElementwiseOperations.h"

#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "utils/TypePrinter.h"
#include "utils/Utils.h"

#include <cstdlib>

using namespace arm_compute;
using namespace utils;

#define TICK(clock_name) \
    auto clock_name##_tick = std::chrono::high_resolution_clock::now();
#define TOCK(clock_name, measurement_map)                                               \
    auto clock_name##_tock                 = std::chrono::high_resolution_clock::now(); \
    measurement_map["\"" #clock_name "\""] = duration_cast<microseconds>(clock_name##_tock - clock_name##_tick);
#define TOCK_AVG(clock_name, measurement_map, num_iterations)                           \
    auto clock_name##_tock                 = std::chrono::high_resolution_clock::now(); \
    measurement_map["\"" #clock_name "\""] = duration_cast<microseconds>((clock_name##_tock - clock_name##_tick) / (num_iterations));

using std::chrono::duration_cast;
using std::chrono::microseconds;
class ClRefConv2dEltwiseAddExample : public Example
{
public:
    bool do_setup(int argc, char **argv) override
    {
        size_t       ih;
        size_t       iw;
        size_t       ifm;
        size_t       wh;
        size_t       ww;
        size_t       ofm;
        size_t       tuner_choice;
        unsigned int pad_x;
        unsigned int pad_y;
        if(argc < 10)
        {
            // Print help
            std::cout << "Usage:  ./cl_conv2d_elementwise_add ih iw ifm wh ww ofm tuner_choice(0=Disable, 1=Rapid, 2=Normal, 3=Exhaustive)\n";
            std::cout << "Too few or no input_matrices provided. Using shape config = SRGAN_0, tuner_choice=2\n\n";
            ih           = 512;
            iw           = 512;
            ifm          = 64;
            wh           = 1;
            ww           = 1;
            ofm          = 3;
            tuner_choice = 2;
            pad_x        = 0;
            pad_y        = 0;
        }
        else
        {
            ih           = strtol(argv[1], nullptr, 10);
            iw           = strtol(argv[2], nullptr, 10);
            ifm          = strtol(argv[3], nullptr, 10);
            wh           = strtol(argv[4], nullptr, 10);
            ww           = strtol(argv[5], nullptr, 10);
            ofm          = strtol(argv[6], nullptr, 10);
            tuner_choice = strtol(argv[7], nullptr, 10);
            pad_x        = strtol(argv[8], nullptr, 10);
            pad_y        = strtol(argv[9], nullptr, 10);
        }

        CLTuner *tuner_to_use;
        switch(tuner_choice)
        {
            case 0:
            {
                tuner_to_use = nullptr;
                break;
            }
            case 1:
            {
                tuner.set_tuner_mode(CLTunerMode::RAPID);
                tuner_to_use = &tuner;
                break;
            }
            case 3:
            {
                tuner.set_tuner_mode(CLTunerMode::EXHAUSTIVE);
                tuner_to_use = &tuner;
                break;
            }
            case 2:
            default:
            {
                tuner.set_tuner_mode(CLTunerMode::NORMAL);
                tuner_to_use = &tuner;
                break;
            }
        }

        CLScheduler::get().default_init(tuner_to_use);

        TICK(startup_time);

        /* Computation:
         * out = add_desc(addend, conv2d1x1(direct_conv)(input, weights, bias))
         */
        const auto          data_type   = DataType::F32;
        const auto          data_layout = DataLayout::NHWC;
        const PadStrideInfo conv_info{ 1, 1, pad_x, pad_y };
        // const auto t_input_shape    = TensorShape(384, 12, 12);
        // const auto t_weight_shape   = TensorShape(384, 1, 1, 64);
        // const auto t_dst_shape      = TensorShape(64, 12, 12);
        const auto t_input_shape  = TensorShape(ifm, iw, ih);
        const auto t_weight_shape = TensorShape(ifm, ww, wh, ofm);
        const auto t_dst_shape    = misc::shape_calculator::compute_deep_convolution_shape(t_input_shape, data_layout, t_weight_shape, conv_info);
        std::cout << "input_shape: " << t_input_shape << std::endl;
        std::cout << "weight_shape: " << t_weight_shape << std::endl;
        std::cout << "dst_shape: " << t_dst_shape << std::endl;
        auto t_input_info     = TensorInfo(t_input_shape, 1, data_type, data_layout);
        auto t_weight_info    = TensorInfo(t_weight_shape, 1, data_type, data_layout);
        auto t_l0_dst_info    = TensorInfo(t_dst_shape, 1, data_type, data_layout); // Intermediate tensor for cond3
        auto t_l1_addend_info = TensorInfo(t_dst_shape, 1, data_type, data_layout);
        auto t_dst_info       = TensorInfo(t_dst_shape, 1, data_type, data_layout);

        // Init tensors
        {
            t_input.allocator()->init(t_input_info);
            t_weight.allocator()->init(t_weight_info);
            t_l1_addend.allocator()->init(t_dst_info);
            t_l0_dst.allocator()->init(t_l0_dst_info);
            t_dst.allocator()->init(t_dst_info);
        }

        op0.configure(&t_input, &t_weight, nullptr, &t_l0_dst, conv_info);
        op1.configure(&t_l0_dst, &t_l1_addend, &t_dst, ConvertPolicy{});

        // Construct tensors
        // Allocate and fill tensors
        {
            t_input.allocator()->allocate();
            t_weight.allocator()->allocate();
            t_l1_addend.allocator()->allocate();
            t_l0_dst.allocator()->allocate();
            t_dst.allocator()->allocate();
            fill_random_tensor(t_input, -1.f, 1.f);
            fill_random_tensor(t_weight, -1.f, 1.f);
            fill_random_tensor(t_l1_addend, -1.f, 1.f);
        }
        // Dummy run for CLTuner
        op0.run();
        op1.run();
        TOCK(startup_time, measurements);
        return true;
    }
    void do_run() override
    {
        // Run the fused op
        op0.run();
        op1.run();

        // Make sure all the OpenCL jobs are done executing:
        CLScheduler::get().sync();
    }

    void do_teardown() override
    {
        for(auto m : measurements)
        {
            std::cout << m.first << ": " << m.second.count() << "us" << std::endl;
        }
    }

private:
    CLTensor                 t_input{};
    CLTensor                 t_weight{};
    CLTensor                 t_l1_addend{};
    CLTensor                 t_l0_dst{};
    CLTensor                 t_dst{};
    CLDirectConvolutionLayer op0{};
    CLArithmeticAddition     op1{};
    CLTuner                  tuner{};
    std::map<std::string, std::chrono::microseconds> measurements{};
};

/** Main program for sgemm test
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments ( [optional] Matrix A, [optional] Matrix B, [optional] Matrix C, [optional] alpha, [optional] beta )
 */
int main(int argc, char **argv)
{
    return utils::run_example<ClRefConv2dEltwiseAddExample>(argc, argv);
}

#undef TICK
#undef TOCK
#undef TOCK_AVG