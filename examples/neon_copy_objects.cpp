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

#include "arm_compute/runtime/NEON/NEFunctions.h"

#include "arm_compute/core/Types.h"
#include "utils/Utils.h"

#include <cstring>
#include <iostream>

using namespace arm_compute;
using namespace utils;

class NEONCopyObjectsExample : public Example
{
public:
    bool do_setup(int argc, char **argv) override
    {
        ARM_COMPUTE_UNUSED(argc);
        ARM_COMPUTE_UNUSED(argv);

        /** [Copy objects example] */
        constexpr unsigned int width  = 4;
        constexpr unsigned int height = 3;
        constexpr unsigned int batch  = 2;

        src_data = new float[width * height * batch];
        dst_data = new float[width * height * batch];

        // Fill src_data with dummy values:
        for(unsigned int b = 0; b < batch; b++)
        {
            for(unsigned int h = 0; h < height; h++)
            {
                for(unsigned int w = 0; w < width; w++)
                {
                    src_data[b * (width * height) + h * width + w] = static_cast<float>(100 * b + 10 * h + w);
                }
            }
        }

        // Initialize the tensors dimensions and type:
        const TensorShape shape(width, height, batch);
        input.allocator()->init(TensorInfo(shape, 1, DataType::F32));
        output.allocator()->init(TensorInfo(shape, 1, DataType::F32));

        // Configure softmax:
        softmax.configure(&input, &output);

        // Allocate the input / output tensors:
        input.allocator()->allocate();
        output.allocator()->allocate();

        // Fill the input tensor:
        // Simplest way: create an iterator to iterate through each element of the input tensor:
        Window input_window;
        input_window.use_tensor_dimensions(input.info()->tensor_shape());
        std::cout << " Dimensions of the input's iterator:\n";
        std::cout << " X = [start=" << input_window.x().start() << ", end=" << input_window.x().end() << ", step=" << input_window.x().step() << "]\n";
        std::cout << " Y = [start=" << input_window.y().start() << ", end=" << input_window.y().end() << ", step=" << input_window.y().step() << "]\n";
        std::cout << " Z = [start=" << input_window.z().start() << ", end=" << input_window.z().end() << ", step=" << input_window.z().step() << "]\n";

        // Create an iterator:
        Iterator input_it(&input, input_window);

        // Iterate through the elements of src_data and copy them one by one to the input tensor:
        // This is equivalent to:
        // for( unsigned int z = 0; z < batch; ++z)
        // {
        //   for( unsigned int y = 0; y < height; ++y)
        //   {
        //     for( unsigned int x = 0; x < width; ++x)
        //     {
        //       *reinterpret_cast<float*>( input.buffer() + input.info()->offset_element_in_bytes(Coordinates(x,y,z))) = src_data[ z * (width*height) + y * width + x];
        //     }
        //   }
        // }
        // Except it works for an arbitrary number of dimensions
        execute_window_loop(input_window, [&](const Coordinates & id)
        {
            std::cout << "Setting item [" << id.x() << "," << id.y() << "," << id.z() << "]\n";
            *reinterpret_cast<float *>(input_it.ptr()) = src_data[id.z() * (width * height) + id.y() * width + id.x()];
        },
        input_it);

        // More efficient way: create an iterator to iterate through each row (instead of each element) of the output tensor:
        Window output_window;
        output_window.use_tensor_dimensions(output.info()->tensor_shape(), /* first_dimension =*/Window::DimY); // Iterate through the rows (not each element)
        std::cout << " Dimensions of the output's iterator:\n";
        std::cout << " X = [start=" << output_window.x().start() << ", end=" << output_window.x().end() << ", step=" << output_window.x().step() << "]\n";
        std::cout << " Y = [start=" << output_window.y().start() << ", end=" << output_window.y().end() << ", step=" << output_window.y().step() << "]\n";
        std::cout << " Z = [start=" << output_window.z().start() << ", end=" << output_window.z().end() << ", step=" << output_window.z().step() << "]\n";

        // Create an iterator:
        Iterator output_it(&output, output_window);

        // Iterate through the rows of the output tensor and copy them to dst_data:
        // This is equivalent to:
        // for( unsigned int z = 0; z < batch; ++z)
        // {
        //   for( unsigned int y = 0; y < height; ++y)
        //   {
        //     memcpy( dst_data + z * (width*height) + y * width, input.buffer() + input.info()->offset_element_in_bytes(Coordinates(0,y,z)), width * sizeof(float));
        //   }
        // }
        // Except it works for an arbitrary number of dimensions
        execute_window_loop(output_window, [&](const Coordinates & id)
        {
            std::cout << "Copying one row starting from [" << id.x() << "," << id.y() << "," << id.z() << "]\n";
            // Copy one whole row:
            memcpy(dst_data + id.z() * (width * height) + id.y() * width, output_it.ptr(), width * sizeof(float));
        },
        output_it);

        /** [Copy objects example] */

        return true;
    }
    void do_run() override
    {
        // Run Neon softmax:
        softmax.run();
    }
    void do_teardown() override
    {
        delete[] src_data;
        delete[] dst_data;
    }

private:
    Tensor         input{}, output{};
    float         *src_data{};
    float         *dst_data{};
    NESoftmaxLayer softmax{};
};
/** Main program for the copy objects test
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments
 */
int main(int argc, char **argv)
{
    return utils::run_example<NEONCopyObjectsExample>(argc, argv);
}
