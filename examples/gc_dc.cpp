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
#ifndef ARM_COMPUTE_GC
#error "This example needs to be built with -DARM_COMPUTE_GC"
#endif /* ARM_COMPUTE_GC */

#include "arm_compute/runtime/GLES_COMPUTE/GCFunctions.h"
#include "arm_compute/runtime/GLES_COMPUTE/GCScheduler.h"
#include "half/half.hpp"
#include "utils/Utils.h"

using namespace arm_compute;
using namespace utils;

class GCDCExample : public Example
{
public:
    void do_setup(int argc, char **argv) override
    {
        ARM_COMPUTE_UNUSED(argc);
        ARM_COMPUTE_UNUSED(argv);

        // init instance
        GCScheduler::get().default_init();

        const TensorShape  src_shape   = TensorShape{ 11U /* W */, 13U /* H */, 4U /* C */, 3U /* N */ };
        const unsigned int kernel_size = 3;
        const int          stride_x    = 1;
        const int          stride_y    = 1;
        const int          pad_x       = 0;
        const int          pad_y       = 0;
        const unsigned int num_kernels = 256;
        const DataType     data_type   = DataType::F16;

        // generate shape
        const TensorShape   weights_shape(kernel_size, kernel_size, src_shape.z(), num_kernels);
        const TensorShape   bias_shape(num_kernels);
        const PadStrideInfo pad_info(stride_x, stride_y, pad_x, pad_y, DimensionRoundingType::FLOOR);

        // output shape should be 9*11*256*3 (W*H*C*N)
        const TensorShape dst_shape = get_output_shape(src_shape, weights_shape, pad_info);

        // create tensors
        src.allocator()->init(TensorInfo(src_shape, 1, data_type));
        weights.allocator()->init(TensorInfo(weights_shape, 1, data_type));
        bias.allocator()->init(TensorInfo(bias_shape, 1, data_type));
        dst.allocator()->init(TensorInfo(dst_shape, 1, data_type));

        // configure layer
        conv.configure(&src, &weights, &bias, &dst, pad_info);

        // allocate tensors
        src.allocator()->allocate();
        weights.allocator()->allocate();
        bias.allocator()->allocate();
        dst.allocator()->allocate();

        // To demonstrate how to fill tensor with some values...
        src.map();
        Window window;
        window.use_tensor_dimensions(src_shape);
        Iterator it(&src, window);
        execute_window_loop(window, [&](const Coordinates & id)
        {
            *reinterpret_cast<half_float::half *>(it.ptr()) = half_float::half(1.f);
        });
        src.unmap();
    }
    void do_run() override
    {
        // run the layer
        conv.run();
    }
    void do_teardown() override
    {
        // check result
        dst.map();
        // do something
        dst.unmap();
    }

private:
    GCTensor                 src{}, weights{}, bias{}, dst{};
    GCDirectConvolutionLayer conv{};

    TensorShape get_output_shape(TensorShape in_shape, TensorShape kernel_shape, const PadStrideInfo &info)
    {
        TensorShape out_shape(in_shape);
        const std::pair<unsigned int, unsigned int> scaled_dims = scaled_dimensions(in_shape.x(),
                                                                                    in_shape.y(),
                                                                                    kernel_shape.x(),
                                                                                    kernel_shape.y(),
                                                                                    info);
        out_shape.set(0, scaled_dims.first);
        out_shape.set(1, scaled_dims.second);
        out_shape.set(2, kernel_shape[3]);
        return out_shape;
    }
};

/** Main program for directconvolution test
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments
 */
int main(int argc, char **argv)
{
    return utils::run_example<GCDCExample>(argc, argv);
}
