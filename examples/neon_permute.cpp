/*
 * Copyright (c) 2019 ARM Limited.
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

class NeonPermuteExample : public Example
{
public:
    NeonPermuteExample() = default;

    bool do_setup(int, char **) override
    {
        // Initialise shapes
        init_tensor(TensorShape(8U, 4U, 2U), tensor_nchw, DataType::U8, DataLayout::NCHW);
        init_tensor(TensorShape(2U, 8U, 4U), tensor_nhwc, DataType::U8, DataLayout::NHWC);
        init_tensor(TensorShape(8U, 4U, 2U), tensor_nchw_result, DataType::U8, DataLayout::NCHW);

        // Create the permutation vector to turn a NCHW tensor to NHWC.
        // The input tensor is NCHW, which means that the fastest changing coordinate is W=8U.
        // For permutation vectors the fastest changing coordinate is the one on the left too.
        // Each element in the permutation vector specifies a mapping from the source tensor to the destination one, thus if we
        // use 2U in the permutation vector's first element we are telling the function to move the channels to the fastest
        // changing coordinate in the destination tensor.

        const PermutationVector vector_nchw_to_nhwc(2U, 0U, 1U);
        permute_nhwc.configure(&tensor_nchw, &tensor_nhwc, vector_nchw_to_nhwc);

        // Allocate and fill tensors
        tensor_nhwc.allocator()->allocate();
        tensor_nchw.allocator()->allocate();
        fill_tensor(tensor_nchw);

        // Demostrate autoconfigure for the output tensor
        const PermutationVector vector_nhwc_to_nchw(1U, 2U, 0U);
        permute_nchw.configure(&tensor_nhwc, &tensor_nchw_result, vector_nhwc_to_nchw);
        tensor_nchw_result.allocator()->allocate();

        return true;
    }
    void do_run() override
    {
        permute_nhwc.run();
        permute_nchw.run();
    }
    void do_teardown() override
    {
#if ARM_COMPUTE_DEBUG_ENABLED
        std::cout << "Tensor NCHW" << std::endl;
        tensor_nchw.print(std::cout);
        std::cout << "Tensor NHWC" << std::endl;
        tensor_nhwc.print(std::cout);
#endif // ARM_COMPUTE_DEBUG_ENABLED
    }

private:
    void validate_result(const Tensor &reference, const Tensor &result)
    {
        Window window;
        window.use_tensor_dimensions(reference.info()->tensor_shape());
        Iterator ref_it(&reference, window);
        Iterator res_it(&result, window);
        execute_window_loop(window, [&](const Coordinates &)
        {
            assert(*reinterpret_cast<unsigned char *>(ref_it.ptr()) == *reinterpret_cast<unsigned char *>(res_it.ptr()));
        },
        ref_it, res_it);
    }

    void fill_tensor(Tensor &tensor)
    {
        Window window;
        window.use_tensor_dimensions(tensor.info()->tensor_shape());
        Iterator      tensor_it(&tensor, window);
        unsigned char val(0);
        execute_window_loop(window, [&](const Coordinates &)
        {
            *reinterpret_cast<unsigned char *>(tensor_it.ptr()) = val++;
        },
        tensor_it);
    }
    void init_tensor(const TensorShape shape, Tensor &tensor, DataType type, DataLayout layout)
    {
        tensor.allocator()->init(TensorInfo(shape, 1, type).set_data_layout(layout));
    }

    Tensor    tensor_nchw{};
    Tensor    tensor_nhwc{};
    Tensor    tensor_nchw_result{};
    NEPermute permute_nhwc{};
    NEPermute permute_nchw{};
};

/** Main program that instantiates a permute function example.
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments
 */
int main(int argc, char **argv)
{
    return utils::run_example<NeonPermuteExample>(argc, argv);
}
