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
#include "arm_compute/runtime/CL/CLFunctions.h"

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/CLHelpers.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "utils/Utils.h"

using namespace arm_compute;
using namespace utils;

namespace
{
} // namespace

class CLCacheExample : public Example
{
public:
    CLCacheExample() = default;

    bool do_setup(int argc, char **argv) override
    {
        std::cout << "Once the program has run and created the file cache.bin, rerun with --restore_cache." << std::endl;
        CLScheduler::get().default_init();

        if(argc > 1)
        {
            std::string argv1 = argv[1];
            std::transform(argv1.begin(), argv1.end(), argv1.begin(), ::tolower);
            if(argv1 == "--restore_cache")
            {
                // Load the precompiled kernels from a file into the kernel library, in this way the next time they are needed
                // compilation won't be required.
                restore_program_cache_from_file();
            }
            else
            {
                std::cout << "Unkown option " << argv1 << std::endl;
            }
        }

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

        // Save the opencl kernels to a file
        save_program_cache_to_file();

        return true;
    }
    void do_run() override
    {
        permute_nhwc.run();
        permute_nchw.run();
    }
    void do_teardown() override
    {
    }

private:
    void validate_result(CLTensor &reference, CLTensor &result)
    {
        reference.map();
        result.map();
        Window window;
        window.use_tensor_dimensions(reference.info()->tensor_shape());
        Iterator it_ref(&reference, window);
        Iterator it_res(&result, window);
        execute_window_loop(window, [&](const Coordinates &)
        {
            assert(*reinterpret_cast<unsigned char *>(it_ref.ptr()) == *reinterpret_cast<unsigned char *>(it_res.ptr()));
        },
        it_ref, it_res);
        reference.unmap();
        result.unmap();
    }

    void fill_tensor(CLTensor &tensor)
    {
        tensor.map();
        Window window;
        window.use_tensor_dimensions(tensor.info()->tensor_shape());
        Iterator      it_tensor(&tensor, window);
        unsigned char val(0);
        execute_window_loop(window, [&](const Coordinates &)
        {
            *reinterpret_cast<unsigned char *>(it_tensor.ptr()) = val++;
        },
        it_tensor);
        tensor.unmap();
    }
    void init_tensor(const TensorShape shape, CLTensor &tensor, DataType type, DataLayout layout)
    {
        tensor.allocator()->init(TensorInfo(shape, 1, type).set_data_layout(layout));
    }

    CLTensor  tensor_nchw{};
    CLTensor  tensor_nhwc{};
    CLTensor  tensor_nchw_result{};
    CLPermute permute_nhwc{};
    CLPermute permute_nchw{};
};

/** Main program creating an example that demostrates how to load precompiled kernels from a file.
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments
 */
int main(int argc, char **argv)
{
    return utils::run_example<CLCacheExample>(argc, argv);
}
