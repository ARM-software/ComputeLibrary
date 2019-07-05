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

#include <chrono>

using namespace arm_compute;
using namespace utils;

namespace
{
/** This function loads prebuilt opencl kernels from a file
 *
 * @param[in] filename Name of the file to be used to load the kernels
 */
void restore_program_cache_from_file(const std::string &filename = "cache.bin")
{
    std::cout << "Loading kernels from file " << filename << std::endl;
    std::ifstream cache_file(filename, std::ios::binary);
    if(cache_file.is_open())
    {
        while(!cache_file.eof())
        {
            size_t name_len   = 0;
            size_t binary_len = 0;
            cache_file.read(reinterpret_cast<char *>(&name_len), sizeof(size_t));
            cache_file.read(reinterpret_cast<char *>(&binary_len), sizeof(size_t));
            if(name_len == 0 || binary_len == 0)
            {
                break;
            }
            std::vector<char>          tmp(name_len);
            std::vector<unsigned char> binary(binary_len);
            std::string                name;
            cache_file.read(tmp.data(), name_len);
            name.assign(tmp.data(), name_len);
            tmp.resize(binary_len);
            cache_file.read(reinterpret_cast<char *>(binary.data()), binary_len);
            cl::Context             context = arm_compute::CLScheduler::get().context();
            cl::Program::Binaries   binaries{ binary };
            std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
            cl::Program             program(context, devices, binaries);
            program.build();
            CLKernelLibrary::get().add_built_program(name, program);
        }
        cache_file.close();
    }
}

/** This function saves opencl kernels library to a file
 *
 * @param[in] filename Name of the file to be used to save the library
 */
void save_program_cache_to_file(const std::string &filename = "cache.bin")
{
    std::cout << "Saving opencl kernels to " << filename << std::endl;
    std::ofstream cache_file(filename, std::ios::binary);
    if(cache_file.is_open())
    {
        for(const auto &it : CLKernelLibrary::get().get_built_programs())
        {
            std::vector<std::vector<unsigned char>> binaries = it.second.getInfo<CL_PROGRAM_BINARIES>();
            ARM_COMPUTE_ERROR_ON(binaries.size() != 1);
            const std::string kernel_name      = it.first;
            size_t            kernel_name_size = kernel_name.length();
            size_t            binary_size      = binaries[0].size();
            cache_file.write(reinterpret_cast<char *>(&kernel_name_size), sizeof(size_t));
            cache_file.write(reinterpret_cast<char *>(&binary_size), sizeof(size_t));
            cache_file.write(kernel_name.c_str(), kernel_name_size);
            cache_file.write(reinterpret_cast<const char *>(binaries[0].data()), binaries[0].size());
        }
        cache_file.close();
    }
}
} // namespace

class CLCacheExample : public Example
{
public:
    CLCacheExample() = default;

    bool do_setup(int argc, char **argv) override
    {
        std::cout << "Once the program has run and created the file cache.bin, rerun with --restore_cache." << std::endl;
        CLScheduler::get().default_init();
        auto start_time = std::chrono::high_resolution_clock::now();
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

        auto end_time        = std::chrono::high_resolution_clock::now();
        auto time_elapsed    = end_time - start_time;
        auto time_elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(time_elapsed).count();
        std::cout << "Configuration time " << time_elapsed_ms << " ms " << std::endl;
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
