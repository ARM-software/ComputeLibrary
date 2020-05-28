/*
 * Copyright (c) 2020 ARM Limited.
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
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/runtime/CL/CLScheduler.h"

#include <fstream>
#include <map>
#include <string>

namespace arm_compute
{
void restore_program_cache_from_file(const std::string &filename)
{
    std::ifstream cache_file(filename, std::ios::binary);
    if(cache_file.is_open())
    {
        if(!CLScheduler::get().is_initialised())
        {
            arm_compute::CLScheduler::get().default_init();
        }

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

void save_program_cache_to_file(const std::string &filename)
{
    if(CLScheduler::get().is_initialised())
    {
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
        else
        {
            ARM_COMPUTE_ERROR("Cannot open cache file");
        }
    }
}
} // namespace arm_compute
