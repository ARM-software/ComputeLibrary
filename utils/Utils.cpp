/*
 * Copyright (c) 2017-2019 ARM Limited.
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
#include "Utils.h"

#ifdef ARM_COMPUTE_CL
#include "arm_compute/runtime/CL/CLScheduler.h"
#endif /* ARM_COMPUTE_CL */

#include <cctype>
#include <cerrno>
#include <iomanip>
#include <string>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wswitch-default"
#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#pragma GCC diagnostic pop

namespace arm_compute
{
namespace utils
{
namespace
{
/* Advance the iterator to the first character which is not a comment
 *
 * @param[in,out] fs Stream to drop comments from
 */
void discard_comments(std::ifstream &fs)
{
    while(fs.peek() == '#')
    {
        fs.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }
}

/* Advance the string iterator to the next character which is neither a space or a comment
 *
 * @param[in,out] fs Stream to drop comments from
 */
void discard_comments_and_spaces(std::ifstream &fs)
{
    while(true)
    {
        discard_comments(fs);

        if(isspace(fs.peek()) == 0)
        {
            break;
        }

        fs.ignore(1);
    }
}
} // namespace

#ifndef BENCHMARK_EXAMPLES
int run_example(int argc, char **argv, std::unique_ptr<Example> example)
{
    std::cout << "\n"
              << argv[0] << "\n\n";

    try
    {
        bool status = example->do_setup(argc, argv);
        if(!status)
        {
            return 1;
        }
        example->do_run();
        example->do_teardown();

        std::cout << "\nTest passed\n";
        return 0;
    }
#ifdef ARM_COMPUTE_CL
    catch(cl::Error &err)
    {
        std::cerr << "!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
        std::cerr << std::endl
                  << "ERROR " << err.what() << "(" << err.err() << ")" << std::endl;
        std::cerr << "!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
    }
#endif /* ARM_COMPUTE_CL */
    catch(std::runtime_error &err)
    {
        std::cerr << "!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
        std::cerr << std::endl
                  << "ERROR " << err.what() << " " << (errno ? strerror(errno) : "") << std::endl;
        std::cerr << "!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
    }

    std::cout << "\nTest FAILED\n";

    return -1;
}
#endif /* BENCHMARK_EXAMPLES */

void draw_detection_rectangle(ITensor *tensor, const DetectionWindow &rect, uint8_t r, uint8_t g, uint8_t b)
{
    ARM_COMPUTE_ERROR_ON_FORMAT_NOT_IN(tensor, Format::RGB888);

    uint8_t *top    = tensor->info()->offset_element_in_bytes(Coordinates(rect.x, rect.y)) + tensor->buffer();
    uint8_t *bottom = tensor->info()->offset_element_in_bytes(Coordinates(rect.x, rect.y + rect.height)) + tensor->buffer();
    uint8_t *left   = top;
    uint8_t *right  = tensor->info()->offset_element_in_bytes(Coordinates(rect.x + rect.width, rect.y)) + tensor->buffer();
    size_t   stride = tensor->info()->strides_in_bytes()[Window::DimY];

    for(size_t x = 0; x < rect.width; ++x)
    {
        top[0]    = r;
        top[1]    = g;
        top[2]    = b;
        bottom[0] = r;
        bottom[1] = g;
        bottom[2] = b;

        top += 3;
        bottom += 3;
    }

    for(size_t y = 0; y < rect.height; ++y)
    {
        left[0]  = r;
        left[1]  = g;
        left[2]  = b;
        right[0] = r;
        right[1] = g;
        right[2] = b;

        left += stride;
        right += stride;
    }
}

ImageType get_image_type_from_file(const std::string &filename)
{
    ImageType type = ImageType::UNKNOWN;

    try
    {
        // Open file
        std::ifstream fs;
        fs.exceptions(std::ifstream::failbit | std::ifstream::badbit);
        fs.open(filename, std::ios::in | std::ios::binary);

        // Identify type from magic number
        std::array<unsigned char, 2> magic_number{ { 0 } };
        fs >> magic_number[0] >> magic_number[1];

        // PPM check
        if(static_cast<char>(magic_number[0]) == 'P' && static_cast<char>(magic_number[1]) == '6')
        {
            type = ImageType::PPM;
        }
        else if(magic_number[0] == 0xFF && magic_number[1] == 0xD8)
        {
            type = ImageType::JPEG;
        }

        fs.close();
    }
    catch(std::runtime_error &e)
    {
        ARM_COMPUTE_ERROR("Accessing %s: %s", filename.c_str(), e.what());
    }

    return type;
}

std::tuple<unsigned int, unsigned int, int> parse_ppm_header(std::ifstream &fs)
{
    // Check the PPM magic number is valid
    std::array<char, 2> magic_number{ { 0 } };
    fs >> magic_number[0] >> magic_number[1];
    ARM_COMPUTE_ERROR_ON_MSG(magic_number[0] != 'P' || magic_number[1] != '6', "Invalid file type");
    ARM_COMPUTE_UNUSED(magic_number);

    discard_comments_and_spaces(fs);

    unsigned int width = 0;
    fs >> width;

    discard_comments_and_spaces(fs);

    unsigned int height = 0;
    fs >> height;

    discard_comments_and_spaces(fs);

    int max_val = 0;
    fs >> max_val;

    discard_comments(fs);

    ARM_COMPUTE_ERROR_ON_MSG(isspace(fs.peek()) == 0, "Invalid PPM header");
    fs.ignore(1);

    return std::make_tuple(width, height, max_val);
}

std::tuple<std::vector<unsigned long>, bool, std::string> parse_npy_header(std::ifstream &fs) //NOLINT
{
    std::vector<unsigned long> shape; // NOLINT

    // Read header
    std::string header = npy::read_header(fs);

    // Parse header
    bool        fortran_order = false;
    std::string typestr;
    npy::parse_header(header, typestr, fortran_order, shape);

    std::reverse(shape.begin(), shape.end());

    return std::make_tuple(shape, fortran_order, typestr);
}

/** This function returns the amount of memory free reading from /proc/meminfo
 *
 * @return The free memory in kB
 */
uint64_t get_mem_free_from_meminfo()
{
    std::string   line_attribute;
    std::ifstream file_meminfo("/proc/meminfo");

    if(file_meminfo.is_open())
    {
        while(!(file_meminfo >> line_attribute).fail())
        {
            //Test if is the line containing MemFree
            if(line_attribute == "MemFree:")
            {
                uint64_t mem_available;
                if(!(file_meminfo >> mem_available).fail())
                {
                    return mem_available;
                }
                else
                {
                    return 0;
                }
            }
            // if it's not MemFree ignore rest of the line
            file_meminfo.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        }
    }
    // Nothing found or an error during opening the file
    return 0;
}

/** This function loads prebuilt opencl kernels from a file
 *
 * @param[in] filename Name of the file to be used to load the kernels
 */
void restore_program_cache_from_file(const std::string &filename)
{
#ifdef ARM_COMPUTE_CL
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
#endif /* ARM_COMPUTE_CL */
}

/** This function saves opencl kernels library to a file
 *
 * @param[in] filename Name of the file to be used to save the library
 */
void save_program_cache_to_file(const std::string &filename)
{
#ifdef ARM_COMPUTE_CL
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
#endif /* ARM_COMPUTE_CL */
}
} // namespace utils
} // namespace arm_compute
