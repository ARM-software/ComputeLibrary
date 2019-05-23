/*
 * Copyright (c) 2016-2019 ARM Limited.
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
#ifndef __UTILS_UTILS_H__
#define __UTILS_UTILS_H__

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"
#include "arm_compute/runtime/Tensor.h"
#include "libnpy/npy.hpp"
#include "support/ToolchainSupport.h"

#ifdef ARM_COMPUTE_CL
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/runtime/CL/CLDistribution1D.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#endif /* ARM_COMPUTE_CL */
#ifdef ARM_COMPUTE_GC
#include "arm_compute/runtime/GLES_COMPUTE/GCTensor.h"
#endif /* ARM_COMPUTE_GC */

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <tuple>
#include <vector>

namespace arm_compute
{
namespace utils
{
/** Supported image types */
enum class ImageType
{
    UNKNOWN,
    PPM,
    JPEG
};

/** Abstract Example class.
 *
 * All examples have to inherit from this class.
 */
class Example
{
public:
    /** Setup the example.
     *
     * @param[in] argc Argument count.
     * @param[in] argv Argument values.
     *
     * @return True in case of no errors in setup else false
     */
    virtual bool do_setup(int argc, char **argv)
    {
        return true;
    };
    /** Run the example. */
    virtual void do_run() {};
    /** Teardown the example. */
    virtual void do_teardown() {};

    /** Default destructor. */
    virtual ~Example() = default;
};

/** Run an example and handle the potential exceptions it throws
 *
 * @param[in] argc    Number of command line arguments
 * @param[in] argv    Command line arguments
 * @param[in] example Example to run
 */
int run_example(int argc, char **argv, std::unique_ptr<Example> example);

template <typename T>
int run_example(int argc, char **argv)
{
    return run_example(argc, argv, support::cpp14::make_unique<T>());
}

/** Draw a RGB rectangular window for the detected object
 *
 * @param[in, out] tensor Input tensor where the rectangle will be drawn on. Format supported: RGB888
 * @param[in]      rect   Geometry of the rectangular window
 * @param[in]      r      Red colour to use
 * @param[in]      g      Green colour to use
 * @param[in]      b      Blue colour to use
 */
void draw_detection_rectangle(arm_compute::ITensor *tensor, const arm_compute::DetectionWindow &rect, uint8_t r, uint8_t g, uint8_t b);

/** Gets image type given a file
 *
 * @param[in] filename File to identify its image type
 *
 * @return Image type
 */
ImageType get_image_type_from_file(const std::string &filename);

/** Parse the ppm header from an input file stream. At the end of the execution,
 *  the file position pointer will be located at the first pixel stored in the ppm file
 *
 * @param[in] fs Input file stream to parse
 *
 * @return The width, height and max value stored in the header of the PPM file
 */
std::tuple<unsigned int, unsigned int, int> parse_ppm_header(std::ifstream &fs);

/** Parse the npy header from an input file stream. At the end of the execution,
 *  the file position pointer will be located at the first pixel stored in the npy file //TODO
 *
 * @param[in] fs Input file stream to parse
 *
 * @return The width and height stored in the header of the NPY file
 */
std::tuple<std::vector<unsigned long>, bool, std::string> parse_npy_header(std::ifstream &fs);

/** Obtain numpy type string from DataType.
 *
 * @param[in] data_type Data type.
 *
 * @return numpy type string.
 */
inline std::string get_typestring(DataType data_type)
{
    // Check endianness
    const unsigned int i = 1;
    const char        *c = reinterpret_cast<const char *>(&i);
    std::string        endianness;
    if(*c == 1)
    {
        endianness = std::string("<");
    }
    else
    {
        endianness = std::string(">");
    }
    const std::string no_endianness("|");

    switch(data_type)
    {
        case DataType::U8:
        case DataType::QASYMM8:
            return no_endianness + "u" + support::cpp11::to_string(sizeof(uint8_t));
        case DataType::S8:
            return no_endianness + "i" + support::cpp11::to_string(sizeof(int8_t));
        case DataType::U16:
            return endianness + "u" + support::cpp11::to_string(sizeof(uint16_t));
        case DataType::S16:
            return endianness + "i" + support::cpp11::to_string(sizeof(int16_t));
        case DataType::U32:
            return endianness + "u" + support::cpp11::to_string(sizeof(uint32_t));
        case DataType::S32:
            return endianness + "i" + support::cpp11::to_string(sizeof(int32_t));
        case DataType::U64:
            return endianness + "u" + support::cpp11::to_string(sizeof(uint64_t));
        case DataType::S64:
            return endianness + "i" + support::cpp11::to_string(sizeof(int64_t));
        case DataType::F16:
            return endianness + "f" + support::cpp11::to_string(sizeof(half));
        case DataType::F32:
            return endianness + "f" + support::cpp11::to_string(sizeof(float));
        case DataType::F64:
            return endianness + "f" + support::cpp11::to_string(sizeof(double));
        case DataType::SIZET:
            return endianness + "u" + support::cpp11::to_string(sizeof(size_t));
        default:
            ARM_COMPUTE_ERROR("Data type not supported");
    }
}

/** Maps a tensor if needed
 *
 * @param[in] tensor   Tensor to be mapped
 * @param[in] blocking Specified if map is blocking or not
 */
template <typename T>
inline void map(T &tensor, bool blocking)
{
    ARM_COMPUTE_UNUSED(tensor);
    ARM_COMPUTE_UNUSED(blocking);
}

/** Unmaps a tensor if needed
 *
 * @param tensor  Tensor to be unmapped
 */
template <typename T>
inline void unmap(T &tensor)
{
    ARM_COMPUTE_UNUSED(tensor);
}

#ifdef ARM_COMPUTE_CL
/** Maps a tensor if needed
 *
 * @param[in] tensor   Tensor to be mapped
 * @param[in] blocking Specified if map is blocking or not
 */
inline void map(CLTensor &tensor, bool blocking)
{
    tensor.map(blocking);
}

/** Unmaps a tensor if needed
 *
 * @param tensor  Tensor to be unmapped
 */
inline void unmap(CLTensor &tensor)
{
    tensor.unmap();
}

/** Maps a distribution if needed
 *
 * @param[in] distribution Distribution to be mapped
 * @param[in] blocking     Specified if map is blocking or not
 */
inline void map(CLDistribution1D &distribution, bool blocking)
{
    distribution.map(blocking);
}

/** Unmaps a distribution if needed
 *
 * @param distribution  Distribution to be unmapped
 */
inline void unmap(CLDistribution1D &distribution)
{
    distribution.unmap();
}
#endif /* ARM_COMPUTE_CL */

#ifdef ARM_COMPUTE_GC
/** Maps a tensor if needed
 *
 * @param[in] tensor   Tensor to be mapped
 * @param[in] blocking Specified if map is blocking or not
 */
inline void map(GCTensor &tensor, bool blocking)
{
    tensor.map(blocking);
}

/** Unmaps a tensor if needed
 *
 * @param tensor  Tensor to be unmapped
 */
inline void unmap(GCTensor &tensor)
{
    tensor.unmap();
}
#endif /* ARM_COMPUTE_GC */

/** Specialized class to generate random non-zero FP16 values.
 *  uniform_real_distribution<half> generates values that get rounded off to zero, causing
 *  differences between ACL and reference implementation
*/
class uniform_real_distribution_fp16
{
    half                                   min{ 0.0f }, max{ 0.0f };
    std::uniform_real_distribution<float>  neg{ min, -0.3f };
    std::uniform_real_distribution<float>  pos{ 0.3f, max };
    std::uniform_int_distribution<uint8_t> sign_picker{ 0, 1 };

public:
    using result_type = half;
    /** Constructor
     *
     * @param[in] a Minimum value of the distribution
     * @param[in] b Maximum value of the distribution
     */
    explicit uniform_real_distribution_fp16(half a = half(0.0), half b = half(1.0))
        : min(a), max(b)
    {
    }

    /** () operator to generate next value
     *
     * @param[in] gen an uniform random bit generator object
     */
    half operator()(std::mt19937 &gen)
    {
        if(sign_picker(gen))
        {
            return (half)neg(gen);
        }
        return (half)pos(gen);
    }
};

/** Numpy data loader */
class NPYLoader
{
public:
    /** Default constructor */
    NPYLoader()
        : _fs(), _shape(), _fortran_order(false), _typestring(), _file_layout(DataLayout::NCHW)
    {
    }

    /** Open a NPY file and reads its metadata
     *
     * @param[in] npy_filename File to open
     * @param[in] file_layout  (Optional) Layout in which the weights are stored in the file.
     */
    void open(const std::string &npy_filename, DataLayout file_layout = DataLayout::NCHW)
    {
        ARM_COMPUTE_ERROR_ON(is_open());
        try
        {
            _fs.open(npy_filename, std::ios::in | std::ios::binary);
            ARM_COMPUTE_EXIT_ON_MSG(!_fs.good(), "Failed to load binary data from %s", npy_filename.c_str());
            _fs.exceptions(std::ifstream::failbit | std::ifstream::badbit);
            _file_layout = file_layout;

            std::tie(_shape, _fortran_order, _typestring) = parse_npy_header(_fs);
        }
        catch(const std::ifstream::failure &e)
        {
            ARM_COMPUTE_ERROR("Accessing %s: %s", npy_filename.c_str(), e.what());
        }
    }
    /** Return true if a NPY file is currently open */
    bool is_open()
    {
        return _fs.is_open();
    }

    /** Return true if a NPY file is in fortran order */
    bool is_fortran()
    {
        return _fortran_order;
    }

    /** Initialise the tensor's metadata with the dimensions of the NPY file currently open
     *
     * @param[out] tensor Tensor to initialise
     * @param[in]  dt     Data type to use for the tensor
     */
    template <typename T>
    void init_tensor(T &tensor, arm_compute::DataType dt)
    {
        ARM_COMPUTE_ERROR_ON(!is_open());
        ARM_COMPUTE_ERROR_ON(dt != arm_compute::DataType::F32);

        // Use the size of the input NPY tensor
        TensorShape shape;
        shape.set_num_dimensions(_shape.size());
        for(size_t i = 0; i < _shape.size(); ++i)
        {
            size_t src = i;
            if(_fortran_order)
            {
                src = _shape.size() - 1 - i;
            }
            shape.set(i, _shape.at(src));
        }

        arm_compute::TensorInfo tensor_info(shape, 1, dt);
        tensor.allocator()->init(tensor_info);
    }

    /** Fill a tensor with the content of the currently open NPY file.
     *
     * @note If the tensor is a CLTensor, the function maps and unmaps the tensor
     *
     * @param[in,out] tensor Tensor to fill (Must be allocated, and of matching dimensions with the opened NPY).
     */
    template <typename T>
    void fill_tensor(T &tensor)
    {
        ARM_COMPUTE_ERROR_ON(!is_open());
        ARM_COMPUTE_ERROR_ON_DATA_TYPE_NOT_IN(&tensor, arm_compute::DataType::QASYMM8, arm_compute::DataType::S32, arm_compute::DataType::F32);
        try
        {
            // Map buffer if creating a CLTensor
            map(tensor, true);

            // Check if the file is large enough to fill the tensor
            const size_t current_position = _fs.tellg();
            _fs.seekg(0, std::ios_base::end);
            const size_t end_position = _fs.tellg();
            _fs.seekg(current_position, std::ios_base::beg);

            ARM_COMPUTE_ERROR_ON_MSG((end_position - current_position) < tensor.info()->tensor_shape().total_size() * tensor.info()->element_size(),
                                     "Not enough data in file");
            ARM_COMPUTE_UNUSED(end_position);

            // Check if the typestring matches the given one
            std::string expect_typestr = get_typestring(tensor.info()->data_type());
            ARM_COMPUTE_ERROR_ON_MSG(_typestring != expect_typestr, "Typestrings mismatch");

            bool are_layouts_different = (_file_layout != tensor.info()->data_layout());
            // Correct dimensions (Needs to match TensorShape dimension corrections)
            if(_shape.size() != tensor.info()->tensor_shape().num_dimensions())
            {
                for(int i = static_cast<int>(_shape.size()) - 1; i > 0; --i)
                {
                    if(_shape[i] == 1)
                    {
                        _shape.pop_back();
                    }
                    else
                    {
                        break;
                    }
                }
            }

            TensorShape                    permuted_shape = tensor.info()->tensor_shape();
            arm_compute::PermutationVector perm;
            if(are_layouts_different && tensor.info()->tensor_shape().num_dimensions() > 2)
            {
                perm                                    = (tensor.info()->data_layout() == arm_compute::DataLayout::NHWC) ? arm_compute::PermutationVector(2U, 0U, 1U) : arm_compute::PermutationVector(1U, 2U, 0U);
                arm_compute::PermutationVector perm_vec = (tensor.info()->data_layout() == arm_compute::DataLayout::NCHW) ? arm_compute::PermutationVector(2U, 0U, 1U) : arm_compute::PermutationVector(1U, 2U, 0U);

                arm_compute::permute(permuted_shape, perm_vec);
            }

            // Validate tensor shape
            ARM_COMPUTE_ERROR_ON_MSG(_shape.size() != tensor.info()->tensor_shape().num_dimensions(), "Tensor ranks mismatch");
            for(size_t i = 0; i < _shape.size(); ++i)
            {
                ARM_COMPUTE_ERROR_ON_MSG(permuted_shape[i] != _shape[i], "Tensor dimensions mismatch");
            }

            switch(tensor.info()->data_type())
            {
                case arm_compute::DataType::QASYMM8:
                case arm_compute::DataType::S32:
                case arm_compute::DataType::F32:
                case arm_compute::DataType::F16:
                {
                    // Read data
                    if(!are_layouts_different && !_fortran_order && tensor.info()->padding().empty())
                    {
                        // If tensor has no padding read directly from stream.
                        _fs.read(reinterpret_cast<char *>(tensor.buffer()), tensor.info()->total_size());
                    }
                    else
                    {
                        // If tensor has padding or is in fortran order accessing tensor elements through execution window.
                        Window             window;
                        const unsigned int num_dims = _shape.size();
                        if(_fortran_order)
                        {
                            for(unsigned int dim = 0; dim < num_dims; dim++)
                            {
                                permuted_shape.set(dim, _shape[num_dims - dim - 1]);
                                perm.set(dim, num_dims - dim - 1);
                            }
                            if(are_layouts_different)
                            {
                                // Permute only if num_dimensions greater than 2
                                if(num_dims > 2)
                                {
                                    if(_file_layout == DataLayout::NHWC) // i.e destination is NCHW --> permute(1,2,0)
                                    {
                                        arm_compute::permute(perm, arm_compute::PermutationVector(1U, 2U, 0U));
                                    }
                                    else
                                    {
                                        arm_compute::permute(perm, arm_compute::PermutationVector(2U, 0U, 1U));
                                    }
                                }
                            }
                        }
                        window.use_tensor_dimensions(permuted_shape);

                        execute_window_loop(window, [&](const Coordinates & id)
                        {
                            Coordinates dst(id);
                            arm_compute::permute(dst, perm);
                            _fs.read(reinterpret_cast<char *>(tensor.ptr_to_element(dst)), tensor.info()->element_size());
                        });
                    }

                    break;
                }
                default:
                    ARM_COMPUTE_ERROR("Unsupported data type");
            }

            // Unmap buffer if creating a CLTensor
            unmap(tensor);
        }
        catch(const std::ifstream::failure &e)
        {
            ARM_COMPUTE_ERROR("Loading NPY file: %s", e.what());
        }
    }

private:
    std::ifstream              _fs;
    std::vector<unsigned long> _shape;
    bool                       _fortran_order;
    std::string                _typestring;
    DataLayout                 _file_layout;
};

/** Template helper function to save a tensor image to a PPM file.
 *
 * @note Only U8 and RGB888 formats supported.
 * @note Only works with 2D tensors.
 * @note If the input tensor is a CLTensor, the function maps and unmaps the image
 *
 * @param[in] tensor       The tensor to save as PPM file
 * @param[in] ppm_filename Filename of the file to create.
 */
template <typename T>
void save_to_ppm(T &tensor, const std::string &ppm_filename)
{
    ARM_COMPUTE_ERROR_ON_FORMAT_NOT_IN(&tensor, arm_compute::Format::RGB888, arm_compute::Format::U8);
    ARM_COMPUTE_ERROR_ON(tensor.info()->num_dimensions() > 2);

    std::ofstream fs;

    try
    {
        fs.exceptions(std::ofstream::failbit | std::ofstream::badbit | std::ofstream::eofbit);
        fs.open(ppm_filename, std::ios::out | std::ios::binary);

        const unsigned int width  = tensor.info()->tensor_shape()[0];
        const unsigned int height = tensor.info()->tensor_shape()[1];

        fs << "P6\n"
           << width << " " << height << " 255\n";

        // Map buffer if creating a CLTensor/GCTensor
        map(tensor, true);

        switch(tensor.info()->format())
        {
            case arm_compute::Format::U8:
            {
                arm_compute::Window window;
                window.set(arm_compute::Window::DimX, arm_compute::Window::Dimension(0, width, 1));
                window.set(arm_compute::Window::DimY, arm_compute::Window::Dimension(0, height, 1));

                arm_compute::Iterator in(&tensor, window);

                arm_compute::execute_window_loop(window, [&](const arm_compute::Coordinates & id)
                {
                    const unsigned char value = *in.ptr();

                    fs << value << value << value;
                },
                in);

                break;
            }
            case arm_compute::Format::RGB888:
            {
                arm_compute::Window window;
                window.set(arm_compute::Window::DimX, arm_compute::Window::Dimension(0, width, width));
                window.set(arm_compute::Window::DimY, arm_compute::Window::Dimension(0, height, 1));

                arm_compute::Iterator in(&tensor, window);

                arm_compute::execute_window_loop(window, [&](const arm_compute::Coordinates & id)
                {
                    fs.write(reinterpret_cast<std::fstream::char_type *>(in.ptr()), width * tensor.info()->element_size());
                },
                in);

                break;
            }
            default:
                ARM_COMPUTE_ERROR("Unsupported format");
        }

        // Unmap buffer if creating a CLTensor/GCTensor
        unmap(tensor);
    }
    catch(const std::ofstream::failure &e)
    {
        ARM_COMPUTE_ERROR("Writing %s: (%s)", ppm_filename.c_str(), e.what());
    }
}

/** Template helper function to save a tensor image to a NPY file.
 *
 * @note Only F32 data type supported.
 * @note If the input tensor is a CLTensor, the function maps and unmaps the image
 *
 * @param[in] tensor        The tensor to save as NPY file
 * @param[in] npy_filename  Filename of the file to create.
 * @param[in] fortran_order If true, save matrix in fortran order.
 */
template <typename T>
void save_to_npy(T &tensor, const std::string &npy_filename, bool fortran_order)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_NOT_IN(&tensor, arm_compute::DataType::F32);

    std::ofstream fs;
    try
    {
        fs.exceptions(std::ofstream::failbit | std::ofstream::badbit | std::ofstream::eofbit);
        fs.open(npy_filename, std::ios::out | std::ios::binary);

        std::vector<npy::ndarray_len_t> shape(tensor.info()->num_dimensions());

        for(unsigned int i = 0, j = tensor.info()->num_dimensions() - 1; i < tensor.info()->num_dimensions(); ++i, --j)
        {
            shape[i] = tensor.info()->tensor_shape()[!fortran_order ? j : i];
        }

        // Map buffer if creating a CLTensor
        map(tensor, true);

        switch(tensor.info()->data_type())
        {
            case arm_compute::DataType::F32:
            {
                std::vector<float> tmp; /* Used only to get the typestring */
                npy::Typestring    typestring_o{ tmp };
                std::string        typestring = typestring_o.str();

                std::ofstream stream(npy_filename, std::ofstream::binary);
                npy::write_header(stream, typestring, fortran_order, shape);

                arm_compute::Window window;
                window.use_tensor_dimensions(tensor.info()->tensor_shape());

                arm_compute::Iterator in(&tensor, window);

                arm_compute::execute_window_loop(window, [&](const arm_compute::Coordinates & id)
                {
                    stream.write(reinterpret_cast<const char *>(in.ptr()), sizeof(float));
                },
                in);

                break;
            }
            default:
                ARM_COMPUTE_ERROR("Unsupported format");
        }

        // Unmap buffer if creating a CLTensor
        unmap(tensor);
    }
    catch(const std::ofstream::failure &e)
    {
        ARM_COMPUTE_ERROR("Writing %s: (%s)", npy_filename.c_str(), e.what());
    }
}

/** Load the tensor with pre-trained data from a binary file
 *
 * @param[in] tensor   The tensor to be filled. Data type supported: F32.
 * @param[in] filename Filename of the binary file to load from.
 */
template <typename T>
void load_trained_data(T &tensor, const std::string &filename)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&tensor, 1, DataType::F32);

    std::ifstream fs;

    try
    {
        fs.exceptions(std::ofstream::failbit | std::ofstream::badbit | std::ofstream::eofbit);
        // Open file
        fs.open(filename, std::ios::in | std::ios::binary);

        if(!fs.good())
        {
            throw std::runtime_error("Could not load binary data: " + filename);
        }

        // Map buffer if creating a CLTensor/GCTensor
        map(tensor, true);

        Window window;

        window.set(arm_compute::Window::DimX, arm_compute::Window::Dimension(0, 1, 1));

        for(unsigned int d = 1; d < tensor.info()->num_dimensions(); ++d)
        {
            window.set(d, Window::Dimension(0, tensor.info()->tensor_shape()[d], 1));
        }

        arm_compute::Iterator in(&tensor, window);

        execute_window_loop(window, [&](const Coordinates & id)
        {
            fs.read(reinterpret_cast<std::fstream::char_type *>(in.ptr()), tensor.info()->tensor_shape()[0] * tensor.info()->element_size());
        },
        in);

        // Unmap buffer if creating a CLTensor/GCTensor
        unmap(tensor);
    }
    catch(const std::ofstream::failure &e)
    {
        ARM_COMPUTE_ERROR("Writing %s: (%s)", filename.c_str(), e.what());
    }
}

template <typename T>
void fill_random_tensor(T &tensor, float lower_bound, float upper_bound)
{
    std::random_device rd;
    std::mt19937       gen(rd());

    Window window;
    window.use_tensor_dimensions(tensor.info()->tensor_shape());

    map(tensor, true);

    Iterator it(&tensor, window);

    switch(tensor.info()->data_type())
    {
        case arm_compute::DataType::F16:
        {
            std::uniform_real_distribution<float> dist(lower_bound, upper_bound);

            execute_window_loop(window, [&](const Coordinates & id)
            {
                *reinterpret_cast<half *>(it.ptr()) = (half)dist(gen);
            },
            it);

            break;
        }
        case arm_compute::DataType::F32:
        {
            std::uniform_real_distribution<float> dist(lower_bound, upper_bound);

            execute_window_loop(window, [&](const Coordinates & id)
            {
                *reinterpret_cast<float *>(it.ptr()) = dist(gen);
            },
            it);

            break;
        }
        default:
        {
            ARM_COMPUTE_ERROR("Unsupported format");
        }
    }

    unmap(tensor);
}

template <typename T>
void init_sgemm_output(T &dst, T &src0, T &src1, arm_compute::DataType dt)
{
    dst.allocator()->init(TensorInfo(TensorShape(src1.info()->dimension(0), src0.info()->dimension(1), src0.info()->dimension(2)), 1, dt));
}
/** This function returns the amount of memory free reading from /proc/meminfo
 *
 * @return The free memory in kB
 */
uint64_t get_mem_free_from_meminfo();

/** Compare two tensors
 *
 * @param[in] tensor1   First tensor to be compared.
 * @param[in] tensor2   Second tensor to be compared.
 * @param[in] tolerance Tolerance used for the comparison.
 *
 * @return The number of mismatches
 */
template <typename T>
int compare_tensor(ITensor &tensor1, ITensor &tensor2, T tolerance)
{
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(&tensor1, &tensor2);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_SHAPES(&tensor1, &tensor2);

    int    num_mismatches = 0;
    Window window;
    window.use_tensor_dimensions(tensor1.info()->tensor_shape());

    map(tensor1, true);
    map(tensor2, true);

    Iterator itensor1(&tensor1, window);
    Iterator itensor2(&tensor2, window);

    execute_window_loop(window, [&](const Coordinates & id)
    {
        if(std::abs(*reinterpret_cast<T *>(itensor1.ptr()) - *reinterpret_cast<T *>(itensor2.ptr())) > tolerance)
        {
            ++num_mismatches;
        }
    },
    itensor1, itensor2);

    unmap(itensor1);
    unmap(itensor2);

    return num_mismatches;
}
} // namespace utils
} // namespace arm_compute
#endif /* __UTILS_UTILS_H__*/
