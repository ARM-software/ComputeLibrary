/*
 * Copyright (c) 2017 ARM Limited.
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

#include "utils/GraphUtils.h"
#include "utils/Utils.h"

#ifdef ARM_COMPUTE_CL
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#endif /* ARM_COMPUTE_CL */

#include "arm_compute/core/Error.h"
#include "libnpy/npy.hpp"

#include <sstream>

using namespace arm_compute::graph_utils;

PPMWriter::PPMWriter(std::string name, unsigned int maximum)
    : _name(std::move(name)), _iterator(0), _maximum(maximum)
{
}

bool PPMWriter::access_tensor(ITensor &tensor)
{
    std::stringstream ss;
    ss << _name << _iterator << ".ppm";
    if(dynamic_cast<Tensor *>(&tensor) != nullptr)
    {
        arm_compute::utils::save_to_ppm(dynamic_cast<Tensor &>(tensor), ss.str());
    }
#ifdef ARM_COMPUTE_CL
    else if(dynamic_cast<CLTensor *>(&tensor) != nullptr)
    {
        arm_compute::utils::save_to_ppm(dynamic_cast<CLTensor &>(tensor), ss.str());
    }
#endif /* ARM_COMPUTE_CL */

    _iterator++;
    if(_maximum == 0)
    {
        return true;
    }
    return _iterator < _maximum;
}

DummyAccessor::DummyAccessor(unsigned int maximum)
    : _iterator(0), _maximum(maximum)
{
}

bool DummyAccessor::access_tensor(ITensor &tensor)
{
    ARM_COMPUTE_UNUSED(tensor);
    bool ret = _maximum == 0 || _iterator < _maximum;
    if(_iterator == _maximum)
    {
        _iterator = 0;
    }
    else
    {
        _iterator++;
    }
    return ret;
}

NumPyBinLoader::NumPyBinLoader(std::string filename)
    : _filename(std::move(filename))
{
}

bool NumPyBinLoader::access_tensor(ITensor &tensor)
{
    const TensorShape          tensor_shape = tensor.info()->tensor_shape();
    std::vector<unsigned long> shape;

    // Open file
    std::ifstream stream(_filename, std::ios::in | std::ios::binary);
    ARM_COMPUTE_ERROR_ON_MSG(!stream.good(), "Failed to load binary data");
    // Check magic bytes and version number
    unsigned char v_major = 0;
    unsigned char v_minor = 0;
    npy::read_magic(stream, &v_major, &v_minor);

    // Read header
    std::string header;
    if(v_major == 1 && v_minor == 0)
    {
        header = npy::read_header_1_0(stream);
    }
    else if(v_major == 2 && v_minor == 0)
    {
        header = npy::read_header_2_0(stream);
    }
    else
    {
        ARM_COMPUTE_ERROR("Unsupported file format version");
    }

    // Parse header
    bool        fortran_order = false;
    std::string typestr;
    npy::ParseHeader(header, typestr, &fortran_order, shape);

    // Check if the typestring matches the given one
    std::string expect_typestr = arm_compute::utils::get_typestring(tensor.info()->data_type());
    ARM_COMPUTE_ERROR_ON_MSG(typestr != expect_typestr, "Typestrings mismatch");

    // Validate tensor shape
    ARM_COMPUTE_ERROR_ON_MSG(shape.size() != tensor_shape.num_dimensions(), "Tensor ranks mismatch");
    if(fortran_order)
    {
        for(size_t i = 0; i < shape.size(); ++i)
        {
            ARM_COMPUTE_ERROR_ON_MSG(tensor_shape[i] != shape[i], "Tensor dimensions mismatch");
        }
    }
    else
    {
        for(size_t i = 0; i < shape.size(); ++i)
        {
            ARM_COMPUTE_ERROR_ON_MSG(tensor_shape[i] != shape[shape.size() - i - 1], "Tensor dimensions mismatch");
        }
    }

    // Read data
    if(tensor.info()->padding().empty())
    {
        // If tensor has no padding read directly from stream.
        stream.read(reinterpret_cast<char *>(tensor.buffer()), tensor.info()->total_size());
    }
    else
    {
        // If tensor has padding accessing tensor elements through execution window.
        Window window;
        window.use_tensor_dimensions(tensor_shape);

        execute_window_loop(window, [&](const Coordinates & id)
        {
            stream.read(reinterpret_cast<char *>(tensor.ptr_to_element(id)), tensor.info()->element_size());
        });
    }
    return true;
}
