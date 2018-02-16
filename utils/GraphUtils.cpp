/*
 * Copyright (c) 2017-2018 ARM Limited.
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
#include "arm_compute/runtime/SubTensor.h"

#ifdef ARM_COMPUTE_CL
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#endif /* ARM_COMPUTE_CL */

#include <iomanip>

using namespace arm_compute::graph_utils;

PPMWriter::PPMWriter(std::string name, unsigned int maximum)
    : _name(std::move(name)), _iterator(0), _maximum(maximum)
{
}

bool PPMWriter::access_tensor(ITensor &tensor)
{
    std::stringstream ss;
    ss << _name << _iterator << ".ppm";

    arm_compute::utils::save_to_ppm(tensor, ss.str());

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

PPMAccessor::PPMAccessor(std::string ppm_path, bool bgr,
                         float mean_r, float mean_g, float mean_b,
                         float std_r, float std_g, float std_b)
    : _ppm_path(std::move(ppm_path)), _bgr(bgr), _mean_r(mean_r), _mean_g(mean_g), _mean_b(mean_b), _std_r(std_r), _std_g(std_g), _std_b(std_b)
{
}

bool PPMAccessor::access_tensor(ITensor &tensor)
{
    utils::PPMLoader ppm;
    const float      mean[3] =
    {
        _bgr ? _mean_b : _mean_r,
        _mean_g,
        _bgr ? _mean_r : _mean_b
    };
    const float std[3] =
    {
        _bgr ? _std_b : _std_r,
        _std_g,
        _bgr ? _std_r : _std_b
    };

    // Open PPM file
    ppm.open(_ppm_path);

    ARM_COMPUTE_ERROR_ON_MSG(ppm.width() != tensor.info()->dimension(0) || ppm.height() != tensor.info()->dimension(1),
                             "Failed to load image file: dimensions [%d,%d] not correct, expected [%d,%d].", ppm.width(), ppm.height(), tensor.info()->dimension(0), tensor.info()->dimension(1));

    // Fill the tensor with the PPM content (BGR)
    ppm.fill_planar_tensor(tensor, _bgr);

    // Subtract the mean value from each channel
    Window window;
    window.use_tensor_dimensions(tensor.info()->tensor_shape());

    execute_window_loop(window, [&](const Coordinates & id)
    {
        const float value                                     = *reinterpret_cast<float *>(tensor.ptr_to_element(id)) - mean[id.z()];
        *reinterpret_cast<float *>(tensor.ptr_to_element(id)) = value / std[id.z()];
    });

    return true;
}

TopNPredictionsAccessor::TopNPredictionsAccessor(const std::string &labels_path, size_t top_n, std::ostream &output_stream)
    : _labels(), _output_stream(output_stream), _top_n(top_n)
{
    _labels.clear();

    std::ifstream ifs;

    try
    {
        ifs.exceptions(std::ifstream::badbit);
        ifs.open(labels_path, std::ios::in | std::ios::binary);

        for(std::string line; !std::getline(ifs, line).fail();)
        {
            _labels.emplace_back(line);
        }
    }
    catch(const std::ifstream::failure &e)
    {
        ARM_COMPUTE_ERROR("Accessing %s: %s", labels_path.c_str(), e.what());
    }
}

template <typename T>
void TopNPredictionsAccessor::access_predictions_tensor(ITensor &tensor)
{
    // Get the predicted class
    std::vector<T>      classes_prob;
    std::vector<size_t> index;

    const auto   output_net  = reinterpret_cast<T *>(tensor.buffer() + tensor.info()->offset_first_element_in_bytes());
    const size_t num_classes = tensor.info()->dimension(0);

    classes_prob.resize(num_classes);
    index.resize(num_classes);

    std::copy(output_net, output_net + num_classes, classes_prob.begin());

    // Sort results
    std::iota(std::begin(index), std::end(index), static_cast<size_t>(0));
    std::sort(std::begin(index), std::end(index),
              [&](size_t a, size_t b)
    {
        return classes_prob[a] > classes_prob[b];
    });

    _output_stream << "---------- Top " << _top_n << " predictions ----------" << std::endl
                   << std::endl;
    for(size_t i = 0; i < _top_n; ++i)
    {
        _output_stream << std::fixed << std::setprecision(4)
                       << +classes_prob[index.at(i)]
                       << " - [id = " << index.at(i) << "]"
                       << ", " << _labels[index.at(i)] << std::endl;
    }
}

bool TopNPredictionsAccessor::access_tensor(ITensor &tensor)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&tensor, 1, DataType::F32, DataType::QASYMM8);
    ARM_COMPUTE_ERROR_ON(_labels.size() != tensor.info()->dimension(0));

    switch(tensor.info()->data_type())
    {
        case DataType::QASYMM8:
            access_predictions_tensor<uint8_t>(tensor);
            break;
        case DataType::F32:
            access_predictions_tensor<float>(tensor);
            break;
        default:
            ARM_COMPUTE_ERROR("NOT SUPPORTED!");
    }

    return false;
}

RandomAccessor::RandomAccessor(PixelValue lower, PixelValue upper, std::random_device::result_type seed)
    : _lower(lower), _upper(upper), _seed(seed)
{
}

template <typename T, typename D>
void RandomAccessor::fill(ITensor &tensor, D &&distribution)
{
    std::mt19937 gen(_seed);

    if(tensor.info()->padding().empty() && !dynamic_cast<SubTensor*>(&tensor))
    {
        for(size_t offset = 0; offset < tensor.info()->total_size(); offset += tensor.info()->element_size())
        {
            const T value                                    = distribution(gen);
            *reinterpret_cast<T *>(tensor.buffer() + offset) = value;
        }
    }
    else
    {
        // If tensor has padding accessing tensor elements through execution window.
        Window window;
        window.use_tensor_dimensions(tensor.info()->tensor_shape());

        execute_window_loop(window, [&](const Coordinates & id)
        {
            const T value                                     = distribution(gen);
            *reinterpret_cast<T *>(tensor.ptr_to_element(id)) = value;
        });
    }
}

bool RandomAccessor::access_tensor(ITensor &tensor)
{
    switch(tensor.info()->data_type())
    {
        case DataType::U8:
        {
            std::uniform_int_distribution<uint8_t> distribution_u8(_lower.get<uint8_t>(), _upper.get<uint8_t>());
            fill<uint8_t>(tensor, distribution_u8);
            break;
        }
        case DataType::S8:
        case DataType::QS8:
        {
            std::uniform_int_distribution<int8_t> distribution_s8(_lower.get<int8_t>(), _upper.get<int8_t>());
            fill<int8_t>(tensor, distribution_s8);
            break;
        }
        case DataType::U16:
        {
            std::uniform_int_distribution<uint16_t> distribution_u16(_lower.get<uint16_t>(), _upper.get<uint16_t>());
            fill<uint16_t>(tensor, distribution_u16);
            break;
        }
        case DataType::S16:
        case DataType::QS16:
        {
            std::uniform_int_distribution<int16_t> distribution_s16(_lower.get<int16_t>(), _upper.get<int16_t>());
            fill<int16_t>(tensor, distribution_s16);
            break;
        }
        case DataType::U32:
        {
            std::uniform_int_distribution<uint32_t> distribution_u32(_lower.get<uint32_t>(), _upper.get<uint32_t>());
            fill<uint32_t>(tensor, distribution_u32);
            break;
        }
        case DataType::S32:
        {
            std::uniform_int_distribution<int32_t> distribution_s32(_lower.get<int32_t>(), _upper.get<int32_t>());
            fill<int32_t>(tensor, distribution_s32);
            break;
        }
        case DataType::U64:
        {
            std::uniform_int_distribution<uint64_t> distribution_u64(_lower.get<uint64_t>(), _upper.get<uint64_t>());
            fill<uint64_t>(tensor, distribution_u64);
            break;
        }
        case DataType::S64:
        {
            std::uniform_int_distribution<int64_t> distribution_s64(_lower.get<int64_t>(), _upper.get<int64_t>());
            fill<int64_t>(tensor, distribution_s64);
            break;
        }
        case DataType::F16:
        {
            std::uniform_real_distribution<float> distribution_f16(_lower.get<float>(), _upper.get<float>());
            fill<float>(tensor, distribution_f16);
            break;
        }
        case DataType::F32:
        {
            std::uniform_real_distribution<float> distribution_f32(_lower.get<float>(), _upper.get<float>());
            fill<float>(tensor, distribution_f32);
            break;
        }
        case DataType::F64:
        {
            std::uniform_real_distribution<double> distribution_f64(_lower.get<double>(), _upper.get<double>());
            fill<double>(tensor, distribution_f64);
            break;
        }
        default:
            ARM_COMPUTE_ERROR("NOT SUPPORTED!");
    }
    return true;
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
    std::string header = npy::read_header(stream);

    // Parse header
    bool        fortran_order = false;
    std::string typestr;
    npy::parse_header(header, typestr, fortran_order, shape);

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
    if(tensor.info()->padding().empty() && !dynamic_cast<SubTensor*>(&tensor))
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
