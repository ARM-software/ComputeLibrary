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

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/SubTensor.h"
#include "utils/Utils.h"

#include <iomanip>

using namespace arm_compute::graph_utils;

namespace
{
std::pair<arm_compute::TensorShape, arm_compute::PermutationVector> compute_permutation_paramaters(const arm_compute::TensorShape &shape,
                                                                                                   arm_compute::DataLayout data_layout)
{
    // Set permutation parameters if needed
    arm_compute::TensorShape       permuted_shape = shape;
    arm_compute::PermutationVector perm;
    // Permute only if num_dimensions greater than 2
    if(shape.num_dimensions() > 2)
    {
        perm = (data_layout == arm_compute::DataLayout::NHWC) ? arm_compute::PermutationVector(2U, 0U, 1U) : arm_compute::PermutationVector(1U, 2U, 0U);

        arm_compute::PermutationVector perm_shape = (data_layout == arm_compute::DataLayout::NCHW) ? arm_compute::PermutationVector(2U, 0U, 1U) : arm_compute::PermutationVector(1U, 2U, 0U);
        arm_compute::permute(permuted_shape, perm_shape);
    }

    return std::make_pair(permuted_shape, perm);
}
} // namespace

void TFPreproccessor::preprocess(ITensor &tensor)
{
    Window window;
    window.use_tensor_dimensions(tensor.info()->tensor_shape());

    execute_window_loop(window, [&](const Coordinates & id)
    {
        const float value                                     = *reinterpret_cast<float *>(tensor.ptr_to_element(id));
        float       res                                       = value / 255.f;      // Normalize to [0, 1]
        res                                                   = (res - 0.5f) * 2.f; // Map to [-1, 1]
        *reinterpret_cast<float *>(tensor.ptr_to_element(id)) = res;
    });
}

CaffePreproccessor::CaffePreproccessor(std::array<float, 3> mean, bool bgr)
    : _mean(mean), _bgr(bgr)
{
    if(_bgr)
    {
        std::swap(_mean[0], _mean[2]);
    }
}

void CaffePreproccessor::preprocess(ITensor &tensor)
{
    Window window;
    window.use_tensor_dimensions(tensor.info()->tensor_shape());

    execute_window_loop(window, [&](const Coordinates & id)
    {
        const float value                                     = *reinterpret_cast<float *>(tensor.ptr_to_element(id)) - _mean[id.z()];
        *reinterpret_cast<float *>(tensor.ptr_to_element(id)) = value;
    });
}

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

NumPyAccessor::NumPyAccessor(std::string npy_path, TensorShape shape, DataType data_type, std::ostream &output_stream)
    : _npy_tensor(), _filename(std::move(npy_path)), _output_stream(output_stream)
{
    NumPyBinLoader loader(_filename);

    TensorInfo info(shape, 1, data_type);
    _npy_tensor.allocator()->init(info);
    _npy_tensor.allocator()->allocate();

    loader.access_tensor(_npy_tensor);
}

template <typename T>
void NumPyAccessor::access_numpy_tensor(ITensor &tensor)
{
    const int num_elements          = tensor.info()->total_size();
    int       num_mismatches        = utils::compare_tensor<T>(tensor, _npy_tensor);
    float     percentage_mismatches = static_cast<float>(num_mismatches) / num_elements;

    _output_stream << "Results: " << 100.f - (percentage_mismatches * 100) << " % matches with the provided output[" << _filename << "]." << std::endl;
}

bool NumPyAccessor::access_tensor(ITensor &tensor)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&tensor, 1, DataType::F32);
    ARM_COMPUTE_ERROR_ON(_npy_tensor.info()->dimension(0) != tensor.info()->dimension(0));

    switch(tensor.info()->data_type())
    {
        case DataType::F32:
            access_numpy_tensor<float>(tensor);
            break;
        default:
            ARM_COMPUTE_ERROR("NOT SUPPORTED!");
    }

    return false;
}

PPMAccessor::PPMAccessor(std::string ppm_path, bool bgr, std::unique_ptr<IPreprocessor> preprocessor)
    : _ppm_path(std::move(ppm_path)), _bgr(bgr), _preprocessor(std::move(preprocessor))
{
}

bool PPMAccessor::access_tensor(ITensor &tensor)
{
    utils::PPMLoader ppm;

    // Open PPM file
    ppm.open(_ppm_path);

    // Get permutated shape and permutation parameters
    TensorShape                    permuted_shape = tensor.info()->tensor_shape();
    arm_compute::PermutationVector perm;
    if(tensor.info()->data_layout() != DataLayout::NCHW)
    {
        std::tie(permuted_shape, perm) = compute_permutation_paramaters(tensor.info()->tensor_shape(), tensor.info()->data_layout());
    }
    ARM_COMPUTE_ERROR_ON_MSG(ppm.width() != permuted_shape.x() || ppm.height() != permuted_shape.y(),
                             "Failed to load image file: dimensions [%d,%d] not correct, expected [%d,%d].", ppm.width(), ppm.height(), permuted_shape.x(), permuted_shape.y());

    // Fill the tensor with the PPM content (BGR)
    ppm.fill_planar_tensor(tensor, _bgr);

    // Preprocess tensor
    if(_preprocessor)
    {
        _preprocessor->preprocess(tensor);
    }

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

    if(tensor.info()->padding().empty() && (dynamic_cast<SubTensor *>(&tensor) == nullptr))
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

NumPyBinLoader::NumPyBinLoader(std::string filename, DataLayout file_layout)
    : _filename(std::move(filename)), _file_layout(file_layout)
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

    // Reverse vector in case of non fortran order
    if(!fortran_order)
    {
        std::reverse(shape.begin(), shape.end());
    }

    // Correct dimensions (Needs to match TensorShape dimension corrections)
    if(shape.size() != tensor_shape.num_dimensions())
    {
        for(int i = static_cast<int>(shape.size()) - 1; i > 0; --i)
        {
            if(shape[i] == 1)
            {
                shape.pop_back();
            }
            else
            {
                break;
            }
        }
    }

    bool are_layouts_different = (_file_layout != tensor.info()->data_layout());

    // Validate tensor ranks
    ARM_COMPUTE_ERROR_ON_MSG(shape.size() != tensor_shape.num_dimensions(), "Tensor ranks mismatch");

    // Set permutation parameters if needed
    TensorShape                    permuted_shape = tensor_shape;
    arm_compute::PermutationVector perm;
    if(are_layouts_different)
    {
        std::tie(permuted_shape, perm) = compute_permutation_paramaters(tensor_shape, tensor.info()->data_layout());
    }

    // Validate shapes
    for(size_t i = 0; i < shape.size(); ++i)
    {
        ARM_COMPUTE_ERROR_ON_MSG(permuted_shape[i] != shape[i], "Tensor dimensions mismatch");
    }

    // Validate shapes and copy tensor
    if(!are_layouts_different || perm.num_dimensions() <= 2)
    {
        // Read data
        if(tensor.info()->padding().empty() && (dynamic_cast<SubTensor *>(&tensor) == nullptr))
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
    }
    else
    {
        // If tensor has padding accessing tensor elements through execution window.
        Window window;
        window.use_tensor_dimensions(permuted_shape);

        execute_window_loop(window, [&](const Coordinates & id)
        {
            Coordinates coords(id);
            arm_compute::permute(coords, perm);
            stream.read(reinterpret_cast<char *>(tensor.ptr_to_element(coords)), tensor.info()->element_size());
        });
    }
    return true;
}
