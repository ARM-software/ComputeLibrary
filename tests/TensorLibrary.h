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
#ifndef __ARM_COMPUTE_TEST_TENSOR_LIBRARY_H__
#define __ARM_COMPUTE_TEST_TENSOR_LIBRARY_H__

#include "RawTensor.h"
#include "TensorCache.h"
#include "Utils.h"

#include "arm_compute/core/Coordinates.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Window.h"

#include <algorithm>
#include <cstddef>
#include <fstream>
#include <random>
#include <string>
#include <type_traits>

#if ARM_COMPUTE_ENABLE_FP16
#include <arm_fp16.h> // needed for float16_t
#endif

namespace arm_compute
{
namespace test
{
/** Factory class to create and fill tensors.
 *
 * Allows to initialise tensors from loaded images or by specifying the shape
 * explicitly. Furthermore, provides methods to fill tensors with the content of
 * loaded images or with random values.
 */
class TensorLibrary final
{
public:
    /** Initialises the library with a @p path to the image directory.
     *
     * @param[in] path Path to load images from.
     */
    TensorLibrary(std::string path);

    /** Initialises the library with a @p path to the image directory.
     * Furthermore, sets the seed for the random generator to @p seed.
     *
     * @param[in] path Path to load images from.
     * @param[in] seed Seed used to initialise the random number generator.
     */
    TensorLibrary(std::string path, std::random_device::result_type seed);

    /** Seed that is used to fill tensors with random values. */
    std::random_device::result_type seed() const;

    /** Provides a tensor shape for the specified image.
     *
     * @param[in] name Image file used to look up the raw tensor.
     */
    TensorShape get_image_shape(const std::string &name);

    /** Creates an uninitialised raw tensor with the given @p shape, @p
     * data_type and @p num_channels.
     *
     * @param[in] shape                Shape used to initialise the tensor.
     * @param[in] data_type            Data type used to initialise the tensor.
     * @param[in] num_channels         (Optional) Number of channels used to initialise the tensor.
     * @param[in] fixed_point_position (Optional) Number of bits for the fractional part of the fixed point numbers
     */
    static RawTensor get(const TensorShape &shape, DataType data_type, int num_channels = 1, int fixed_point_position = 0);

    /** Creates an uninitialised raw tensor with the given @p shape and @p format.
     *
     * @param[in] shape  Shape used to initialise the tensor.
     * @param[in] format Format used to initialise the tensor.
     */
    static RawTensor get(const TensorShape &shape, Format format);

    /** Provides a contant raw tensor for the specified image.
     *
     * @param[in] name Image file used to look up the raw tensor.
     */
    const RawTensor &get(const std::string &name) const;

    /** Provides a raw tensor for the specified image.
     *
     * @param[in] name Image file used to look up the raw tensor.
     */
    RawTensor get(const std::string &name);

    /** Creates an uninitialised raw tensor with the given @p data_type and @p
     * num_channels. The shape is derived from the specified image.
     *
     * @param[in] name         Image file used to initialise the tensor.
     * @param[in] data_type    Data type used to initialise the tensor.
     * @param[in] num_channels Number of channels used to initialise the tensor.
     */
    RawTensor get(const std::string &name, DataType data_type, int num_channels = 1) const;

    /** Provides a contant raw tensor for the specified image after it has been
     * converted to @p format.
     *
     * @param[in] name   Image file used to look up the raw tensor.
     * @param[in] format Format used to look up the raw tensor.
     */
    const RawTensor &get(const std::string &name, Format format) const;

    /** Provides a raw tensor for the specified image after it has been
     * converted to @p format.
     *
     * @param[in] name   Image file used to look up the raw tensor.
     * @param[in] format Format used to look up the raw tensor.
     */
    RawTensor get(const std::string &name, Format format);

    /** Provides a contant raw tensor for the specified channel after it has
     * been extracted form the given image.
     *
     * @param[in] name    Image file used to look up the raw tensor.
     * @param[in] channel Channel used to look up the raw tensor.
     *
     * @note The channel has to be unambiguous so that the format can be
     *       inferred automatically.
     */
    const RawTensor &get(const std::string &name, Channel channel) const;

    /** Provides a raw tensor for the specified channel after it has been
     * extracted form the given image.
     *
     * @param[in] name    Image file used to look up the raw tensor.
     * @param[in] channel Channel used to look up the raw tensor.
     *
     * @note The channel has to be unambiguous so that the format can be
     *       inferred automatically.
     */
    RawTensor get(const std::string &name, Channel channel);

    /** Provides a constant raw tensor for the specified channel after it has
     * been extracted form the given image formatted to @p format.
     *
     * @param[in] name    Image file used to look up the raw tensor.
     * @param[in] format  Format used to look up the raw tensor.
     * @param[in] channel Channel used to look up the raw tensor.
     */
    const RawTensor &get(const std::string &name, Format format, Channel channel) const;

    /** Provides a raw tensor for the specified channel after it has been
     * extracted form the given image formatted to @p format.
     *
     * @param[in] name    Image file used to look up the raw tensor.
     * @param[in] format  Format used to look up the raw tensor.
     * @param[in] channel Channel used to look up the raw tensor.
     */
    RawTensor get(const std::string &name, Format format, Channel channel);

    /** Fills the specified @p tensor with random values drawn from @p
     * distribution.
     *
     * @param[in, out] tensor       To be filled tensor.
     * @param[in]      distribution Distribution used to fill the tensor.
     * @param[in]      seed_offset  The offset will be added to the global seed before initialising the random generator.
     *
     * @note The @p distribution has to provide operator(Generator &) which
     *       will be used to draw samples.
     */
    template <typename T, typename D>
    void fill(T &&tensor, D &&distribution, std::random_device::result_type seed_offset) const;

    /** Fills the specified @p raw tensor with random values drawn from @p
     * distribution.
     *
     * @param[in, out] raw          To be filled raw.
     * @param[in]      distribution Distribution used to fill the tensor.
     * @param[in]      seed_offset  The offset will be added to the global seed before initialising the random generator.
     *
     * @note The @p distribution has to provide operator(Generator &) which
     *       will be used to draw samples.
     */
    template <typename D>
    void fill(RawTensor &raw, D &&distribution, std::random_device::result_type seed_offset) const;

    /** Fills the specified @p tensor with the content of the specified image
     * converted to the given format.
     *
     * @param[in, out] tensor To be filled tensor.
     * @param[in]      name   Image file used to fill the tensor.
     * @param[in]      format Format of the image used to fill the tensor.
     *
     * @warning No check is performed that the specified format actually
     *          matches the format of the tensor.
     */
    template <typename T>
    void fill(T &&tensor, const std::string &name, Format format) const;

    /** Fills the raw tensor with the content of the specified image
     * converted to the given format.
     *
     * @param[in, out] raw    To be filled raw tensor.
     * @param[in]      name   Image file used to fill the tensor.
     * @param[in]      format Format of the image used to fill the tensor.
     *
     * @warning No check is performed that the specified format actually
     *          matches the format of the tensor.
     */
    void fill(RawTensor &raw, const std::string &name, Format format) const;

    /** Fills the specified @p tensor with the content of the specified channel
     * extracted from the given image.
     *
     * @param[in, out] tensor  To be filled tensor.
     * @param[in]      name    Image file used to fill the tensor.
     * @param[in]      channel Channel of the image used to fill the tensor.
     *
     * @note The channel has to be unambiguous so that the format can be
     *       inferred automatically.
     *
     * @warning No check is performed that the specified format actually
     *          matches the format of the tensor.
     */
    template <typename T>
    void fill(T &&tensor, const std::string &name, Channel channel) const;

    /** Fills the raw tensor with the content of the specified channel
     * extracted from the given image.
     *
     * @param[in, out] raw     To be filled raw tensor.
     * @param[in]      name    Image file used to fill the tensor.
     * @param[in]      channel Channel of the image used to fill the tensor.
     *
     * @note The channel has to be unambiguous so that the format can be
     *       inferred automatically.
     *
     * @warning No check is performed that the specified format actually
     *          matches the format of the tensor.
     */
    void fill(RawTensor &raw, const std::string &name, Channel channel) const;

    /** Fills the specified @p tensor with the content of the specified channel
     * extracted from the given image after it has been converted to the given
     * format.
     *
     * @param[in, out] tensor  To be filled tensor.
     * @param[in]      name    Image file used to fill the tensor.
     * @param[in]      format  Format of the image used to fill the tensor.
     * @param[in]      channel Channel of the image used to fill the tensor.
     *
     * @warning No check is performed that the specified format actually
     *          matches the format of the tensor.
     */
    template <typename T>
    void fill(T &&tensor, const std::string &name, Format format, Channel channel) const;

    /** Fills the raw tensor with the content of the specified channel
     * extracted from the given image after it has been converted to the given
     * format.
     *
     * @param[in, out] raw     To be filled raw tensor.
     * @param[in]      name    Image file used to fill the tensor.
     * @param[in]      format  Format of the image used to fill the tensor.
     * @param[in]      channel Channel of the image used to fill the tensor.
     *
     * @warning No check is performed that the specified format actually
     *          matches the format of the tensor.
     */
    void fill(RawTensor &raw, const std::string &name, Format format, Channel channel) const;

    /** Fill a tensor with uniform distribution across the range of its type
     *
     * @param[in, out] tensor      To be filled tensor.
     * @param[in]      seed_offset The offset will be added to the global seed before initialising the random generator.
     */
    template <typename T>
    void fill_tensor_uniform(T &&tensor, std::random_device::result_type seed_offset) const;

    /** Fill a tensor with uniform distribution across the a specified range
     *
     * @param[in, out] tensor      To be filled tensor.
     * @param[in]      seed_offset The offset will be added to the global seed before initialising the random generator.
     * @param[in]      low         lowest value in the range (inclusive)
     * @param[in]      high        highest value in the range (inclusive)
     *
     * @note    @p low and @p high must be of the same type as the data type of @p tensor
     */
    template <typename T, typename D>
    void fill_tensor_uniform(T &&tensor, std::random_device::result_type seed_offset, D low, D high) const;

    /** Fills the specified @p tensor with data loaded from binary in specified path.
     *
     * @param[in, out] tensor To be filled tensor.
     * @param[in]      name   Data file.
     */
    template <typename T>
    void fill_layer_data(T &&tensor, std::string name) const;

private:
    // Function prototype to convert between image formats.
    using Converter = void (*)(const RawTensor &src, RawTensor &dst);
    // Function prototype to extract a channel from an image.
    using Extractor = void (*)(const RawTensor &src, RawTensor &dst);
    // Function prototype to load an image file.
    using Loader = RawTensor (*)(const std::string &path);

    const Converter &get_converter(Format src, Format dst) const;
    const Converter &get_converter(DataType src, Format dst) const;
    const Converter &get_converter(Format src, DataType dst) const;
    const Converter &get_converter(DataType src, DataType dst) const;
    const Extractor &get_extractor(Format format, Channel) const;
    const Loader &get_loader(const std::string &extension) const;

    /** Creates a raw tensor from the specified image.
     *
     * @param[in] name To be loaded image file.
     *
     * @note If use_single_image is true @p name is ignored and the user image
     *       is loaded instead.
     */
    RawTensor load_image(const std::string &name) const;

    /** Provides a raw tensor for the specified image and format.
     *
     * @param[in] name   Image file used to look up the raw tensor.
     * @param[in] format Format used to look up the raw tensor.
     *
     * If the tensor has already been requested before the cached version will
     * be returned. Otherwise the tensor will be added to the cache.
     *
     * @note If use_single_image is true @p name is ignored and the user image
     *       is loaded instead.
     */
    const RawTensor &find_or_create_raw_tensor(const std::string &name, Format format) const;

    /** Provides a raw tensor for the specified image, format and channel.
     *
     * @param[in] name    Image file used to look up the raw tensor.
     * @param[in] format  Format used to look up the raw tensor.
     * @param[in] channel Channel used to look up the raw tensor.
     *
     * If the tensor has already been requested before the cached version will
     * be returned. Otherwise the tensor will be added to the cache.
     *
     * @note If use_single_image is true @p name is ignored and the user image
     *       is loaded instead.
     */
    const RawTensor &find_or_create_raw_tensor(const std::string &name, Format format, Channel channel) const;

    mutable TensorCache             _cache{};
    mutable std::mutex              _format_lock{};
    mutable std::mutex              _channel_lock{};
    std::string                     _library_path;
    std::random_device::result_type _seed;
};

template <typename T, typename D>
void TensorLibrary::fill(T &&tensor, D &&distribution, std::random_device::result_type seed_offset) const
{
    Window window;
    for(unsigned int d = 0; d < tensor.shape().num_dimensions(); ++d)
    {
        window.set(d, Window::Dimension(0, tensor.shape()[d], 1));
    }

    std::mt19937 gen(_seed + seed_offset);

    //FIXME: Replace with normal loop
    execute_window_loop(window, [&](const Coordinates & id)
    {
        using ResultType         = typename std::remove_reference<D>::type::result_type;
        const ResultType value   = distribution(gen);
        void *const      out_ptr = tensor(id);
        store_value_with_data_type(out_ptr, value, tensor.data_type());
    });
}

template <typename D>
void TensorLibrary::fill(RawTensor &raw, D &&distribution, std::random_device::result_type seed_offset) const
{
    std::mt19937 gen(_seed + seed_offset);

    for(size_t offset = 0; offset < raw.size(); offset += raw.element_size())
    {
        using ResultType       = typename std::remove_reference<D>::type::result_type;
        const ResultType value = distribution(gen);
        store_value_with_data_type(raw.data() + offset, value, raw.data_type());
    }
}

template <typename T>
void TensorLibrary::fill(T &&tensor, const std::string &name, Format format) const
{
    const RawTensor &raw = get(name, format);

    for(size_t offset = 0; offset < raw.size(); offset += raw.element_size())
    {
        const Coordinates id = index2coord(raw.shape(), offset / raw.element_size());

        const RawTensor::BufferType *const raw_ptr = raw.data() + offset;
        const auto                         out_ptr = static_cast<RawTensor::BufferType *>(tensor(id));
        std::copy_n(raw_ptr, raw.element_size(), out_ptr);
    }
}

template <typename T>
void TensorLibrary::fill(T &&tensor, const std::string &name, Channel channel) const
{
    fill(std::forward<T>(tensor), name, get_format_for_channel(channel), channel);
}

template <typename T>
void TensorLibrary::fill(T &&tensor, const std::string &name, Format format, Channel channel) const
{
    const RawTensor &raw = get(name, format, channel);

    for(size_t offset = 0; offset < raw.size(); offset += raw.element_size())
    {
        const Coordinates id = index2coord(raw.shape(), offset / raw.element_size());

        const RawTensor::BufferType *const raw_ptr = raw.data() + offset;
        const auto                         out_ptr = static_cast<RawTensor::BufferType *>(tensor(id));
        std::copy_n(raw_ptr, raw.element_size(), out_ptr);
    }
}

template <typename T>
void TensorLibrary::fill_tensor_uniform(T &&tensor, std::random_device::result_type seed_offset) const
{
    switch(tensor.data_type())
    {
        case DataType::U8:
        {
            std::uniform_int_distribution<uint8_t> distribution_u8(std::numeric_limits<uint8_t>::lowest(), std::numeric_limits<uint8_t>::max());
            fill(tensor, distribution_u8, seed_offset);
            break;
        }
        case DataType::S8:
        case DataType::QS8:
        {
            std::uniform_int_distribution<int8_t> distribution_s8(std::numeric_limits<int8_t>::lowest(), std::numeric_limits<int8_t>::max());
            fill(tensor, distribution_s8, seed_offset);
            break;
        }
        case DataType::U16:
        {
            std::uniform_int_distribution<uint16_t> distribution_u16(std::numeric_limits<uint16_t>::lowest(), std::numeric_limits<uint16_t>::max());
            fill(tensor, distribution_u16, seed_offset);
            break;
        }
        case DataType::S16:
        {
            std::uniform_int_distribution<int16_t> distribution_s16(std::numeric_limits<int16_t>::lowest(), std::numeric_limits<int16_t>::max());
            fill(tensor, distribution_s16, seed_offset);
            break;
        }
        case DataType::U32:
        {
            std::uniform_int_distribution<uint32_t> distribution_u32(std::numeric_limits<uint32_t>::lowest(), std::numeric_limits<uint32_t>::max());
            fill(tensor, distribution_u32, seed_offset);
            break;
        }
        case DataType::S32:
        {
            std::uniform_int_distribution<int32_t> distribution_s32(std::numeric_limits<int32_t>::lowest(), std::numeric_limits<int32_t>::max());
            fill(tensor, distribution_s32, seed_offset);
            break;
        }
        case DataType::U64:
        {
            std::uniform_int_distribution<uint64_t> distribution_u64(std::numeric_limits<uint64_t>::lowest(), std::numeric_limits<uint64_t>::max());
            fill(tensor, distribution_u64, seed_offset);
            break;
        }
        case DataType::S64:
        {
            std::uniform_int_distribution<int64_t> distribution_s64(std::numeric_limits<int64_t>::lowest(), std::numeric_limits<int64_t>::max());
            fill(tensor, distribution_s64, seed_offset);
            break;
        }
#if ARM_COMPUTE_ENABLE_FP16
        case DataType::F16:
        {
            std::uniform_real_distribution<float> distribution_f16(-1000.f, 1000.f);
            fill(tensor, distribution_f16, seed_offset);
            break;
        }
#endif /* ARM_COMPUTE_ENABLE_FP16 */
        case DataType::F32:
        {
            // It doesn't make sense to check [-inf, inf], so hard code it to a big number
            std::uniform_real_distribution<float> distribution_f32(-1000.f, 1000.f);
            fill(tensor, distribution_f32, seed_offset);
            break;
        }
        case DataType::F64:
        {
            // It doesn't make sense to check [-inf, inf], so hard code it to a big number
            std::uniform_real_distribution<double> distribution_f64(-1000.f, 1000.f);
            fill(tensor, distribution_f64, seed_offset);
            break;
        }
        case DataType::SIZET:
        {
            std::uniform_int_distribution<size_t> distribution_sizet(std::numeric_limits<size_t>::lowest(), std::numeric_limits<size_t>::max());
            fill(tensor, distribution_sizet, seed_offset);
            break;
        }
        default:
            ARM_COMPUTE_ERROR("NOT SUPPORTED!");
    }
}

template <typename T, typename D>
void TensorLibrary::fill_tensor_uniform(T &&tensor, std::random_device::result_type seed_offset, D low, D high) const
{
    switch(tensor.data_type())
    {
        case DataType::U8:
        {
            ARM_COMPUTE_ERROR_ON(!(std::is_same<uint8_t, D>::value));
            std::uniform_int_distribution<uint8_t> distribution_u8(low, high);
            fill(tensor, distribution_u8, seed_offset);
            break;
        }
        case DataType::S8:
        case DataType::QS8:
        {
            ARM_COMPUTE_ERROR_ON(!(std::is_same<int8_t, D>::value));
            std::uniform_int_distribution<int8_t> distribution_s8(low, high);
            fill(tensor, distribution_s8, seed_offset);
            break;
        }
        case DataType::U16:
        {
            ARM_COMPUTE_ERROR_ON(!(std::is_same<uint16_t, D>::value));
            std::uniform_int_distribution<uint16_t> distribution_u16(low, high);
            fill(tensor, distribution_u16, seed_offset);
            break;
        }
        case DataType::S16:
        {
            ARM_COMPUTE_ERROR_ON(!(std::is_same<int16_t, D>::value));
            std::uniform_int_distribution<int16_t> distribution_s16(low, high);
            fill(tensor, distribution_s16, seed_offset);
            break;
        }
        case DataType::U32:
        {
            ARM_COMPUTE_ERROR_ON(!(std::is_same<uint32_t, D>::value));
            std::uniform_int_distribution<uint32_t> distribution_u32(low, high);
            fill(tensor, distribution_u32, seed_offset);
            break;
        }
        case DataType::S32:
        {
            ARM_COMPUTE_ERROR_ON(!(std::is_same<int32_t, D>::value));
            std::uniform_int_distribution<int32_t> distribution_s32(low, high);
            fill(tensor, distribution_s32, seed_offset);
            break;
        }
        case DataType::U64:
        {
            ARM_COMPUTE_ERROR_ON(!(std::is_same<uint64_t, D>::value));
            std::uniform_int_distribution<uint64_t> distribution_u64(low, high);
            fill(tensor, distribution_u64, seed_offset);
            break;
        }
        case DataType::S64:
        {
            ARM_COMPUTE_ERROR_ON(!(std::is_same<int64_t, D>::value));
            std::uniform_int_distribution<int64_t> distribution_s64(low, high);
            fill(tensor, distribution_s64, seed_offset);
            break;
        }
#if ARM_COMPUTE_ENABLE_FP16
        case DataType::F16:
        {
            std::uniform_real_distribution<float_t> distribution_f16(low, high);
            fill(tensor, distribution_f16, seed_offset);
            break;
        }
#endif
        case DataType::F32:
        {
            ARM_COMPUTE_ERROR_ON(!(std::is_same<float, D>::value));
            std::uniform_real_distribution<float> distribution_f32(low, high);
            fill(tensor, distribution_f32, seed_offset);
            break;
        }
        case DataType::F64:
        {
            ARM_COMPUTE_ERROR_ON(!(std::is_same<double, D>::value));
            std::uniform_real_distribution<double> distribution_f64(low, high);
            fill(tensor, distribution_f64, seed_offset);
            break;
        }
        case DataType::SIZET:
        {
            ARM_COMPUTE_ERROR_ON(!(std::is_same<size_t, D>::value));
            std::uniform_int_distribution<size_t> distribution_sizet(low, high);
            fill(tensor, distribution_sizet, seed_offset);
            break;
        }
        default:
            ARM_COMPUTE_ERROR("NOT SUPPORTED!");
    }
}

template <typename T>
void TensorLibrary::fill_layer_data(T &&tensor, std::string name) const
{
#ifdef _WIN32
    const std::string path_separator("\\");
#else
    const std::string path_separator("/");
#endif

    const std::string path = _library_path + path_separator + name;

    // Open file
    std::ifstream file(path, std::ios::in | std::ios::binary);
    if(!file.good())
    {
        throw std::runtime_error("Could not load binary data: " + path);
    }

    Window window;
    for(unsigned int d = 0; d < tensor.shape().num_dimensions(); ++d)
    {
        window.set(d, Window::Dimension(0, tensor.shape()[d], 1));
    }

    //FIXME : Replace with normal loop
    execute_window_loop(window, [&](const Coordinates & id)
    {
        float val;
        file.read(reinterpret_cast<char *>(&val), sizeof(float));
        void *const out_ptr = tensor(id);
        store_value_with_data_type(out_ptr, val, tensor.data_type());
    });
}
} // namespace test
} // namespace arm_compute
#endif
