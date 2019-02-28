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
#ifndef __ARM_COMPUTE_TEST_TENSOR_LIBRARY_H__
#define __ARM_COMPUTE_TEST_TENSOR_LIBRARY_H__

#include "arm_compute/core/Coordinates.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Window.h"
#include "arm_compute/core/utils/misc/Random.h"
#include "libnpy/npy.hpp"
#include "tests/RawTensor.h"
#include "tests/TensorCache.h"
#include "tests/Utils.h"
#include "tests/framework/Exceptions.h"

#include <algorithm>
#include <cstddef>
#include <fstream>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

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
class AssetsLibrary final
{
public:
    using RangePair = std::pair<float, float>;

public:
    /** Initialises the library with a @p path to the assets directory.
     * Furthermore, sets the seed for the random generator to @p seed.
     *
     * @param[in] path Path to load assets from.
     * @param[in] seed Seed used to initialise the random number generator.
     */
    AssetsLibrary(std::string path, std::random_device::result_type seed);

    /** Path to assets directory used to initialise library.
     *
     * @return the path to the assets directory.
     */
    std::string path() const;

    /** Seed that is used to fill tensors with random values.
     *
     * @return the initial random seed.
     */
    std::random_device::result_type seed() const;

    /** Provides a tensor shape for the specified image.
     *
     * @param[in] name Image file used to look up the raw tensor.
     *
     * @return the tensor shape for the specified image.
     */
    TensorShape get_image_shape(const std::string &name);

    /** Provides a constant raw tensor for the specified image.
     *
     * @param[in] name Image file used to look up the raw tensor.
     *
     * @return a raw tensor for the specified image.
     */
    const RawTensor &get(const std::string &name) const;

    /** Provides a raw tensor for the specified image.
     *
     * @param[in] name Image file used to look up the raw tensor.
     *
     * @return a raw tensor for the specified image.
     */
    RawTensor get(const std::string &name);

    /** Creates an uninitialised raw tensor with the given @p data_type and @p
     * num_channels. The shape is derived from the specified image.
     *
     * @param[in] name         Image file used to initialise the tensor.
     * @param[in] data_type    Data type used to initialise the tensor.
     * @param[in] num_channels Number of channels used to initialise the tensor.
     *
     * @return a raw tensor for the specified image.
     */
    RawTensor get(const std::string &name, DataType data_type, int num_channels = 1) const;

    /** Provides a contant raw tensor for the specified image after it has been
     * converted to @p format.
     *
     * @param[in] name   Image file used to look up the raw tensor.
     * @param[in] format Format used to look up the raw tensor.
     *
     * @return a raw tensor for the specified image.
     */
    const RawTensor &get(const std::string &name, Format format) const;

    /** Provides a raw tensor for the specified image after it has been
     * converted to @p format.
     *
     * @param[in] name   Image file used to look up the raw tensor.
     * @param[in] format Format used to look up the raw tensor.
     *
     * @return a raw tensor for the specified image.
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
     *
     * @return a raw tensor for the specified image channel.
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
     *
     * @return a raw tensor for the specified image channel.
     */
    RawTensor get(const std::string &name, Channel channel);

    /** Provides a constant raw tensor for the specified channel after it has
     * been extracted form the given image formatted to @p format.
     *
     * @param[in] name    Image file used to look up the raw tensor.
     * @param[in] format  Format used to look up the raw tensor.
     * @param[in] channel Channel used to look up the raw tensor.
     *
     * @return a raw tensor for the specified image channel.
     */
    const RawTensor &get(const std::string &name, Format format, Channel channel) const;

    /** Provides a raw tensor for the specified channel after it has been
     * extracted form the given image formatted to @p format.
     *
     * @param[in] name    Image file used to look up the raw tensor.
     * @param[in] format  Format used to look up the raw tensor.
     * @param[in] channel Channel used to look up the raw tensor.
     *
     * @return a raw tensor for the specified image channel.
     */
    RawTensor get(const std::string &name, Format format, Channel channel);

    /** Puts garbage values all around the tensor for testing purposes
     *
     * @param[in, out] tensor       To be filled tensor.
     * @param[in]      distribution Distribution used to fill the tensor's surroundings.
     * @param[in]      seed_offset  The offset will be added to the global seed before initialising the random generator.
     */
    template <typename T, typename D>
    void fill_borders_with_garbage(T &&tensor, D &&distribution, std::random_device::result_type seed_offset) const;

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

    template <typename T, typename D>
    void fill_boxes(T &&tensor, D &&distribution, std::random_device::result_type seed_offset) const;

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

    /** Fills the specified @p tensor with the content of the raw tensor.
     *
     * @param[in, out] tensor To be filled tensor.
     * @param[in]      raw    Raw tensor used to fill the tensor.
     *
     * @warning No check is performed that the specified format actually
     *          matches the format of the tensor.
     */
    template <typename T>
    void fill(T &&tensor, RawTensor raw) const;

    /** Fill a tensor with uniform distribution
     *
     * @param[in, out] tensor      To be filled tensor.
     * @param[in]      seed_offset The offset will be added to the global seed before initialising the random generator.
     */
    template <typename T>
    void fill_tensor_uniform(T &&tensor, std::random_device::result_type seed_offset) const;

    /** Fill a tensor with uniform distribution
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

    /** Fill a tensor with uniform distribution across the specified range
     *
     * @param[in, out] tensor               To be filled tensor.
     * @param[in]      seed_offset          The offset will be added to the global seed before initialising the random generator.
     * @param[in]      excluded_range_pairs Ranges to exclude from the generator
     */
    template <typename T>
    void fill_tensor_uniform_ranged(T                                          &&tensor,
                                    std::random_device::result_type              seed_offset,
                                    const std::vector<AssetsLibrary::RangePair> &excluded_range_pairs) const;

    /** Fills the specified @p tensor with data loaded from .npy (numpy binary) in specified path.
     *
     * @param[in, out] tensor To be filled tensor.
     * @param[in]      name   Data file.
     *
     * @note The numpy array stored in the binary .npy file must be row-major in the sense that it
     * must store elements within a row consecutively in the memory, then rows within a 2D slice,
     * then 2D slices within a 3D slice and so on. Note that it imposes no restrictions on what
     * indexing convention is used in the numpy array. That is, the numpy array can be either fortran
     * style or C style as long as it adheres to the rule above.
     *
     * More concretely, the orders of dimensions for each style are as follows:
     * C-style (numpy default):
     *      array[HigherDims..., Z, Y, X]
     * Fortran style:
     *      array[X, Y, Z, HigherDims...]
     */
    template <typename T>
    void fill_layer_data(T &&tensor, std::string name) const;

    /** Fill a tensor with a constant value
     *
     * @param[in, out] tensor To be filled tensor.
     * @param[in]      value  Value to be assigned to all elements of the input tensor.
     *
     * @note    @p value must be of the same type as the data type of @p tensor
     */
    template <typename T, typename D>
    void fill_tensor_value(T &&tensor, D value) const;

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
    mutable arm_compute::Mutex      _format_lock{};
    mutable arm_compute::Mutex      _channel_lock{};
    const std::string               _library_path;
    std::random_device::result_type _seed;
};

namespace detail
{
template <typename T>
inline std::vector<std::pair<T, T>> convert_range_pair(const std::vector<AssetsLibrary::RangePair> &excluded_range_pairs)
{
    std::vector<std::pair<T, T>> converted;
    std::transform(excluded_range_pairs.begin(),
                   excluded_range_pairs.end(),
                   std::back_inserter(converted),
                   [](const AssetsLibrary::RangePair & p)
    {
        return std::pair<T, T>(static_cast<T>(p.first), static_cast<T>(p.second));
    });
    return converted;
}
} // namespace detail

template <typename T, typename D>
void AssetsLibrary::fill_borders_with_garbage(T &&tensor, D &&distribution, std::random_device::result_type seed_offset) const
{
    const PaddingSize padding_size = tensor.padding();

    Window window;
    window.set(0, Window::Dimension(-padding_size.left, tensor.shape()[0] + padding_size.right, 1));
    if(tensor.shape().num_dimensions() > 1)
    {
        window.set(1, Window::Dimension(-padding_size.top, tensor.shape()[1] + padding_size.bottom, 1));
    }

    std::mt19937 gen(_seed);

    execute_window_loop(window, [&](const Coordinates & id)
    {
        TensorShape shape = tensor.shape();

        // If outside of valid region
        if(id.x() < 0 || id.x() >= static_cast<int>(shape.x()) || id.y() < 0 || id.y() >= static_cast<int>(shape.y()))
        {
            using ResultType         = typename std::remove_reference<D>::type::result_type;
            const ResultType value   = distribution(gen);
            void *const      out_ptr = tensor(id);
            store_value_with_data_type(out_ptr, value, tensor.data_type());
        }
    });
}

template <typename T, typename D>
void AssetsLibrary::fill_boxes(T &&tensor, D &&distribution, std::random_device::result_type seed_offset) const
{
    using ResultType = typename std::remove_reference<D>::type::result_type;
    std::mt19937 gen(_seed + seed_offset);
    TensorShape  shape(tensor.shape());
    const int    num_boxes = tensor.num_elements() / 4;
    // Iterate over all elements
    std::uniform_real_distribution<> size_dist(0.f, 1.f);
    for(int element_idx = 0; element_idx < num_boxes * 4; element_idx += 4)
    {
        const ResultType delta   = size_dist(gen);
        const ResultType epsilon = size_dist(gen);
        const ResultType left    = distribution(gen);
        const ResultType top     = distribution(gen);
        const ResultType right   = left + delta;
        const ResultType bottom  = top + epsilon;
        const std::tuple<ResultType, ResultType, ResultType, ResultType> box(left, top, right, bottom);
        Coordinates x1              = index2coord(shape, element_idx);
        Coordinates y1              = index2coord(shape, element_idx + 1);
        Coordinates x2              = index2coord(shape, element_idx + 2);
        Coordinates y2              = index2coord(shape, element_idx + 3);
        ResultType &target_value_x1 = reinterpret_cast<ResultType *>(tensor(x1))[0];
        ResultType &target_value_y1 = reinterpret_cast<ResultType *>(tensor(y1))[0];
        ResultType &target_value_x2 = reinterpret_cast<ResultType *>(tensor(x2))[0];
        ResultType &target_value_y2 = reinterpret_cast<ResultType *>(tensor(y2))[0];
        store_value_with_data_type(&target_value_x1, std::get<0>(box), tensor.data_type());
        store_value_with_data_type(&target_value_y1, std::get<1>(box), tensor.data_type());
        store_value_with_data_type(&target_value_x2, std::get<2>(box), tensor.data_type());
        store_value_with_data_type(&target_value_y2, std::get<3>(box), tensor.data_type());
    }
    fill_borders_with_garbage(tensor, distribution, seed_offset);
}

template <typename T, typename D>
void AssetsLibrary::fill(T &&tensor, D &&distribution, std::random_device::result_type seed_offset) const
{
    using ResultType = typename std::remove_reference<D>::type::result_type;

    std::mt19937 gen(_seed + seed_offset);

    const bool  is_nhwc = tensor.data_layout() == DataLayout::NHWC;
    TensorShape shape(tensor.shape());

    if(is_nhwc)
    {
        // Ensure that the equivalent tensors will be filled for both data layouts
        permute(shape, PermutationVector(1U, 2U, 0U));
    }

    // Iterate over all elements
    for(int element_idx = 0; element_idx < tensor.num_elements(); ++element_idx)
    {
        Coordinates id = index2coord(shape, element_idx);

        if(is_nhwc)
        {
            // Write in the correct id for permuted shapes
            permute(id, PermutationVector(2U, 0U, 1U));
        }

        // Iterate over all channels
        for(int channel = 0; channel < tensor.num_channels(); ++channel)
        {
            const ResultType value        = distribution(gen);
            ResultType      &target_value = reinterpret_cast<ResultType *>(tensor(id))[channel];

            store_value_with_data_type(&target_value, value, tensor.data_type());
        }
    }

    fill_borders_with_garbage(tensor, distribution, seed_offset);
}

template <typename D>
void AssetsLibrary::fill(RawTensor &raw, D &&distribution, std::random_device::result_type seed_offset) const
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
void AssetsLibrary::fill(T &&tensor, const std::string &name, Format format) const
{
    const RawTensor &raw = get(name, format);

    for(size_t offset = 0; offset < raw.size(); offset += raw.element_size())
    {
        const Coordinates id = index2coord(raw.shape(), offset / raw.element_size());

        const RawTensor::value_type *const raw_ptr = raw.data() + offset;
        const auto                         out_ptr = static_cast<RawTensor::value_type *>(tensor(id));
        std::copy_n(raw_ptr, raw.element_size(), out_ptr);
    }
}

template <typename T>
void AssetsLibrary::fill(T &&tensor, const std::string &name, Channel channel) const
{
    fill(std::forward<T>(tensor), name, get_format_for_channel(channel), channel);
}

template <typename T>
void AssetsLibrary::fill(T &&tensor, const std::string &name, Format format, Channel channel) const
{
    const RawTensor &raw = get(name, format, channel);

    for(size_t offset = 0; offset < raw.size(); offset += raw.element_size())
    {
        const Coordinates id = index2coord(raw.shape(), offset / raw.element_size());

        const RawTensor::value_type *const raw_ptr = raw.data() + offset;
        const auto                         out_ptr = static_cast<RawTensor::value_type *>(tensor(id));
        std::copy_n(raw_ptr, raw.element_size(), out_ptr);
    }
}

template <typename T>
void AssetsLibrary::fill(T &&tensor, RawTensor raw) const
{
    for(size_t offset = 0; offset < raw.size(); offset += raw.element_size())
    {
        const Coordinates id = index2coord(raw.shape(), offset / raw.element_size());

        const RawTensor::value_type *const raw_ptr = raw.data() + offset;
        const auto                         out_ptr = static_cast<RawTensor::value_type *>(tensor(id));
        std::copy_n(raw_ptr, raw.element_size(), out_ptr);
    }
}

template <typename T>
void AssetsLibrary::fill_tensor_uniform(T &&tensor, std::random_device::result_type seed_offset) const
{
    switch(tensor.data_type())
    {
        case DataType::U8:
        case DataType::QASYMM8:
        {
            std::uniform_int_distribution<uint8_t> distribution_u8(std::numeric_limits<uint8_t>::lowest(), std::numeric_limits<uint8_t>::max());
            fill(tensor, distribution_u8, seed_offset);
            break;
        }
        case DataType::S8:
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
        case DataType::F16:
        {
            // It doesn't make sense to check [-inf, inf], so hard code it to a big number
            std::uniform_real_distribution<float> distribution_f16(-100.f, 100.f);
            fill(tensor, distribution_f16, seed_offset);
            break;
        }
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

template <typename T>
void AssetsLibrary::fill_tensor_uniform_ranged(T                                          &&tensor,
                                               std::random_device::result_type              seed_offset,
                                               const std::vector<AssetsLibrary::RangePair> &excluded_range_pairs) const
{
    using namespace arm_compute::utils::random;

    switch(tensor.data_type())
    {
        case DataType::U8:
        case DataType::QASYMM8:
        {
            const auto                         converted_pairs = detail::convert_range_pair<uint8_t>(excluded_range_pairs);
            RangedUniformDistribution<uint8_t> distribution_u8(std::numeric_limits<uint8_t>::lowest(),
                                                               std::numeric_limits<uint8_t>::max(),
                                                               converted_pairs);
            fill(tensor, distribution_u8, seed_offset);
            break;
        }
        case DataType::S8:
        {
            const auto                        converted_pairs = detail::convert_range_pair<int8_t>(excluded_range_pairs);
            RangedUniformDistribution<int8_t> distribution_s8(std::numeric_limits<int8_t>::lowest(),
                                                              std::numeric_limits<int8_t>::max(),
                                                              converted_pairs);
            fill(tensor, distribution_s8, seed_offset);
            break;
        }
        case DataType::U16:
        {
            const auto                          converted_pairs = detail::convert_range_pair<uint16_t>(excluded_range_pairs);
            RangedUniformDistribution<uint16_t> distribution_u16(std::numeric_limits<uint16_t>::lowest(),
                                                                 std::numeric_limits<uint16_t>::max(),
                                                                 converted_pairs);
            fill(tensor, distribution_u16, seed_offset);
            break;
        }
        case DataType::S16:
        {
            const auto                         converted_pairs = detail::convert_range_pair<int16_t>(excluded_range_pairs);
            RangedUniformDistribution<int16_t> distribution_s16(std::numeric_limits<int16_t>::lowest(),
                                                                std::numeric_limits<int16_t>::max(),
                                                                converted_pairs);
            fill(tensor, distribution_s16, seed_offset);
            break;
        }
        case DataType::U32:
        {
            const auto                          converted_pairs = detail::convert_range_pair<uint32_t>(excluded_range_pairs);
            RangedUniformDistribution<uint32_t> distribution_u32(std::numeric_limits<uint32_t>::lowest(),
                                                                 std::numeric_limits<uint32_t>::max(),
                                                                 converted_pairs);
            fill(tensor, distribution_u32, seed_offset);
            break;
        }
        case DataType::S32:
        {
            const auto                         converted_pairs = detail::convert_range_pair<int32_t>(excluded_range_pairs);
            RangedUniformDistribution<int32_t> distribution_s32(std::numeric_limits<int32_t>::lowest(),
                                                                std::numeric_limits<int32_t>::max(),
                                                                converted_pairs);
            fill(tensor, distribution_s32, seed_offset);
            break;
        }
        case DataType::F16:
        {
            // It doesn't make sense to check [-inf, inf], so hard code it to a big number
            const auto                       converted_pairs = detail::convert_range_pair<float>(excluded_range_pairs);
            RangedUniformDistribution<float> distribution_f16(-100.f, 100.f, converted_pairs);
            fill(tensor, distribution_f16, seed_offset);
            break;
        }
        case DataType::F32:
        {
            // It doesn't make sense to check [-inf, inf], so hard code it to a big number
            const auto                       converted_pairs = detail::convert_range_pair<float>(excluded_range_pairs);
            RangedUniformDistribution<float> distribution_f32(-1000.f, 1000.f, converted_pairs);
            fill(tensor, distribution_f32, seed_offset);
            break;
        }
        default:
            ARM_COMPUTE_ERROR("NOT SUPPORTED!");
    }
}

template <typename T, typename D>
void AssetsLibrary::fill_tensor_uniform(T &&tensor, std::random_device::result_type seed_offset, D low, D high) const
{
    switch(tensor.data_type())
    {
        case DataType::U8:
        case DataType::QASYMM8:
        {
            ARM_COMPUTE_ERROR_ON(!(std::is_same<uint8_t, D>::value));
            std::uniform_int_distribution<uint8_t> distribution_u8(low, high);
            fill(tensor, distribution_u8, seed_offset);
            break;
        }
        case DataType::S8:
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
        case DataType::F16:
        {
            std::uniform_real_distribution<float> distribution_f16(low, high);
            fill(tensor, distribution_f16, seed_offset);
            break;
        }
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
void AssetsLibrary::fill_layer_data(T &&tensor, std::string name) const
{
#ifdef _WIN32
    const std::string path_separator("\\");
#else  /* _WIN32 */
    const std::string path_separator("/");
#endif /* _WIN32 */
    const std::string path = _library_path + path_separator + name;

    std::vector<unsigned long> shape;

    // Open file
    std::ifstream stream(path, std::ios::in | std::ios::binary);
    if(!stream.good())
    {
        throw framework::FileNotFound("Could not load npy file: " + path);
    }
    std::string header = npy::read_header(stream);

    // Parse header
    bool        fortran_order = false;
    std::string typestr;
    npy::parse_header(header, typestr, fortran_order, shape);

    // Check if the typestring matches the given one
    std::string expect_typestr = get_typestring(tensor.data_type());
    ARM_COMPUTE_ERROR_ON_MSG(typestr != expect_typestr, "Typestrings mismatch");

    // Validate tensor shape
    ARM_COMPUTE_ERROR_ON_MSG(shape.size() != tensor.shape().num_dimensions(), "Tensor ranks mismatch");
    if(fortran_order)
    {
        for(size_t i = 0; i < shape.size(); ++i)
        {
            ARM_COMPUTE_ERROR_ON_MSG(tensor.shape()[i] != shape[i], "Tensor dimensions mismatch");
        }
    }
    else
    {
        for(size_t i = 0; i < shape.size(); ++i)
        {
            ARM_COMPUTE_ERROR_ON_MSG(tensor.shape()[i] != shape[shape.size() - i - 1], "Tensor dimensions mismatch");
        }
    }

    // Read data
    if(tensor.padding().empty())
    {
        // If tensor has no padding read directly from stream.
        stream.read(reinterpret_cast<char *>(tensor.data()), tensor.size());
    }
    else
    {
        // If tensor has padding accessing tensor elements through execution window.
        Window window;
        window.use_tensor_dimensions(tensor.shape());

        execute_window_loop(window, [&](const Coordinates & id)
        {
            stream.read(reinterpret_cast<char *>(tensor(id)), tensor.element_size());
        });
    }
}

template <typename T, typename D>
void AssetsLibrary::fill_tensor_value(T &&tensor, D value) const
{
    fill_tensor_uniform(tensor, 0, value, value);
}
} // namespace test
} // namespace arm_compute
#endif /* __ARM_COMPUTE_TEST_TENSOR_LIBRARY_H__ */
