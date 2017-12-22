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
#include "tests/AssetsLibrary.h"

#include "Utils.h"
#include "utils/TypePrinter.h"

#include "arm_compute/core/ITensor.h"

#include <cctype>
#include <fstream>
#include <limits>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <utility>

namespace arm_compute
{
namespace test
{
namespace
{
template <typename T, typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
void rgb_to_luminance(const RawTensor &src, RawTensor &dst)
{
    const size_t min_size = std::min(src.size(), dst.size());

    for(size_t i = 0, j = 0; i < min_size; i += 3, ++j)
    {
        reinterpret_cast<T *>(dst.data())[j] = 0.2126f * src.data()[i + 0] + 0.7152f * src.data()[i + 1] + 0.0722f * src.data()[i + 2];
    }
}

void extract_r_from_rgb(const RawTensor &src, RawTensor &dst)
{
    const size_t min_size = std::min(src.size(), dst.size());

    for(size_t i = 0, j = 0; i < min_size; i += 3, ++j)
    {
        dst.data()[j] = src.data()[i];
    }
}

void extract_g_from_rgb(const RawTensor &src, RawTensor &dst)
{
    const size_t min_size = std::min(src.size(), dst.size());

    for(size_t i = 1, j = 0; i < min_size; i += 3, ++j)
    {
        dst.data()[j] = src.data()[i];
    }
}

void discard_comments(std::ifstream &fs)
{
    while(fs.peek() == '#')
    {
        fs.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }
}

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

std::tuple<unsigned int, unsigned int, int> parse_ppm_header(std::ifstream &fs)
{
    // Check the PPM magic number is valid
    std::array<char, 2> magic_number{ { 0 } };
    fs >> magic_number[0] >> magic_number[1];

    if(magic_number[0] != 'P' || magic_number[1] != '6')
    {
        throw std::runtime_error("Only raw PPM format is suported");
    }

    discard_comments_and_spaces(fs);

    unsigned int width = 0;
    fs >> width;

    discard_comments_and_spaces(fs);

    unsigned int height = 0;
    fs >> height;

    discard_comments_and_spaces(fs);

    int max_value = 0;
    fs >> max_value;

    if(!fs.good())
    {
        throw std::runtime_error("Cannot read image dimensions");
    }

    if(max_value != 255)
    {
        throw std::runtime_error("RawTensor doesn't have 8-bit values");
    }

    discard_comments(fs);

    if(isspace(fs.peek()) == 0)
    {
        throw std::runtime_error("Invalid PPM header");
    }

    fs.ignore(1);

    return std::make_tuple(width, height, max_value);
}

RawTensor load_ppm(const std::string &path)
{
    std::ifstream file(path, std::ios::in | std::ios::binary);

    if(!file.good())
    {
        throw framework::FileNotFound("Could not load PPM image: " + path);
    }

    unsigned int width  = 0;
    unsigned int height = 0;

    std::tie(width, height, std::ignore) = parse_ppm_header(file);

    RawTensor raw(TensorShape(width, height), Format::RGB888);

    // Check if the file is large enough to fill the image
    const size_t current_position = file.tellg();
    file.seekg(0, std::ios_base::end);
    const size_t end_position = file.tellg();
    file.seekg(current_position, std::ios_base::beg);

    if((end_position - current_position) < raw.size())
    {
        throw std::runtime_error("Not enough data in file");
    }

    file.read(reinterpret_cast<std::fstream::char_type *>(raw.data()), raw.size());

    if(!file.good())
    {
        throw std::runtime_error("Failure while reading image buffer");
    }

    return raw;
}
} // namespace

AssetsLibrary::AssetsLibrary(std::string path, std::random_device::result_type seed) //NOLINT
    : _library_path(std::move(path)),
      _seed{ seed }
{
}

std::string AssetsLibrary::path() const
{
    return _library_path;
}

std::random_device::result_type AssetsLibrary::seed() const
{
    return _seed;
}

void AssetsLibrary::fill(RawTensor &raw, const std::string &name, Format format) const
{
    const RawTensor &src = get(name, format);
    std::copy_n(src.data(), raw.size(), raw.data());
}

void AssetsLibrary::fill(RawTensor &raw, const std::string &name, Channel channel) const
{
    fill(raw, name, get_format_for_channel(channel), channel);
}

void AssetsLibrary::fill(RawTensor &raw, const std::string &name, Format format, Channel channel) const
{
    const RawTensor &src = get(name, format, channel);
    std::copy_n(src.data(), raw.size(), raw.data());
}

const AssetsLibrary::Loader &AssetsLibrary::get_loader(const std::string &extension) const
{
    static std::unordered_map<std::string, Loader> loaders =
    {
        { "ppm", load_ppm }
    };

    const auto it = loaders.find(extension);

    if(it != loaders.end())
    {
        return it->second;
    }
    else
    {
        throw std::invalid_argument("Cannot load image with extension '" + extension + "'");
    }
}

const AssetsLibrary::Converter &AssetsLibrary::get_converter(Format src, Format dst) const
{
    static std::map<std::pair<Format, Format>, Converter> converters =
    {
        { std::make_pair(Format::RGB888, Format::U8), rgb_to_luminance<uint8_t> },
        { std::make_pair(Format::RGB888, Format::U16), rgb_to_luminance<uint16_t> },
        { std::make_pair(Format::RGB888, Format::S16), rgb_to_luminance<int16_t> },
        { std::make_pair(Format::RGB888, Format::U32), rgb_to_luminance<uint32_t> }
    };

    const auto it = converters.find(std::make_pair(src, dst));

    if(it != converters.end())
    {
        return it->second;
    }
    else
    {
        std::stringstream msg;
        msg << "Cannot convert from format '" << src << "' to format '" << dst << "'\n";
        throw std::invalid_argument(msg.str());
    }
}

const AssetsLibrary::Converter &AssetsLibrary::get_converter(DataType src, Format dst) const
{
    static std::map<std::pair<DataType, Format>, Converter> converters = {};

    const auto it = converters.find(std::make_pair(src, dst));

    if(it != converters.end())
    {
        return it->second;
    }
    else
    {
        std::stringstream msg;
        msg << "Cannot convert from data type '" << src << "' to format '" << dst << "'\n";
        throw std::invalid_argument(msg.str());
    }
}

const AssetsLibrary::Converter &AssetsLibrary::get_converter(DataType src, DataType dst) const
{
    static std::map<std::pair<DataType, DataType>, Converter> converters = {};

    const auto it = converters.find(std::make_pair(src, dst));

    if(it != converters.end())
    {
        return it->second;
    }
    else
    {
        std::stringstream msg;
        msg << "Cannot convert from data type '" << src << "' to data type '" << dst << "'\n";
        throw std::invalid_argument(msg.str());
    }
}

const AssetsLibrary::Converter &AssetsLibrary::get_converter(Format src, DataType dst) const
{
    static std::map<std::pair<Format, DataType>, Converter> converters = {};

    const auto it = converters.find(std::make_pair(src, dst));

    if(it != converters.end())
    {
        return it->second;
    }
    else
    {
        std::stringstream msg;
        msg << "Cannot convert from format '" << src << "' to data type '" << dst << "'\n";
        throw std::invalid_argument(msg.str());
    }
}

const AssetsLibrary::Extractor &AssetsLibrary::get_extractor(Format format, Channel channel) const
{
    static std::map<std::pair<Format, Channel>, Extractor> extractors =
    {
        { std::make_pair(Format::RGB888, Channel::R), extract_r_from_rgb },
        { std::make_pair(Format::RGB888, Channel::G), extract_g_from_rgb }
    };

    const auto it = extractors.find(std::make_pair(format, channel));

    if(it != extractors.end())
    {
        return it->second;
    }
    else
    {
        std::stringstream msg;
        msg << "Cannot extract channel '" << channel << "' from format '" << format << "'\n";
        throw std::invalid_argument(msg.str());
    }
}

RawTensor AssetsLibrary::load_image(const std::string &name) const
{
#ifdef _WIN32
    const std::string image_path = ("\\images\\");
#else  /* _WIN32 */
    const std::string image_path = ("/images/");
#endif /* _WIN32 */

    const std::string path      = _library_path + image_path + name;
    const std::string extension = path.substr(path.find_last_of('.') + 1);
    return (*get_loader(extension))(path);
}

const RawTensor &AssetsLibrary::find_or_create_raw_tensor(const std::string &name, Format format) const
{
    std::lock_guard<std::mutex> guard(_format_lock);

    const RawTensor *ptr = _cache.find(std::forward_as_tuple(name, format));

    if(ptr != nullptr)
    {
        return *ptr;
    }

    RawTensor raw = load_image(name);

    if(raw.format() != format)
    {
        RawTensor dst(raw.shape(), format);
        (*get_converter(raw.format(), format))(raw, dst);
        raw = std::move(dst);
    }

    return _cache.add(std::forward_as_tuple(name, format), std::move(raw));
}

const RawTensor &AssetsLibrary::find_or_create_raw_tensor(const std::string &name, Format format, Channel channel) const
{
    std::lock_guard<std::mutex> guard(_channel_lock);

    const RawTensor *ptr = _cache.find(std::forward_as_tuple(name, format, channel));

    if(ptr != nullptr)
    {
        return *ptr;
    }

    const RawTensor &src = get(name, format);
    RawTensor dst(src.shape(), get_channel_format(channel));

    (*get_extractor(format, channel))(src, dst);

    return _cache.add(std::forward_as_tuple(name, format, channel), std::move(dst));
}

TensorShape AssetsLibrary::get_image_shape(const std::string &name)
{
    return load_image(name).shape();
}

const RawTensor &AssetsLibrary::get(const std::string &name) const
{
    return find_or_create_raw_tensor(name, Format::RGB888);
}

RawTensor AssetsLibrary::get(const std::string &name)
{
    return RawTensor(find_or_create_raw_tensor(name, Format::RGB888));
}

RawTensor AssetsLibrary::get(const std::string &name, DataType data_type, int num_channels) const
{
    const RawTensor &raw = get(name);

    return RawTensor(raw.shape(), data_type, num_channels);
}

const RawTensor &AssetsLibrary::get(const std::string &name, Format format) const
{
    return find_or_create_raw_tensor(name, format);
}

RawTensor AssetsLibrary::get(const std::string &name, Format format)
{
    return RawTensor(find_or_create_raw_tensor(name, format));
}

const RawTensor &AssetsLibrary::get(const std::string &name, Channel channel) const
{
    return get(name, get_format_for_channel(channel), channel);
}

RawTensor AssetsLibrary::get(const std::string &name, Channel channel)
{
    return RawTensor(get(name, get_format_for_channel(channel), channel));
}

const RawTensor &AssetsLibrary::get(const std::string &name, Format format, Channel channel) const
{
    return find_or_create_raw_tensor(name, format, channel);
}

RawTensor AssetsLibrary::get(const std::string &name, Format format, Channel channel)
{
    return RawTensor(find_or_create_raw_tensor(name, format, channel));
}
} // namespace test
} // namespace arm_compute
