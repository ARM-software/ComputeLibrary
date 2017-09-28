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
#ifndef __ARM_COMPUTE_TEST_TENSOR_CACHE_H__
#define __ARM_COMPUTE_TEST_TENSOR_CACHE_H__

#include "RawTensor.h"

#include <map>
#include <mutex>
#include <utility>

namespace arm_compute
{
namespace test
{
/** Stores @ref RawTensor categorised by the image they are created from
 * including name, format and channel.
 */
class TensorCache
{
public:
    /* Search the cache for a tensor of created from the specified image and
     * format.
     *
     * @param[in] key Key to look up the tensor. Consists of image name and format.
     *
     * @return The cached tensor matching the image name and format if found. A
     *         nullptr otherwise.
     */
    RawTensor *find(std::tuple<const std::string &, Format> key);

    /* Search the cache for a tensor of created from the specified image,
     * format and channel.
     *
     * @param[in] key Key to look up the tensor. Consists of image name, format and channel.
     *
     * @return The cached tensor matching the image name and format if found. A
     *         nullptr otherwise.
     */
    RawTensor *find(std::tuple<const std::string &, Format, Channel> key);

    /** Add the given tensor to the cache. Can later be found under the given
     * image name and format.
     *
     * @param[in] key Key under which to store the tensor. Consists of image name and format.
     * @param[in] raw Raw tensor to be stored.
     *
     * @return A reference to the cached tensor.
     */
    RawTensor &add(std::tuple<const std::string &, Format> key, RawTensor raw);

    /** Add the given tensor to the cache. Can later be found under the given
     * image name, format and channel.
     *
     * @param[in] key Key under which to store the tensor. Consists of image name, format and channel.
     * @param[in] raw Raw tensor to be stored.
     *
     * @return A reference to the cached tensor.
     */
    RawTensor &add(std::tuple<const std::string &, Format, Channel> key, RawTensor raw);

private:
    using FormatMap  = std::map<std::tuple<std::string, Format>, RawTensor>;
    using ChannelMap = std::map<std::tuple<std::string, Format, Channel>, RawTensor>;

    FormatMap  _raw_tensor_cache{};
    ChannelMap _raw_tensor_channel_cache{};
    std::mutex _raw_tensor_cache_mutex{};
    std::mutex _raw_tensor_channel_cache_mutex{};
};

inline RawTensor *TensorCache::find(std::tuple<const std::string &, Format> key)
{
    const auto it = _raw_tensor_cache.find(key);
    return it == _raw_tensor_cache.end() ? nullptr : &it->second;
}

inline RawTensor *TensorCache::find(std::tuple<const std::string &, Format, Channel> key)
{
    const auto it = _raw_tensor_channel_cache.find(key);
    return it == _raw_tensor_channel_cache.end() ? nullptr : &it->second;
}

inline RawTensor &TensorCache::add(std::tuple<const std::string &, Format> key, RawTensor raw)
{
    std::lock_guard<std::mutex> lock(_raw_tensor_channel_cache_mutex);
    return std::get<0>(_raw_tensor_cache.emplace(std::move(key), std::move(raw)))->second;
}

inline RawTensor &TensorCache::add(std::tuple<const std::string &, Format, Channel> key, RawTensor raw)
{
    std::lock_guard<std::mutex> lock(_raw_tensor_channel_cache_mutex);
    return std::get<0>(_raw_tensor_channel_cache.emplace(std::move(key), std::move(raw)))->second;
}
} // namespace test
} // namespace arm_compute
#endif /* __ARM_COMPUTE_TEST_TENSOR_CACHE_H__ */
