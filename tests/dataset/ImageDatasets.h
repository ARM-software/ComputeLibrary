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
#ifndef __ARM_COMPUTE_TEST_IMAGE_DATASETS_H__
#define __ARM_COMPUTE_TEST_IMAGE_DATASETS_H__

#include <string>
#include <type_traits>

#ifdef BOOST
#include "boost_wrapper.h"
#endif

namespace arm_compute
{
namespace test
{
/** Abstract data set containing image names.
 *
 * Can be used as input for Boost data test cases to automatically run a test
 * case on different images.
 */
template <unsigned int Size>
class ImageDataset
{
public:
    /** Type of the samples in the data set. */
    using sample = const std::string;

    /** Dimensionality of the data set. */
    enum
    {
        arity = 1
    };

    /** Number of samples in the data set. */
#ifdef BOOST
    boost::unit_test::data::size_t size() const
#else
    unsigned int size() const
#endif
    {
        return _images.size();
    }

    /** Type of the iterator used to step through all samples in the data set.
     * Needs to support operator*() and operator++() which a pointer does.
     */
    using iterator = const std::string *;

    /** Iterator to the first sample in the data set. */
    iterator begin() const
    {
        return _images.data();
    }

protected:
    /** Protected constructor to make the class abstract. */
    template <typename... Ts>
    ImageDataset(Ts... images)
        : _images{ { images... } }
    {
    }

    /** Protected destructor to prevent deletion of derived class through a
     * pointer to the base class.
     */
    ~ImageDataset() = default;

private:
    std::array<std::string, Size> _images;
};

/** Data set containing names of small images. */
class SmallImages final : public ImageDataset<2>
{
public:
    SmallImages()
        : ImageDataset("128x128.ppm", "640x480.ppm")
    {
    }
};

/** Data set containing names of large images. */
class LargeImages final : public ImageDataset<3>
{
public:
    LargeImages()
#ifdef INTERNAL_ONLY
        : ImageDataset("1280x720.ppm", "1920x1080.ppm", "4160x3120.ppm")
          // The 4k image is too large to distribute
#else
        : ImageDataset("1280x720.ppm", "1920x1080.ppm")
#endif /* INTERNAL_ONLY */
    {
    }
};
} // namespace test
} // namespace arm_compute
#endif
