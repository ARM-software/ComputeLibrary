/*
 * Copyright (c) 2018-2019 ARM Limited.
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
#ifndef ARM_COMPUTE_TEST_IMAGE_FILE_DATASET
#define ARM_COMPUTE_TEST_IMAGE_FILE_DATASET

#include "tests/framework/datasets/Datasets.h"

#include <type_traits>

namespace arm_compute
{
namespace test
{
namespace datasets
{
class ImageFileDataset
{
public:
    struct iterator
    {
        iterator(std::vector<std::string>::const_iterator name_it)
            : _name_it{ std::move(name_it) }
        {
        }

        std::string description() const
        {
            std::stringstream description;
            description << "ImageFile=" << *_name_it;
            return description.str();
        }

        std::tuple<std::string> operator*() const
        {
            return std::make_tuple(*_name_it);
        }

        iterator &operator++()
        {
            ++_name_it;

            return *this;
        }

    private:
        std::vector<std::string>::const_iterator _name_it;
    };

    iterator begin() const
    {
        return iterator(_names.begin());
    }

    int size() const
    {
        return _names.size();
    }

    void add_image_file(std::string name)
    {
        _names.emplace_back(std::move(name));
    }

protected:
    ImageFileDataset()                    = default;
    ImageFileDataset(ImageFileDataset &&) = default;

private:
    std::vector<std::string> _names{};
};

/** Data set containing names of small image files. */
class SmallImageFiles final : public ImageFileDataset
{
public:
    SmallImageFiles()
    {
        add_image_file("640x480.ppm");
        add_image_file("800x600.ppm");
    }
};

/** Data set containing names of large image files. */
class LargeImageFiles final : public ImageFileDataset
{
public:
    LargeImageFiles()
    {
        add_image_file("1280x720.ppm");
        add_image_file("1920x1080.ppm");
    }
};

} // namespace datasets
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_IMAGE_FILE_DATASET */
