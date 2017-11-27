/*
 * Copyright (c) 2017, 2018 ARM Limited.
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
#ifndef ARM_COMPUTE_TEST_HOG_DESCRIPTOR_DATASET
#define ARM_COMPUTE_TEST_HOG_DESCRIPTOR_DATASET

#include "utils/TypePrinter.h"

namespace arm_compute
{
namespace test
{
namespace datasets
{
class HOGDescriptorDataset
{
public:
    using type = std::tuple<std::string, HOGInfo>;

    struct iterator
    {
        iterator(std::vector<std::string>::const_iterator image_it,
                 std::vector<HOGInfo>::const_iterator     hog_info_it)
            : _image_it{ std::move(image_it) },
              _hog_info_it{ std::move(hog_info_it) }
        {
        }

        std::string description() const
        {
            std::stringstream description;
            description << "Image=" << *_image_it << ":";
            description << "HOGInfo=" << *_hog_info_it;

            return description.str();
        }

        HOGDescriptorDataset::type operator*() const
        {
            return std::make_tuple(*_image_it, *_hog_info_it);
        }

        iterator &operator++()
        {
            ++_image_it;
            ++_hog_info_it;

            return *this;
        }

    private:
        std::vector<std::string>::const_iterator _image_it;
        std::vector<HOGInfo>::const_iterator     _hog_info_it;
    };

    iterator begin() const
    {
        return iterator(_image.begin(), _hog_info.begin());
    }

    int size() const
    {
        return std::min(_image.size(), _hog_info.size());
    }

    void add_config(std::string image,
                    Size2D cell_size, Size2D block_size, Size2D detection_window_size, Size2D block_stride,
                    size_t num_bins, HOGNormType normalization_type, float l2_hyst_threshold, PhaseType phase_type)
    {
        _image.emplace_back(std::move(image));
        _hog_info.emplace_back(HOGInfo(cell_size, block_size, detection_window_size, block_stride, num_bins, normalization_type, l2_hyst_threshold, phase_type));
    }

protected:
    HOGDescriptorDataset()                        = default;
    HOGDescriptorDataset(HOGDescriptorDataset &&) = default;

private:
    std::vector<std::string> _image{};
    std::vector<HOGInfo>     _hog_info{};
};

// *INDENT-OFF*
// clang-format off
class SmallHOGDescriptorDataset final : public HOGDescriptorDataset
{
public:
    SmallHOGDescriptorDataset()
    {
        //         image          cell_size       block_size        detection_size     block_stride    bin  normalization_type       thresh phase_type
        add_config("800x600.ppm", Size2D(8U, 8U), Size2D(16U, 16U), Size2D(64U, 128U), Size2D(8U, 8U), 9U,  HOGNormType::L2HYS_NORM, 0.2f,  PhaseType::SIGNED);
        add_config("800x600.ppm", Size2D(8U, 8U), Size2D(16U, 16U), Size2D(64U, 128U), Size2D(8U, 8U), 9U,  HOGNormType::L2HYS_NORM, 0.2f,  PhaseType::UNSIGNED);
    }
};

class LargeHOGDescriptorDataset final : public HOGDescriptorDataset
{
public:
    LargeHOGDescriptorDataset()
    {
        //         image            cell_size       block_size        detection_size     block_stride    bin  normalization_type       thresh phase_type
        add_config("1920x1080.ppm", Size2D(8U, 8U), Size2D(16U, 16U), Size2D(64U, 128U), Size2D(8U, 8U), 9U,  HOGNormType::L2HYS_NORM, 0.2f,  PhaseType::SIGNED);
        add_config("1920x1080.ppm", Size2D(8U, 8U), Size2D(16U, 16U), Size2D(64U, 128U), Size2D(8U, 8U), 9U,  HOGNormType::L2_NORM,    0.2f,  PhaseType::SIGNED);
        add_config("1920x1080.ppm", Size2D(8U, 8U), Size2D(16U, 16U), Size2D(64U, 128U), Size2D(8U, 8U), 9U,  HOGNormType::L1_NORM,    0.2f,  PhaseType::SIGNED);

        add_config("1920x1080.ppm", Size2D(8U, 8U), Size2D(16U, 16U), Size2D(64U, 128U), Size2D(8U, 8U), 9U,  HOGNormType::L2HYS_NORM, 0.2f,  PhaseType::UNSIGNED);
        add_config("1920x1080.ppm", Size2D(8U, 8U), Size2D(16U, 16U), Size2D(64U, 128U), Size2D(8U, 8U), 9U,  HOGNormType::L2_NORM,    0.2f,  PhaseType::UNSIGNED);
        add_config("1920x1080.ppm", Size2D(8U, 8U), Size2D(16U, 16U), Size2D(64U, 128U), Size2D(8U, 8U), 9U,  HOGNormType::L1_NORM,    0.2f,  PhaseType::UNSIGNED);
    }
};
// clang-format on
// *INDENT-ON*

} // namespace datasets
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_HOG_DESCRIPTOR_DATASET */
