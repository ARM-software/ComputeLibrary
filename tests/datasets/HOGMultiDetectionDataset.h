/*
 * Copyright (c) 2018 ARM Limited.
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
#ifndef ARM_COMPUTE_HOG_MULTI_DETECTION_DATASET
#define ARM_COMPUTE_HOG_MULTI_DETECTION_DATASET

#include "arm_compute/core/HOGInfo.h"
#include "tests/framework/datasets/Datasets.h"

namespace arm_compute
{
namespace test
{
namespace datasets
{
class HOGMultiDetectionDataset
{
public:
    using type = std::tuple<std::string, std::vector<HOGInfo>>;

    struct iterator
    {
        iterator(std::vector<std::string>::const_iterator          image_it,
                 std::vector<std::string>::const_iterator          hog_infos_name_it,
                 std::vector<std::vector<HOGInfo>>::const_iterator hog_infos_it)
            : _image_it{ std::move(image_it) },
              _hog_infos_name_it{ std::move(hog_infos_name_it) },
              _hog_infos_it{ std::move(hog_infos_it) }
        {
        }

        std::string description() const
        {
            std::stringstream description;
            description << "Image=" << *_image_it << ":";
            description << "HOGInfoSet=" << *_hog_infos_name_it;

            return description.str();
        }

        HOGMultiDetectionDataset::type operator*() const
        {
            return std::make_tuple(*_image_it, *_hog_infos_it);
        }

        iterator &operator++()
        {
            ++_image_it;
            ++_hog_infos_name_it;
            ++_hog_infos_it;

            return *this;
        }

    private:
        std::vector<std::string>::const_iterator          _image_it;
        std::vector<std::string>::const_iterator          _hog_infos_name_it;
        std::vector<std::vector<HOGInfo>>::const_iterator _hog_infos_it;
    };

    iterator begin() const
    {
        return iterator(_image.begin(), _hog_infos_name.begin(), _hog_infos.begin());
    }

    int size() const
    {
        return std::min(_image.size(), _hog_infos.size());
    }

    void add_config(std::string          image,
                    std::string          hog_infos_name,
                    std::vector<HOGInfo> hog_info_vec)
    {
        _image.emplace_back(std::move(image));
        _hog_infos_name.emplace_back(std::move(hog_infos_name));
        _hog_infos.emplace_back(hog_info_vec);
    }

protected:
    HOGMultiDetectionDataset()                            = default;
    HOGMultiDetectionDataset(HOGMultiDetectionDataset &&) = default;

private:
    std::vector<std::string>          _image{};
    std::vector<std::string>          _hog_infos_name{};
    std::vector<std::vector<HOGInfo>> _hog_infos{};
};

using MultiHOGDataset = std::vector<HOGInfo>;

// *INDENT-OFF*
// clang-format off
static const MultiHOGDataset mixed
{
    //      cell_size         block_size        detection_size      block_stride      bin normalization_type    thresh phase_type
    HOGInfo(Size2D(8U, 8U),   Size2D(16U, 16U), Size2D(64U, 128U),  Size2D(8U, 8U),   3U, HOGNormType::L1_NORM, 0.2f,  PhaseType::SIGNED),
    HOGInfo(Size2D(8U, 8U),   Size2D(16U, 16U), Size2D(128U, 256U), Size2D(8U, 8U),   5U, HOGNormType::L1_NORM, 0.3f,  PhaseType::SIGNED),
    HOGInfo(Size2D(16U, 16U), Size2D(32U, 32U), Size2D(64U, 128U),  Size2D(32U, 32U), 7U, HOGNormType::L1_NORM, 0.4f,  PhaseType::SIGNED),
    HOGInfo(Size2D(16U, 16U), Size2D(32U, 32U), Size2D(128U, 256U), Size2D(32U, 32U), 9U, HOGNormType::L1_NORM, 0.5f,  PhaseType::SIGNED),
};

// cell_size and bin_size fixed
static const MultiHOGDataset skip_binning
{
    //      cell_size       block_size        detection_size      block_stride      bin normalization_type       thresh phase_type
    HOGInfo(Size2D(8U, 8U), Size2D(16U, 16U), Size2D(64U, 128U),  Size2D(8U, 8U),   9U, HOGNormType::L2HYS_NORM, 0.2f,  PhaseType::SIGNED),
    HOGInfo(Size2D(8U, 8U), Size2D(16U, 16U), Size2D(128U, 256U), Size2D(8U, 8U),   9U, HOGNormType::L2HYS_NORM, 0.2f,  PhaseType::SIGNED),
    HOGInfo(Size2D(8U, 8U), Size2D(32U, 32U), Size2D(64U, 128U),  Size2D(16U, 16U), 9U, HOGNormType::L2HYS_NORM, 0.2f,  PhaseType::SIGNED),
    HOGInfo(Size2D(8U, 8U), Size2D(32U, 32U), Size2D(128U, 256U), Size2D(16U, 16U), 9U, HOGNormType::L2HYS_NORM, 0.2f,  PhaseType::SIGNED),
};

// cell_size and bin_size and block_size and block_stride fixed
static const MultiHOGDataset skip_normalization
{
    //      cell_size       block_size        detection_size      block_stride    bin normalization_type    thresh phase_type
    HOGInfo(Size2D(8U, 8U), Size2D(16U, 16U), Size2D(64U, 128U),  Size2D(8U, 8U), 9U, HOGNormType::L2_NORM, 0.2f,  PhaseType::SIGNED),
    HOGInfo(Size2D(8U, 8U), Size2D(16U, 16U), Size2D(128U, 256U), Size2D(8U, 8U), 9U, HOGNormType::L2_NORM, 0.3f,  PhaseType::SIGNED),
    HOGInfo(Size2D(8U, 8U), Size2D(16U, 16U), Size2D(64U, 128U),  Size2D(8U, 8U), 9U, HOGNormType::L2_NORM, 0.4f,  PhaseType::SIGNED),
    HOGInfo(Size2D(8U, 8U), Size2D(16U, 16U), Size2D(128U, 256U), Size2D(8U, 8U), 9U, HOGNormType::L2_NORM, 0.5f,  PhaseType::SIGNED),
};
// clang-format on
// *INDENT-ON*

class SmallHOGMultiDetectionDataset final : public HOGMultiDetectionDataset
{
public:
    SmallHOGMultiDetectionDataset()
    {
        add_config("800x600.ppm", "MIXED", mixed);
        add_config("800x600.ppm", "SKIP_BINNING", skip_binning);
        add_config("800x600.ppm", "SKIP_NORMALIZATION", skip_normalization);
    }
};

class LargeHOGMultiDetectionDataset final : public HOGMultiDetectionDataset
{
public:
    LargeHOGMultiDetectionDataset()
    {
        add_config("1920x1080.ppm", "MIXED", mixed);
        add_config("1920x1080.ppm", "SKIP_BINNING", skip_binning);
        add_config("1920x1080.ppm", "SKIP_NORMALIZATION", skip_normalization);
    }
};

} // namespace datasets
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_HOG_MULTI_DETECTION_DATASET */
