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
#include "HOGDescriptor.h"

#include "Derivative.h"
#include "Magnitude.h"
#include "Phase.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace reference
{
namespace
{
template <typename T>
void hog_orientation_compute(const SimpleTensor<T> &mag, const SimpleTensor<T> &phase, std::vector<T> &bins, const HOGInfo &hog_info)
{
    const size_t num_bins    = hog_info.num_bins();
    const size_t cell_height = hog_info.cell_size().height;
    const size_t cell_width  = hog_info.cell_size().width;

    float phase_scale = (PhaseType::SIGNED == hog_info.phase_type() ? num_bins / 360.0f : num_bins / 180.0f);
    phase_scale *= (PhaseType::SIGNED == hog_info.phase_type() ? 360.0f / 255.0f : 1.0f);

    int row_idx = 0;
    for(size_t yc = 0; yc < cell_height; ++yc)
    {
        for(size_t xc = 0; xc < cell_height; xc++)
        {
            const float mag_value   = mag[(row_idx + xc)];
            const float phase_value = phase[(row_idx + xc)] * phase_scale + 0.5f;
            const float w1          = phase_value - floor(phase_value);

            // The quantised phase is the histogram index [0, num_bins - 1]
            // Check limit of histogram index. If hidx == num_bins, hidx = 0
            const auto hidx = static_cast<unsigned int>(phase_value) % num_bins;

            // Weighted vote between 2 bins
            bins[hidx] += mag_value * (1.0f - w1);
            bins[(hidx + 1) % num_bins] += mag_value * w1;
        }

        row_idx += cell_width;
    }
}

template <typename T>
void hog_block_normalization_compute(SimpleTensor<T> &block, SimpleTensor<T> &desc, const HOGInfo &hog_info, size_t block_idx)
{
    const int         num_bins_per_block = desc.num_channels();
    const HOGNormType norm_type          = hog_info.normalization_type();
    const Coordinates id                 = index2coord(desc.shape(), block_idx);

    float sum = 0.0f;

    // Calculate sum
    for(int i = 0; i < num_bins_per_block; ++i)
    {
        const float val = block[i];
        sum += (norm_type == HOGNormType::L1_NORM) ? std::fabs(val) : val * val;
    }

    // Calculate normalization scale
    float scale = 1.0f / (std::sqrt(sum) + num_bins_per_block * 0.1f);

    if(norm_type == HOGNormType::L2HYS_NORM)
    {
        // Reset sum
        sum = 0.0f;
        for(int i = 0; i < num_bins_per_block; ++i)
        {
            float val = block[i] * scale;

            // Clip scaled input_value if over l2_hyst_threshold
            val = fmin(val, hog_info.l2_hyst_threshold());
            sum += val * val;
            block[i] = val;
        }

        // We use the same constants of OpenCV
        scale = 1.0f / (std::sqrt(sum) + 1e-3f);
    }

    for(int i = 0; i < num_bins_per_block; ++i)
    {
        block[i] *= scale;
        reinterpret_cast<float *>(desc(id))[i] = block[i];
    }
}
} // namespace

template <typename T, typename U, typename V>
void hog_orientation_binning(const SimpleTensor<T> &mag, const SimpleTensor<U> &phase, SimpleTensor<V> &hog_space, const HOGInfo &hog_info)
{
    const size_t cell_width   = hog_info.cell_size().width;
    const size_t cell_height  = hog_info.cell_size().height;
    const size_t shape_width  = hog_space.shape().x() * hog_info.cell_size().width;
    const size_t shape_height = hog_space.shape().y() * hog_info.cell_size().height;

    SimpleTensor<V> mag_cell(TensorShape(cell_width, cell_height), DataType::F32);
    SimpleTensor<V> phase_cell(TensorShape(cell_width, cell_height), DataType::F32);

    int cell_idx = 0;
    int y_offset = 0;
    int x_offset = 0;

    // Traverse shape
    for(auto sy = cell_height - 1; sy < shape_height; sy += cell_height)
    {
        x_offset = 0;
        for(auto sx = cell_width - 1; sx < shape_width; sx += cell_width)
        {
            int row_idx  = 0;
            int elem_idx = 0;

            // Traverse cell
            for(auto y = 0u; y < cell_height; ++y)
            {
                for(auto x = 0u; x < cell_width; ++x)
                {
                    int shape_idx        = x + row_idx + x_offset + y_offset;
                    mag_cell[elem_idx]   = mag[shape_idx];
                    phase_cell[elem_idx] = phase[shape_idx];
                    elem_idx++;
                }

                row_idx += shape_width;
            }

            // Partition magnitude values into bins based on phase values
            std::vector<V> bins(hog_info.num_bins());
            hog_orientation_compute(mag_cell, phase_cell, bins, hog_info);

            for(size_t i = 0; i < hog_info.num_bins(); ++i)
            {
                hog_space[cell_idx * hog_info.num_bins() + i] = bins[i];
            }

            x_offset += cell_width;
            cell_idx++;
        }

        y_offset += (cell_height * shape_width);
    }
}

template <typename T>
void hog_block_normalization(SimpleTensor<T> &desc, const SimpleTensor<T> &hog_space, const HOGInfo &hog_info)
{
    const Size2D cells_per_block        = hog_info.num_cells_per_block();
    const Size2D cells_per_block_stride = hog_info.num_cells_per_block_stride();

    const size_t block_width         = hog_info.block_size().width;
    const size_t block_height        = hog_info.block_size().height;
    const size_t block_stride_width  = hog_info.block_stride().width;
    const size_t block_stride_height = hog_info.block_stride().height;
    const size_t shape_width         = hog_space.shape().x() * hog_info.cell_size().width;
    const size_t shape_height        = hog_space.shape().y() * hog_info.cell_size().height;

    const size_t num_bins     = hog_info.num_bins();
    const size_t num_channels = cells_per_block.area() * num_bins;

    SimpleTensor<T> block(TensorShape{ 1u, 1u }, DataType::F32, num_channels);

    int block_idx      = 0;
    int block_y_offset = 0;

    // Traverse shape
    for(auto sy = block_width - 1; sy < shape_height; sy += block_stride_height)
    {
        int block_x_offset = 0;
        for(auto sx = block_height - 1; sx < shape_width; sx += block_stride_width)
        {
            int cell_y_offset = 0;
            int elem_idx      = 0;

            // Traverse block
            for(auto y = 0u; y < cells_per_block.height; ++y)
            {
                int cell_x_offset = 0;
                for(auto x = 0u; x < cells_per_block.width; ++x)
                {
                    for(auto bin = 0u; bin < num_bins; ++bin)
                    {
                        int idx         = bin + cell_x_offset + cell_y_offset + block_x_offset + block_y_offset;
                        block[elem_idx] = hog_space[idx];
                        elem_idx++;
                    }

                    cell_x_offset += num_bins;
                }

                cell_y_offset += hog_space.shape().x() * num_bins;
            }

            // Normalize block and write to descriptor
            hog_block_normalization_compute(block, desc, hog_info, block_idx);

            block_x_offset += cells_per_block_stride.width * num_bins;
            block_idx++;
        }

        block_y_offset += cells_per_block_stride.height * num_bins * hog_space.shape().x();
    }
}

template <typename T, typename U>
SimpleTensor<T> hog_descriptor(const SimpleTensor<U> &src, BorderMode border_mode, U constant_border_value, const HOGInfo &hog_info)
{
    SimpleTensor<int16_t> _mag;
    SimpleTensor<uint8_t> _phase;

    SimpleTensor<int16_t> grad_x;
    SimpleTensor<int16_t> grad_y;

    // Create tensor info for HOG descriptor
    TensorInfo      desc_info(hog_info, src.shape().x(), src.shape().y());
    SimpleTensor<T> desc(desc_info.tensor_shape(), DataType::F32, desc_info.num_channels());

    // Create HOG space tensor (num_cells_x, num_cells_y)
    TensorShape hog_space_shape(src.shape().x() / hog_info.cell_size().width,
                                src.shape().y() / hog_info.cell_size().height);

    // For each cell a histogram with a num_bins is created
    TensorInfo      info_hog_space(hog_space_shape, hog_info.num_bins(), DataType::F32);
    SimpleTensor<T> hog_space(info_hog_space.tensor_shape(), DataType::F32, info_hog_space.num_channels());

    // Calculate derivative
    std::tie(grad_x, grad_y) = derivative<int16_t>(src, border_mode, constant_border_value, GradientDimension::GRAD_XY);

    // Calculate magnitude and phase
    _mag   = magnitude(grad_x, grad_y, MagnitudeType::L2NORM);
    _phase = phase(grad_x, grad_y, hog_info.phase_type());

    // For each cell create histogram based on magnitude and phase
    hog_orientation_binning(_mag, _phase, hog_space, hog_info);

    // Normalize histograms based on block size
    hog_block_normalization(desc, hog_space, hog_info);

    return desc;
}

template SimpleTensor<float> hog_descriptor(const SimpleTensor<uint8_t> &src, BorderMode border_mode, uint8_t constant_border_value, const HOGInfo &hog_info);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
