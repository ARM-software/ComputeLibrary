/*
 * Copyright (c) 2016-2019 ARM Limited.
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
#include "arm_compute/core/HOGInfo.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Validate.h"

using namespace arm_compute;

HOGInfo::HOGInfo()
    : _cell_size(), _block_size(), _detection_window_size(), _block_stride(), _num_bins(0), _normalization_type(HOGNormType::L2HYS_NORM), _l2_hyst_threshold(0.0f), _phase_type(PhaseType::UNSIGNED),
      _descriptor_size(0)
{
}

HOGInfo::HOGInfo(const Size2D &cell_size, const Size2D &block_size, const Size2D &detection_window_size, const Size2D &block_stride, size_t num_bins,
                 HOGNormType normalization_type, float l2_hyst_threshold, PhaseType phase_type)
    : HOGInfo()
{
    init(cell_size, block_size, detection_window_size, block_stride, num_bins, normalization_type, l2_hyst_threshold, phase_type);
}

void HOGInfo::init(const Size2D &cell_size, const Size2D &block_size, const Size2D &detection_window_size, const Size2D &block_stride, size_t num_bins,
                   HOGNormType normalization_type, float l2_hyst_threshold, PhaseType phase_type)
{
    ARM_COMPUTE_ERROR_ON_MSG((block_size.width % cell_size.width), "The block width must be multiple of cell width");
    ARM_COMPUTE_ERROR_ON_MSG((block_size.height % cell_size.height), "Block height must be multiple of cell height");
    ARM_COMPUTE_ERROR_ON_MSG((block_stride.width % cell_size.width), "Block stride width must be multiple of cell width");
    ARM_COMPUTE_ERROR_ON_MSG((block_stride.height % cell_size.height), "Block stride height must be multiple of cell height");
    ARM_COMPUTE_ERROR_ON_MSG(((detection_window_size.width - block_size.width) % block_stride.width), "Window width must be multiple of block width and block stride width");
    ARM_COMPUTE_ERROR_ON_MSG(((detection_window_size.height - block_size.height) % block_stride.height), "Window height must be multiple of block height and block stride height");

    _cell_size             = cell_size;
    _block_size            = block_size;
    _detection_window_size = detection_window_size;
    _block_stride          = block_stride;
    _num_bins              = num_bins;
    _normalization_type    = normalization_type;
    _l2_hyst_threshold     = l2_hyst_threshold;
    _phase_type            = phase_type;

    // Compute descriptor size. +1 takes into account of the bias
    _descriptor_size = num_cells_per_block().area() * num_block_positions_per_image(_detection_window_size).area() * _num_bins + 1;
}

Size2D HOGInfo::num_cells_per_block() const
{
    ARM_COMPUTE_ERROR_ON(_cell_size.width == 0 || _cell_size.height == 0);

    return Size2D{ _block_size.width / _cell_size.width,
                   _block_size.height / _cell_size.height };
}

Size2D HOGInfo::num_cells_per_block_stride() const
{
    ARM_COMPUTE_ERROR_ON(_cell_size.width == 0 || _cell_size.height == 0);

    return Size2D{ _block_stride.width / _cell_size.width,
                   _block_stride.height / _cell_size.height };
}

Size2D HOGInfo::num_block_positions_per_image(const Size2D &image_size) const
{
    ARM_COMPUTE_ERROR_ON(_block_stride.width == 0 || _block_stride.height == 0);

    return Size2D{ ((image_size.width - _block_size.width) / _block_stride.width) + 1,
                   ((image_size.height - _block_size.height) / _block_stride.height) + 1 };
}

const Size2D &HOGInfo::cell_size() const
{
    return _cell_size;
}

const Size2D &HOGInfo::block_size() const
{
    return _block_size;
}

const Size2D &HOGInfo::detection_window_size() const
{
    return _detection_window_size;
}

const Size2D &HOGInfo::block_stride() const
{
    return _block_stride;
}

size_t HOGInfo::num_bins() const
{
    return _num_bins;
}

HOGNormType HOGInfo::normalization_type() const
{
    return _normalization_type;
}

float HOGInfo::l2_hyst_threshold() const
{
    return _l2_hyst_threshold;
}

PhaseType HOGInfo::phase_type() const
{
    return _phase_type;
}

size_t HOGInfo::descriptor_size() const
{
    return _descriptor_size;
}
