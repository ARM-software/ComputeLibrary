/*
 * Copyright (c) 2016, 2018 ARM Limited.
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
#ifndef __ARM_COMPUTE_HOGINFO_H__
#define __ARM_COMPUTE_HOGINFO_H__

#include "arm_compute/core/Size2D.h"
#include "arm_compute/core/Types.h"

#include <cstddef>

namespace arm_compute
{
/** Store the HOG's metadata */
class HOGInfo
{
public:
    /** Default constructor */
    HOGInfo();
    /** Default destructor */
    virtual ~HOGInfo() = default;
    /** Allow instances of this class to be copy constructed */
    HOGInfo(const HOGInfo &) = default;
    /** Allow instances of this class to be copied */
    HOGInfo &operator=(const HOGInfo &) = default;
    /** Allow instances of this class to be move constructed */
    HOGInfo(HOGInfo &&) = default;
    /** Allow instances of this class to be moved */
    HOGInfo &operator=(HOGInfo &&) = default;
    /** Constructor
     *
     * @param[in] cell_size             Cell size in pixels
     * @param[in] block_size            Block size in pixels. Must be a multiple of cell_size.
     * @param[in] detection_window_size Detection window size in pixels. Must be a multiple of block_size and block_stride.
     * @param[in] block_stride          Distance in pixels between 2 consecutive blocks along the x and y direction. Must be a multiple of cell size
     * @param[in] num_bins              Number of histogram bins for each cell
     * @param[in] normalization_type    (Optional) Normalization type to use for each block
     * @param[in] l2_hyst_threshold     (Optional) Threshold used for L2HYS_NORM normalization method
     * @param[in] phase_type            (Optional) Type of @ref PhaseType
     */
    HOGInfo(const Size2D &cell_size, const Size2D &block_size, const Size2D &detection_window_size, const Size2D &block_stride, size_t num_bins,
            HOGNormType normalization_type = HOGNormType::L2HYS_NORM, float l2_hyst_threshold = 0.2f, PhaseType phase_type = PhaseType::UNSIGNED);
    /** Initialize the metadata structure with the given parameters
     *
     * @param[in] cell_size             Cell size in pixels
     * @param[in] block_size            Block size in pixels. Must be a multiple of cell_size.
     * @param[in] detection_window_size Detection window size in pixels. Must be a multiple of block_size and block_stride.
     * @param[in] block_stride          Distance in pixels between 2 consecutive blocks along the x and y direction. Must be a multiple of cell size
     * @param[in] num_bins              Number of histogram bins for each cell
     * @param[in] normalization_type    (Optional) Normalization type to use for each block
     * @param[in] l2_hyst_threshold     (Optional) Threshold used for L2HYS_NORM normalization method
     * @param[in] phase_type            (Optional) Type of @ref PhaseType
     */
    void init(const Size2D &cell_size, const Size2D &block_size, const Size2D &detection_window_size, const Size2D &block_stride, size_t num_bins,
              HOGNormType normalization_type = HOGNormType::L2HYS_NORM, float l2_hyst_threshold = 0.2f, PhaseType phase_type = PhaseType::UNSIGNED);
    /** The cell size in pixels
     *
     * @return The cell size in pixels
     */
    const Size2D &cell_size() const;
    /** The block size in pixels
     *
     * @return The block size in pixels
     */
    const Size2D &block_size() const;
    /** The detection window size in pixels
     *
     * @return The detection window size in pixels
     */
    const Size2D &detection_window_size() const;
    /** The block stride in pixels. The block stride is the distance between 2 consecutive blocks
     *
     * @return The block stride in pixels
     */
    const Size2D &block_stride() const;
    /** The number of histogram bins for each cell
     *
     * @return The number of histogram bins for each cell
     */
    size_t num_bins() const;
    /** The normalization type
     *
     * @return The normalization type
     */
    HOGNormType normalization_type() const;
    /** Threshold used for L2HYS_NORM normalization type
     *
     * @return Threshold used for L2HYS_NORM normalization type
     */
    float l2_hyst_threshold() const;
    /** The type of @ref PhaseType
     *
     * @return The type of @ref PhaseType
     */
    PhaseType phase_type() const;
    /** The size of HOG descriptor
     *
     * @return The size of HOG descriptor
     */
    size_t descriptor_size() const;
    /** Calculates the number of cells for each block
     *
     * @return The Size2D data object which stores the number of cells along the x and y directions
     */
    Size2D num_cells_per_block() const;

    /** Calculates the number of cells per block stride
     *
     * @return The Size2D data object which stores the number of cells per block stride along the x and y directions
     */
    Size2D num_cells_per_block_stride() const;
    /** Calculates the number of blocks for the given image size
     *
     * @param[in] image_size The input image size data object
     *
     * @return The Size2D data object which stores the number of blocks along the x and y directions
     */
    Size2D num_blocks_per_image(const Size2D &image_size) const;

private:
    Size2D      _cell_size;
    Size2D      _block_size;
    Size2D      _detection_window_size;
    Size2D      _block_stride;
    size_t      _num_bins;
    HOGNormType _normalization_type;
    float       _l2_hyst_threshold;
    PhaseType   _phase_type;
    size_t      _descriptor_size;
};
}
#endif /*__ARM_COMPUTE_HOGINFO_H__ */
