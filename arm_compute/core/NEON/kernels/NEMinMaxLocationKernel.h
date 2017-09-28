/*
 * Copyright (c) 2016, 2017 ARM Limited.
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
#ifndef __ARM_COMPUTE_NEMINMAXLOCATIONKERNEL_H__
#define __ARM_COMPUTE_NEMINMAXLOCATIONKERNEL_H__

#include "arm_compute/core/IArray.h"
#include "arm_compute/core/NEON/INEKernel.h"
#include "support/Mutex.h"

#include <cstdint>

namespace arm_compute
{
class ITensor;
using IImage = ITensor;

/** Interface for the kernel to perform min max search on an image. */
class NEMinMaxKernel : public INEKernel
{
public:
    /** Default constructor */
    NEMinMaxKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEMinMaxKernel(const NEMinMaxKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEMinMaxKernel &operator=(const NEMinMaxKernel &) = delete;
    /** Allow instances of this class to be moved */
    NEMinMaxKernel(NEMinMaxKernel &&) = default;
    /** Allow instances of this class to be moved */
    NEMinMaxKernel &operator=(NEMinMaxKernel &&) = default;
    /** Default destructor */
    ~NEMinMaxKernel() = default;

    /** Initialise the kernel's input and outputs.
     *
     * @param[in]  input Input Image. Data types supported: U8/S16/F32.
     * @param[out] min   Minimum value of image. Data types supported: S32 if input type is U8/S16, F32 if input type is F32.
     * @param[out] max   Maximum value of image. Data types supported: S32 if input type is U8/S16, F32 if input type is F32.
     */
    void configure(const IImage *input, void *min, void *max);
    /** Resets global minimum and maximum. */
    void reset();

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

private:
    /** Performs the min/max algorithm on U8 images on a given window.
     *
     * @param win The window to run the algorithm on.
     */
    void minmax_U8(Window win);
    /** Performs the min/max algorithm on S16 images on a given window.
     *
     * @param win The window to run the algorithm on.
     */
    void minmax_S16(Window win);
    /** Performs the min/max algorithm on F32 images on a given window.
     *
     * @param win The window to run the algorithm on.
     */
    void minmax_F32(Window win);
    /** Common signature for all the specialised MinMax functions
     *
     * @param[in] window Region on which to execute the kernel.
     */
    using MinMaxFunction = void (NEMinMaxKernel::*)(Window window);
    /** MinMax function to use for the particular image types passed to configure() */
    MinMaxFunction _func;
    /** Helper to update min/max values **/
    template <typename T>
    void update_min_max(T min, T max);

    const IImage      *_input; /**< Input image. */
    void              *_min;   /**< Minimum value. */
    void              *_max;   /**< Maximum value. */
    arm_compute::Mutex _mtx;   /**< Mutex used for result reduction. */
};

/** Interface for the kernel to find min max locations of an image. */
class NEMinMaxLocationKernel : public INEKernel
{
public:
    /** Default constructor */
    NEMinMaxLocationKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEMinMaxLocationKernel(const NEMinMaxLocationKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEMinMaxLocationKernel &operator=(const NEMinMaxLocationKernel &) = delete;
    /** Allow instances of this class to be moved */
    NEMinMaxLocationKernel(NEMinMaxLocationKernel &&) = default;
    /** Allow instances of this class to be moved */
    NEMinMaxLocationKernel &operator=(NEMinMaxLocationKernel &&) = default;
    /** Default destructor */
    ~NEMinMaxLocationKernel() = default;

    /** Initialise the kernel's input and outputs.
     *
     * @param[in]  input     Input Image. Data types supported: U8/S16/F32.
     * @param[out] min       Minimum value of image. Data types supported: S32 if input type is U8/S16, F32 if input type is F32.
     * @param[out] max       Maximum value of image. Data types supported: S32 if input type is U8/S16, F32 if input type is F32.
     * @param[out] min_loc   Array of minimum value locations.
     * @param[out] max_loc   Array of maximum value locations.
     * @param[out] min_count Number of minimum value encounters.
     * @param[out] max_count Number of maximum value encounters.
     */
    void configure(const IImage *input, void *min, void *max,
                   ICoordinates2DArray *min_loc = nullptr, ICoordinates2DArray *max_loc = nullptr,
                   uint32_t *min_count = nullptr, uint32_t *max_count = nullptr);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;
    bool is_parallelisable() const override;

private:
    /** Performs the min/max location algorithm on T type images on a given window.
     *
     * @param win The window to run the algorithm on.
     */
    template <class T, bool count_min, bool count_max, bool loc_min, bool loc_max>
    void minmax_loc(const Window &win);
    /** Common signature for all the specialised MinMaxLoc functions
     *
     * @param[in] window Region on which to execute the kernel.
     */
    using MinMaxLocFunction = void (NEMinMaxLocationKernel::*)(const Window &window);
    /** MinMaxLoc function to use for the particular image types passed to configure() */
    MinMaxLocFunction _func;
    /** Helper to create a function pointer table for the parameterized MinMaxLocation functions. */
    template <class T, typename>
    struct create_func_table;

    const IImage        *_input;     /**< Input image. */
    void                *_min;       /**< Minimum value. */
    void                *_max;       /**< Maximum value. */
    uint32_t            *_min_count; /**< Count of minimum value encounters. */
    uint32_t            *_max_count; /**< Count of maximum value encounters. */
    ICoordinates2DArray *_min_loc;   /**< Locations of minimum values. */
    ICoordinates2DArray *_max_loc;   /**< Locations of maximum values. */
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_NEMINMAXLOCATIONKERNEL_H__ */
