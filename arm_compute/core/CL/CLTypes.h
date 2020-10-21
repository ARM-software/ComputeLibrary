/*
 * Copyright (c) 2017-2020 Arm Limited.
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
#ifndef ARM_COMPUTE_CL_TYPES_H
#define ARM_COMPUTE_CL_TYPES_H

#include "arm_compute/core/CL/ICLArray.h"
#include "arm_compute/core/GPUTarget.h"

#include <set>
#include <string>

namespace arm_compute
{
/** Default string for the CLKernel configuration id */
static const std::string default_config_id = "no_config_id";

/** Available OpenCL Version */
enum class CLVersion
{
    CL10,   /* the OpenCL 1.0 */
    CL11,   /* the OpenCL 1.1 */
    CL12,   /* the OpenCL 1.2 */
    CL20,   /* the OpenCL 2.0 and above */
    UNKNOWN /* unkown version */
};

/** OpenCL device options */
struct CLDeviceOptions
{
    std::string           name{};           /**< Device name */
    std::string           device_version{}; /**< Device version string */
    std::set<std::string> extensions{};     /**< List of supported extensions */
    std::string           ddk_version{};    /**< DDK version */
    GPUTarget             gpu_target{};     /**< GPU target architecture/instance */
    CLVersion             version{};        /**< Device OpenCL version */
    size_t                compute_units{};  /**< Number of compute units */
    size_t                cache_size{};     /**< Cache size */
};

/** OpenCL quantization data */
struct CLQuantization
{
    /** Default Constructor */
    CLQuantization()
        : scale(nullptr), offset(nullptr) {};
    /** Constructor
     *
     * @param[in] scale  OpenCL scale array
     * @param[in] offset OpenCL offset array
     */
    CLQuantization(const ICLFloatArray *scale, const ICLInt32Array *offset)
        : scale(scale), offset(offset) {};

    const ICLFloatArray *scale;  /**< Quantization scale array */
    const ICLInt32Array *offset; /**< Quantization offset array */
};

/** Internal keypoint structure for Lucas-Kanade Optical Flow */
struct CLLKInternalKeypoint
{
    float x{ 0.f };               /**< x coordinate of the keypoint */
    float y{ 0.f };               /**< y coordinate of the keypoint */
    float tracking_status{ 0.f }; /**< the tracking status of the keypoint */
    float dummy{ 0.f };           /**< Dummy field, to make sure the data structure 128-bit align, so that GPU can use vload4 */
};

/** Structure for storing Spatial Gradient Matrix and the minimum eigenvalue for each keypoint */
struct CLCoefficientTable
{
    float A11;     /**< iA11 * FLT_SCALE */
    float A12;     /**< iA11 * FLT_SCALE */
    float A22;     /**< iA11 * FLT_SCALE */
    float min_eig; /**< Minimum eigenvalue */
};

/** Structure for storing ival, ixval and iyval for each point inside the window */
struct CLOldValue
{
    int16_t ival;  /**< ival extracts from old image */
    int16_t ixval; /**< ixval extracts from scharr Gx image */
    int16_t iyval; /**< iyval extracts from scharr Gy image */
    int16_t dummy; /**< Dummy field, to make sure the data structure 128-bit align, so that GPU can use vload4 */
};

/** Interface for OpenCL Array of Internal Key Points. */
using ICLLKInternalKeypointArray = ICLArray<CLLKInternalKeypoint>;
/** Interface for OpenCL Array of Coefficient Tables. */
using ICLCoefficientTableArray = ICLArray<CLCoefficientTable>;
/** Interface for OpenCL Array of Old Values. */
using ICLOldValArray = ICLArray<CLOldValue>;

} // namespace arm_compute
#endif /* ARM_COMPUTE_CL_TYPES_H */
