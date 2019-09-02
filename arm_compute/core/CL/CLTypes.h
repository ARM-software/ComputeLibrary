/*
 * Copyright (c) 2017-2019 ARM Limited.
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
#ifndef __ARM_COMPUTE_CL_TYPES_H__
#define __ARM_COMPUTE_CL_TYPES_H__

#include "arm_compute/core/CL/ICLArray.h"
#include "arm_compute/core/GPUTarget.h"

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
    std::string name;        /**< Device name */
    std::string extensions;  /**< List of supported extensions */
    std::string ddk_version; /**< DDK version */
    GPUTarget   gpu_target;  /**< GPU target architecture/instance */
    size_t      num_cores;   /**< Number of cores */
    size_t      cache_size;  /**< Cache size */
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
} // namespace arm_compute
#endif /* __ARM_COMPUTE_CL_TYPES_H__ */
