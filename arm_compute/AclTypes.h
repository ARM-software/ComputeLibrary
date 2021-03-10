/*
 * Copyright (c) 2021 Arm Limited.
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
#ifndef ARM_COMPUTE_ACLTYPES_H_
#define ARM_COMPUTE_ACLTYPES_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/**< Opaque Context object */
typedef struct AclContext_ *AclContext;

// Capabilities bitfield (Note: if multiple are enabled ComputeLibrary will pick the best possible)
typedef uint64_t AclTargetCapabilities;

/**< Error codes returned by the public entry-points */
typedef enum AclStatus : int32_t
{
    AclSuccess            = 0, /**< Call succeeded, leading to valid state for all involved objects/data */
    AclRuntimeError       = 1, /**< Call failed during execution */
    AclOutOfMemory        = 2, /**< Call failed due to failure to allocate resources */
    AclUnimplemented      = 3, /**< Call failed as requested capability is not implemented */
    AclUnsupportedTarget  = 4, /**< Call failed as an invalid backend was requested */
    AclInvalidTarget      = 5, /**< Call failed as invalid argument was passed */
    AclInvalidArgument    = 6, /**< Call failed as invalid argument was passed */
    AclUnsupportedConfig  = 7, /**< Call failed as configuration is unsupported */
    AclInvalidObjectState = 8, /**< Call failed as an object has invalid state */
} AclStatus;

/**< Supported CPU targets */
typedef enum AclTarget
{
    AclCpu    = 0, /**< Cpu target that uses SIMD extensions */
    AclGpuOcl = 1, /**< OpenCL target for GPU */
} AclTarget;

/** Execution mode types */
typedef enum AclExecutionMode
{
    AclPreferFastRerun = 0, /**< Prioritize performance when multiple iterations are performed */
    AclPreferFastStart = 1, /**< Prioritize performance when a single iterations is expected to be performed */
} AclExecutionMode;

/** Available CPU capabilities */
typedef enum AclCpuCapabilities
{
    AclCpuCapabilitiesAuto = 0, /**< Automatic discovery of capabilities */

    AclCpuCapabilitiesNeon = (1 << 0), /**< Enable NEON optimized paths */
    AclCpuCapabilitiesSve  = (1 << 1), /**< Enable SVE optimized paths */
    AclCpuCapabilitiesSve2 = (1 << 2), /**< Enable SVE2 optimized paths */
    // Reserve 3, 4, 5, 6

    AclCpuCapabilitiesFp16 = (1 << 7), /**< Enable float16 data-type support */
    AclCpuCapabilitiesBf16 = (1 << 8), /**< Enable bfloat16 data-type support */
    // Reserve 9, 10, 11, 12

    AclCpuCapabilitiesDot      = (1 << 13), /**< Enable paths that use the udot/sdot instructions */
    AclCpuCapabilitiesMmlaInt8 = (1 << 14), /**< Enable paths that use the mmla integer instructions */
    AclCpuCapabilitiesMmlaFp   = (1 << 15), /**< Enable paths that use the mmla float instructions */

    AclCpuCapabilitiesAll = ~0 /**< Enable all paths */
} AclCpuCapabilities;

/**< Allocator interface that can be passed to a context */
typedef struct AclAllocator
{
    /** Allocate a block of size bytes of memory.
     *
     * @param[in] user_data User provided data that can be used by the allocator
     * @param[in] size      Size of the allocation
     *
     * @return A pointer to the allocated block if successfull else NULL
     */
    void *(*alloc)(void *user_data, size_t size);
    /** Release a block of size bytes of memory.
     *
     * @param[in] user_data User provided data that can be used by the allocator
     * @param[in] size      Size of the allocation
     */
    void (*free)(void *user_data, void *ptr);
    /** Allocate a block of size bytes of memory.
     *
     * @param[in] user_data User provided data that can be used by the allocator
     * @param[in] size      Size of the allocation
     *
     * @return A pointer to the allocated block if successfull else NULL
     */
    void *(*aligned_alloc)(void *user_data, size_t size, size_t alignment);
    /** Allocate a block of size bytes of memory.
     *
     * @param[in] user_data User provided data that can be used by the allocator
     * @param[in] size      Size of the allocation
     */
    void (*aligned_free)(void *user_data, void *ptr);

    /**< User provided information */
    void *user_data;
} AclAllocator;

/**< Context options */
typedef struct AclContextOptions
{
    AclExecutionMode      mode;               /**< Execution mode to use */
    AclTargetCapabilities capabilities;       /**< Target capabilities */
    bool                  enable_fast_math;   /**< Allow precision loss */
    const char           *kernel_config_file; /**< Kernel cofiguration file */
    int32_t               max_compute_units;  /**< Max compute units that can be used by a queue created from the context.
                                                   If <=0 the system will use the hw concurency insted */
    AclAllocator         *allocator;          /**< Allocator to be used by all the memory internally */
} AclContextOptions;

/** Default context */
const AclContextOptions acl_default_ctx_options =
{
    AclPreferFastRerun,     /* mode */
    AclCpuCapabilitiesAuto, /* capabilities */
    false,                  /* enable_fast_math */
    "default.mlgo",         /* kernel_config_file */
    -1,                     /* max_compute_units */
    nullptr                 /* allocator */
};

#ifdef __cplusplus
}
#endif /* __cplusplus */
#endif /* ARM_COMPUTE_ACLTYPES_H_ */
