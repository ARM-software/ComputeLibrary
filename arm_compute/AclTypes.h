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
#ifndef ARM_COMPUTE_ACL_TYPES_H_
#define ARM_COMPUTE_ACL_TYPES_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/**< Opaque Context object */
typedef struct AclContext_ *AclContext;
/**< Opaque Queue object */
typedef struct AclQueue_ *AclQueue;
/**< Opaque Tensor object */
typedef struct AclTensor_ *AclTensor;
/**< Opaque Tensor pack object */
typedef struct AclTensorPack_ *AclTensorPack;
/**< Opaque Operator object */
typedef struct AclOperator_ *AclOperator;

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

/**< Supported tuning modes */
typedef enum
{
    AclTuningModeNone = 0, /**< No tuning */
    AclRapid          = 1, /**< Fast tuning mode, testing a small portion of the tuning space */
    AclNormal         = 2, /**< Normal tuning mode, gives a good balance between tuning mode and performance */
    AclExhaustive     = 3, /**< Exhaustive tuning mode, increased tuning time but with best results */
} AclTuningMode;

/**< Queue options */
typedef struct
{
    AclTuningMode mode;          /**< Tuning mode */
    int32_t       compute_units; /**< Compute Units that the queue will deploy */
} AclQueueOptions;

/**< Supported data types */
typedef enum AclDataType
{
    AclDataTypeUnknown = 0, /**< Unknown data type */
    AclUInt8           = 1, /**< 8-bit unsigned integer */
    AclInt8            = 2, /**< 8-bit signed integer */
    AclUInt16          = 3, /**< 16-bit unsigned integer */
    AclInt16           = 4, /**< 16-bit signed integer */
    AclUint32          = 5, /**< 32-bit unsigned integer */
    AclInt32           = 6, /**< 32-bit signed integer */
    AclFloat16         = 7, /**< 16-bit floating point */
    AclBFloat16        = 8, /**< 16-bit brain floating point */
    AclFloat32         = 9, /**< 32-bit floating point */
} AclDataType;

/**< Supported data layouts for operations */
typedef enum AclDataLayout
{
    AclDataLayoutUnknown = 0, /**< Unknown data layout */
    AclNhwc              = 1, /**< Native, performant, Compute Library data layout */
    AclNchw              = 2, /**< Data layout where width is the fastest changing dimension */
} AclDataLayout;

/** Type of memory to be imported */
typedef enum AclImportMemoryType
{
    AclHostPtr = 0 /**< Host allocated memory */
} AclImportMemoryType;

/**< Tensor Descriptor */
typedef struct AclTensorDescriptor
{
    int32_t     ndims;     /**< Number or dimensions */
    int32_t    *shape;     /**< Tensor Shape */
    AclDataType data_type; /**< Tensor Data type */
    int64_t    *strides;   /**< Strides on each dimension. Linear memory is assumed if nullptr */
    int64_t     boffset;   /**< Offset in terms of bytes for the first element */
} AclTensorDescriptor;

/**< Slot type of a tensor */
typedef enum
{
    AclSlotUnknown = -1,
    AclSrc         = 0,
    AclSrc0        = 0,
    AclSrc1        = 1,
    AclDst         = 30,
    AclSrcVec      = 256,
} AclTensorSlot;

#ifdef __cplusplus
}
#endif /* __cplusplus */
#endif /* ARM_COMPUTE_ACL_TYPES_H_ */
