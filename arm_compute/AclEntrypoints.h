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
#ifndef ARM_COMPUTE_ACL_ENTRYPOINTS_H_
#define ARM_COMPUTE_ACL_ENTRYPOINTS_H_

#include "arm_compute/AclTypes.h"

#ifdef __cplusplus
extern "C" {
#endif /** __cplusplus */

/** Create a context object
 *
 * Context is responsible for retaining internal information and work as an aggregate service mechanism
 *
 * @param[in, out] ctx     A valid non-zero context object if no failure occurs
 * @param[in]      target  Target to create the context for
 * @param[in]      options Context options to be used for all the kernels that are created under the context
 *
 * @return Status code
 *
 * Returns:
 *  - @ref AclSuccess if function was completed successfully
 *  - @ref AclOutOfMemory if there was a failure allocating memory resources
 *  - @ref AclUnsupportedTarget if the requested target is unsupported
 *  - @ref AclInvalidArgument if a given argument is invalid
 */
AclStatus AclCreateContext(AclContext              *ctx,
                           AclTarget                target,
                           const AclContextOptions *options);

/** Destroy a given context object
 *
 * @param[in] ctx A valid context object to destroy
 *
 * @return Status code
 *
 * Returns:
 *  - @ref AclSuccess if functions was completed successfully
 *  - @ref AclInvalidArgument if the provided context is invalid
 */
AclStatus AclDestroyContext(AclContext ctx);

/** Create an operator queue
 *
 * Queue is responsible for any scheduling related activities
 *
 * @param[in, out] queue   A valid non-zero queue object is not failures occur
 * @param[in]      ctx     Context to be used
 * @param[in]      options Queue options to be used for the operators using the queue
 *
 * @return Status code
 *
 * Returns:
 *  - @ref AclSuccess if function was completed successfully
 *  - @ref AclOutOfMemory if there was a failure allocating memory resources
 *  - @ref AclUnsupportedTarget if the requested target is unsupported
 *  - @ref AclInvalidArgument if a given argument is invalid
 */
AclStatus AclCreateQueue(AclQueue *queue, AclContext ctx, const AclQueueOptions *options);

/** Wait until all elements on the queue have been completed
 *
 * @param[in] queue Queue to wait on completion
 *
 * @return Status code
 *
 * Returns:
 *  - @ref AclSuccess if functions was completed successfully
 *  - @ref AclInvalidArgument if the provided queue is invalid
 *  - @ref AclRuntimeError on any other runtime related error
 */
AclStatus AclQueueFinish(AclQueue queue);

/** Destroy a given queue object
 *
 * @param[in] queue A valid context object to destroy
 *
 * @return Status code
 *
 * Returns:
 *  - @ref AclSuccess if functions was completed successfully
 *  - @ref AclInvalidArgument if the provided context is invalid
 */
AclStatus AclDestroyQueue(AclQueue queue);

/** Create a Tensor object
 *
 * Tensor is a generalized matrix construct that can represent up to ND dimensionality (where N = 6 for Compute Library)
 * The object holds a backing memory along-side to operate on
 *
 * @param[in, out] tensor   A valid non-zero tensor object if no failures occur
 * @param[in]      ctx      Context to be used
 * @param[in]      desc     Tensor representation meta-data
 * @param[in]      allocate Instructs allocation of the tensor objects
 *
 * Returns:
 *  - @ref AclSuccess if function was completed successfully
 *  - @ref AclOutOfMemory if there was a failure allocating memory resources
 *  - @ref AclUnsupportedTarget if the requested target is unsupported
 *  - @ref AclInvalidArgument if a given argument is invalid
 */
AclStatus AclCreateTensor(AclTensor *tensor, AclContext ctx, const AclTensorDescriptor *desc, bool allocate);

/** Map a tensor's backing memory to the host
 *
 * @param[in]      tensor Tensor to be mapped
 * @param[in, out] handle A handle to the underlying backing memory
 *
 * @return Status code
 *
 * Returns:
 *  - @ref AclSuccess if function was completed successfully
 *  - @ref AclInvalidArgument if a given argument is invalid
 */
AclStatus AclMapTensor(AclTensor tensor, void **handle);

/** Unmap the tensor's backing memory
 *
 * @param[in] tensor tensor to unmap memory from
 * @param[in] handle Backing memory to be unmapped
 *
 * @return Status code
 *
 * Returns:
 *  - @ref AclSuccess if function was completed successfully
 *  - @ref AclInvalidArgument if a given argument is invalid
 */
AclStatus AclUnmapTensor(AclTensor tensor, void *handle);

/** Import external memory to a given tensor object
 *
 * @param[in, out] tensor Tensor to import memory to
 * @param[in]      handle Backing memory to be imported
 * @param[in]      type   Type of the imported memory
 *
 * Returns:
 *  - @ref AclSuccess if function was completed successfully
 *  - @ref AclInvalidArgument if a given argument is invalid
 */
AclStatus AclTensorImport(AclTensor tensor, void *handle, AclImportMemoryType type);

/** Destroy a given tensor object
 *
 * @param[in,out] tensor A valid tensor object to be destroyed
 *
 * @return Status code
 *
 * Returns:
 *  - @ref AclSuccess if function was completed successfully
 *  - @ref AclInvalidArgument if the provided tensor is invalid
 */
AclStatus AclDestroyTensor(AclTensor tensor);

/** Creates a tensor pack
 *
 * Tensor packs are used to create a collection of tensors that can be passed around for operator execution
 *
 * @param[in,out] pack A valid non-zero tensor pack object if no failures occur
 * @param[in]     ctx  Context to be used
 *
 * @return Status code
 *
 * Returns:
 *  - @ref AclSuccess if function was completed successfully
 *  - @ref AclOutOfMemory if there was a failure allocating memory resources
 *  - @ref AclInvalidArgument if a given argument is invalid
 */
AclStatus AclCreateTensorPack(AclTensorPack *pack, AclContext ctx);

/** Add a tensor to a tensor pack
 *
 * @param[in,out] pack    Pack to append a tensor to
 * @param[in]     tensor  Tensor to pack
 * @param[in]     slot_id Slot of the operator that the tensors corresponds to
 *
 * @return Status code
 *
 * Returns:
 *  - @ref AclSuccess if function was completed successfully
 *  - @ref AclOutOfMemory if there was a failure allocating memory resources
 *  - @ref AclInvalidArgument if a given argument is invalid
 */
AclStatus AclPackTensor(AclTensorPack pack, AclTensor tensor, int32_t slot_id);

/** A list of tensors to a tensor pack
 *
 * @param[in,out] pack        Pack to append the tensors to
 * @param[in]     tensors     Tensors to append to the pack
 * @param[in]     slot_ids    Slot IDs of each tensors to the operators
 * @param[in]     num_tensors Number of tensors that are passed
 *
 * @return Status code
 *
 * Returns:
 *  - @ref AclSuccess if function was completed successfully
 *  - @ref AclOutOfMemory if there was a failure allocating memory resources
 *  - @ref AclInvalidArgument if a given argument is invalid
 */
AclStatus AclPackTensors(AclTensorPack pack, AclTensor *tensors, int32_t *slot_ids, size_t num_tensors);

/** Destroy a given tensor pack object
 *
 * @param[in,out] pack A valid tensor pack object to destroy
 *
 * @return Status code
 *
 * Returns:
 *  - @ref AclSuccess if functions was completed successfully
 *  - @ref AclInvalidArgument if the provided context is invalid
 */
AclStatus AclDestroyTensorPack(AclTensorPack pack);

/** Eager execution of a given operator on a list of inputs and outputs
 *
 * @param[in]     op      Operator to execute
 * @param[in]     queue   Queue to schedule the operator on
 * @param[in,out] tensors A list of input and outputs tensors to execute the operator on
 *
 * @return Status Code
 *
 * Returns:
 *  - @ref AclSuccess if function was completed successfully
 *  - @ref AclOutOfMemory if there was a failure allocating memory resources
 *  - @ref AclUnsupportedTarget if the requested target is unsupported
 *  - @ref AclInvalidArgument if a given argument is invalid
 *  - @ref AclRuntimeError on any other runtime related error
 */
AclStatus AclRunOperator(AclOperator op, AclQueue queue, AclTensorPack tensors);

/** Destroy a given operator object
 *
 * @param[in,out] op A valid operator object to destroy
 *
 * @return Status code
 *
 * Returns:
 *  - @ref AclSuccess if functions was completed successfully
 *  - @ref AclInvalidArgument if the provided context is invalid
 */
AclStatus AclDestroyOperator(AclOperator op);
#ifdef __cplusplus
}
#endif /* __cplusplus */
#endif /* ARM_COMPUTE_ACL_ENTRYPOINTS_H_ */
