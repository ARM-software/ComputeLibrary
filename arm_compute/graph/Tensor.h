/*
 * Copyright (c) 2018-2019 Arm Limited.
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
#ifndef ARM_COMPUTE_GRAPH_TENSOR_H
#define ARM_COMPUTE_GRAPH_TENSOR_H

#include "arm_compute/graph/Types.h"

#include "arm_compute/graph/ITensorAccessor.h"
#include "arm_compute/graph/ITensorHandle.h"
#include "arm_compute/graph/TensorDescriptor.h"

#include <memory>
#include <set>

namespace arm_compute
{
namespace graph
{
/** Tensor object **/
class Tensor final
{
public:
    /** Default constructor
     *
     * @param[in] id   Tensor ID
     * @param[in] desc Tensor information
     */
    Tensor(TensorID id, TensorDescriptor desc);
    /** Tensor ID accessor
     *
     * @return Tensor ID
     */
    TensorID id() const;
    /** TensorInfo metadata accessor
     *
     * @return Tensor descriptor metadata
     */
    TensorDescriptor &desc();
    /** TensorInfo metadata accessor
     *
     * @return Tensor descriptor metadata
     */
    const TensorDescriptor &desc() const;
    /** Sets the backend tensor
     *
     * @param[in] backend_tensor Backend tensor to set
     */
    void set_handle(std::unique_ptr<ITensorHandle> backend_tensor);
    /** Backend tensor handle accessor
     *
     * @return Backend tensor handle
     */
    ITensorHandle *handle();
    /** Sets the backend tensor accessor
     *
     * @param[in] accessor Accessor to set
     */
    void set_accessor(std::unique_ptr<ITensorAccessor> accessor);
    /** Backend tensor accessor
     *
     * @return Backend tensor accessor
     */
    ITensorAccessor *accessor();
    /** Extracts accessor from the tensor
     *
     * @warning Accessor gets unbound from the tensor
     *
     * @return The accessor of the tensor
     */
    std::unique_ptr<ITensorAccessor> extract_accessor();
    /** Calls accessor on tensor
     *
     * @return True if the accessor was called else false
     */
    bool call_accessor();
    /** Binds the tensor with an edge
     *
     * @param[in] eid Edge ID that is bound to the tensor
     */
    void bind_edge(EdgeID eid);
    /** Unbinds an edge from a tensor
     *
     * @param[in] eid Edge to unbind
     */
    void unbind_edge(EdgeID eid);
    /** Accessor the edges that are bound with the tensor
     *
     * @return Bound edges
     */
    std::set<EdgeID> bound_edges() const;

private:
    TensorID                         _id;          /**< Tensor id */
    TensorDescriptor                 _desc;        /**< Tensor metadata */
    std::unique_ptr<ITensorHandle>   _handle;      /**< Tensor Handle */
    std::unique_ptr<ITensorAccessor> _accessor;    /**< Tensor Accessor */
    std::set<EdgeID>                 _bound_edges; /**< Edges bound to this tensor */
};
} // namespace graph
} // namespace arm_compute
#endif /* ARM_COMPUTE_GRAPH_TENSOR_H */
