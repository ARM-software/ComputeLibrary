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
#ifndef SRC_GPU_CLCONTEXT_H
#define SRC_GPU_CLCONTEXT_H

#include "src/common/IContext.h"
#include "src/runtime/CL/mlgo/MLGOHeuristics.h"

#include "arm_compute/core/CL/OpenCL.h"

namespace arm_compute
{
namespace gpu
{
namespace opencl
{
/** OpenCL context implementation class */
class ClContext final : public IContext
{
public:
    /** Default Constructor
     *
     * @param[in] options Creational options
     */
    explicit ClContext(const AclContextOptions *options);

    /** Extract MLGO heuristics
     *
     * @return Heuristics tree
     */
    const mlgo::MLGOHeuristics &mlgo() const;

    /** Underlying cl context accessor
     *
     * @return the cl context used
     */
    ::cl::Context cl_ctx();

    /** Underlying cl device accessor
     *
     * @return the cl device used
     */
    ::cl::Device cl_dev();

    /** Update/inject an underlying cl context object
     *
     * @warning Context will be able to set if the object doesn't have any pending reference to other objects
     *
     * @param[in] ctx Underlying cl context to be used
     *
     * @return true if the context was set successfully else falseS
     */
    bool set_cl_ctx(::cl::Context ctx);

    // Inherrited methods overridden
    ITensorV2 *create_tensor(const AclTensorDescriptor &desc, bool allocate) override;
    IQueue *create_queue(const AclQueueOptions *options) override;
    std::tuple<IOperator *, StatusCode> create_activation(const AclTensorDescriptor &src,
                                                          const AclTensorDescriptor     &dst,
                                                          const AclActivationDescriptor &act,
                                                          bool                           is_validate) override;

private:
    mlgo::MLGOHeuristics _mlgo_heuristics;
    ::cl::Context        _cl_ctx;
    ::cl::Device         _cl_dev;
};
} // namespace opencl
} // namespace gpu
} // namespace arm_compute

#endif /* SRC_GPU_CLCONTEXT_H */