/*
 * Copyright (c) 2023 Arm Limited.
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

#ifndef ACL_SRC_DYNAMIC_FUSION_SKETCH_GPU_GPUWORKLOADCONTEXTIMPL_H
#define ACL_SRC_DYNAMIC_FUSION_SKETCH_GPU_GPUWORKLOADCONTEXTIMPL_H

#include "arm_compute/core/CL/CLCompileContext.h"
#include "arm_compute/core/ITensorInfo.h"
#include "arm_compute/dynamic_fusion/sketch/MemoryDescriptor.h"
#include "arm_compute/dynamic_fusion/sketch/gpu/GpuWorkloadContext.h"

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
/** Internal implementation of workload context. */
class GpuWorkloadContext::Impl
{
public:
    /** Constructor
     *
     * @param[in] gpu_language   Target GPU language.
     * @param[in] cl_compile_ctx CL compile context.
     */
    Impl(GpuLanguage gpu_language, CLCompileContext *cl_compile_ctx);

    /** Copy constructor */
    Impl(Impl &) = default;

    /** Assignment */
    Impl &operator=(Impl &) = default;

    /** Get target GPU language. */
    GpuLanguage gpu_language() const;

    /** Get CL compile context. */
    const CLCompileContext *cl_compile_context() const;

    /** Get memory descriptor registry. */
    const MemoryDescriptorMap &mem_map() const;

    /** Set a new ID and register the user tensor info.
     *
     * @param[in, out] tensor_info The tensor info to be registered.
     */
    void register_user_tensor(ITensorInfo &tensor_info);

    /** Create a virtual (see @ref MemoryType) tensor info and save it
     *
     * @return ITensorInfo*  The created virtual tensor info object pointer
     */
    ITensorInfo *create_virtual_tensor();
    /** Create an auxiliary (see @ref MemoryType) tensor info and save it
     *
     * @param[in] tensor_info @ref ITensorInfo to copy from
     *
     * @return ITensorInfo*  The created auxiliary tensor info object pointer
     */
    ITensorInfo *create_auxiliary_tensor(const ITensorInfo &tensor_info);

    /** Get tensor info created by this context, from id */
    ITensorInfo *get_tensor_info(ITensorInfo::Id id);

    /** Get tensor info created by this context, from id */
    const ITensorInfo *get_tensor_info(ITensorInfo::Id id) const;

private:
    ITensorInfo::Id next_tensor_id();

    GpuLanguage       _gpu_language;
    CLCompileContext *_cl_compile_ctx;

    ITensorInfo::Id     _next_tensor_id;
    MemoryDescriptorMap _mem_map;
    std::map<ITensorInfo::Id, std::unique_ptr<TensorInfo>> _managed_tensor_info;
};

} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute

#endif // ACL_SRC_DYNAMIC_FUSION_SKETCH_GPU_GPUWORKLOADCONTEXTIMPL_H
