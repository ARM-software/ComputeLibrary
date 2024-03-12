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
#ifndef ACL_SRC_DYNAMIC_FUSION_SKETCH_GPU_CKW_DRIVER_COMPONENTS_GPUCKWSTORE
#define ACL_SRC_DYNAMIC_FUSION_SKETCH_GPU_CKW_DRIVER_COMPONENTS_GPUCKWSTORE

#include "src/core/common/Macros.h"
#include "src/dynamic_fusion/sketch/gpu/ckw_driver/IGpuCkwComponentDriver.h"

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
/** An interface used by @ref ClTemplateWriter to write source code for a kernel component
 */
class GpuCkwStore : public IGpuCkwComponentDriver
{
public:
    /** Constructor
     *
     * @param[in] id      Component id
     * @param[in] tensors Tensor arguments to the component
     */
    GpuCkwStore(ComponentId id, const ArgumentPack<ITensorInfo> &tensors);
    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(GpuCkwStore);
    /** Destructor */
    ~GpuCkwStore() override = default;
    // Inherited methods overriden:
    virtual void write_component_code(const ComponentGroup    &comp_group,
                                      GpuCkwVariableTable     &vtable,
                                      GpuCkwScopedKernelWriter writer) const override;
    std::string  get_name(const ComponentGroup &comp_group) const override;

private:
    const ITensorInfo *_src;
    const ITensorInfo *_dst;
};
} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute

#endif /* ACL_SRC_DYNAMIC_FUSION_SKETCH_GPU_CKW_DRIVER_COMPONENTS_GPUCKWSTORE */
