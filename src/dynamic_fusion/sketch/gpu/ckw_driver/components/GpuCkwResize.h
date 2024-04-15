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
#ifndef ACL_SRC_DYNAMIC_FUSION_SKETCH_GPU_CKW_DRIVER_COMPONENTS_GPUCKWRESIZE_H
#define ACL_SRC_DYNAMIC_FUSION_SKETCH_GPU_CKW_DRIVER_COMPONENTS_GPUCKWRESIZE_H

#include "src/core/common/Macros.h"
#include "src/dynamic_fusion/sketch/gpu/ckw_driver/IGpuCkwComponentDriver.h"
#include "src/dynamic_fusion/sketch/gpu/components/cl/ClComponentResize.h"

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
class GpuCkwResize final : public IGpuCkwComponentDriver
{
public:
    using Attributes = ClComponentResize::Attributes;

public:
    /** Constructor
     *
     * @param[in] id         Component id
     * @param[in] tensors    Tensor arguments to the components
     * @param[in] attributes Component attributes
     */
    GpuCkwResize(ComponentId id, const ArgumentPack<ITensorInfo> &tensors, const Attributes &attributes);

    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(GpuCkwResize);

    /** Destructor */
    ~GpuCkwResize() override = default;

    // Inherited methods overriden
    virtual void write_component_code(const ComponentGroup    &comp_group,
                                      GpuCkwVariableTable     &vtable,
                                      GpuCkwScopedKernelWriter writer) const override;
    Window       get_window() const override;
    std::string  get_name(const ComponentGroup &comp_group) const override;
    std::string  get_tuner_id(const ComponentGroup &comp_group) const override;

private:
    /** Resize using nearest neighbor interpolation
     *
     * @param[in]      comp_group Component group to which this component belongs to
     * @param[in, out] vtable     Table of variables declared by this component
     * @param[in, out] writer     CKW writer that writes code scoped to this kernel component
     */
    void do_nearest_neighbor_resize(const ComponentGroup    &comp_group,
                                    GpuCkwVariableTable     &vtable,
                                    GpuCkwScopedKernelWriter writer) const;

    /** Resize using bilinear interpolation
     *
     * @param[in]      comp_group Component group to which this component belongs to
     * @param[in, out] vtable     Table of variables declared by this component
     * @param[in, out] writer     CKW writer that writes code scoped to this kernel component
     */
    void do_bilinear_resize(const ComponentGroup    &comp_group,
                            GpuCkwVariableTable     &vtable,
                            GpuCkwScopedKernelWriter writer) const;

    const ITensorInfo *_src;
    const ITensorInfo *_dst;
    Attributes         _attributes;
};
} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute

#endif // ACL_SRC_DYNAMIC_FUSION_SKETCH_GPU_CKW_DRIVER_COMPONENTS_GPUCKWRESIZE_H
