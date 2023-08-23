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

#ifndef ACL_SRC_DYNAMIC_FUSION_SKETCH_GPU_CKW_DRIVER_COMPONENTS_GPUCKWDIRECTCONV2D
#define ACL_SRC_DYNAMIC_FUSION_SKETCH_GPU_CKW_DRIVER_COMPONENTS_GPUCKWDIRECTCONV2D

#include "arm_compute/dynamic_fusion/sketch/attributes/Conv2dAttributes.h"

#include "src/core/common/Macros.h"
#include "src/dynamic_fusion/sketch/gpu/ckw_driver/IGpuCkwComponentDriver.h"
#include "src/dynamic_fusion/sketch/gpu/components/cl/ClComponentDirectConv2d.h"

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
class GpuCkwDirectConv2d final : public IGpuCkwComponentDriver
{
public:
    using Attributes = ClComponentDirectConv2d::Attributes;
    using Settings   = ClComponentDirectConv2d::Settings;

public:
    /** Constructor
     *
     * For supported configurations please refer to @ref ClComponentDirectConv2d::validate()
     *
     * @param[in] id         Component id
     * @param[in] tensors    Tensor arguments to the component
     * @param[in] attributes Component attributes. Attributes are a set of parameters that define what a component does
     * @param[in] settings   Component settings. Settings are a set of parameters that influence the implementation of a component
     */
    GpuCkwDirectConv2d(ComponentId                      id,
                       const ArgumentPack<ITensorInfo> &tensors,
                       const Attributes                &attributes,
                       const Settings                  &settings);

    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(GpuCkwDirectConv2d);

    /** Destructor */
    ~GpuCkwDirectConv2d() override = default;

    // Inherited methods overriden
    virtual void write_component_code(const ComponentGroup    &comp_group,
                                      GpuCkwVariableTable     &vtable,
                                      GpuCkwScopedKernelWriter writer) const override;
    Window       get_window() const override;
    std::string  get_name(const ComponentGroup &comp_group) const override;

private:
    const ITensorInfo *_src;
    const ITensorInfo *_wei;
    const ITensorInfo *_bia;
    const ITensorInfo *_dst;

    const Attributes _attributes;
    const Settings   _settings;
};
} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute

#endif /* ACL_SRC_DYNAMIC_FUSION_SKETCH_GPU_CKW_DRIVER_COMPONENTS_GPUCKWDIRECTCONV2D */
