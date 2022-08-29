/*
 * Copyright (c) 2022 Arm Limited.
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
#ifndef SRC_DYNAMIC_FUSION_SKETCH_GPU_COMPONENTS_IGPUKERNELCOMPONENT
#define SRC_DYNAMIC_FUSION_SKETCH_GPU_COMPONENTS_IGPUKERNELCOMPONENT

#include "Types.h"

#include "src/dynamic_fusion/sketch/ArgumentPack.h"
#include "src/dynamic_fusion/sketch/gpu/GpuWorkloadSourceCode.h"

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
/** Properties common to all kernel component types */
class KernelProperties
{
public:
    KernelProperties &stage(const UnitWorkloadStage &stage)
    {
        _stage = stage;
        return *this;
    }
    UnitWorkloadStage stage() const
    {
        return _stage;
    }

private:
    UnitWorkloadStage _stage{};
};

inline bool operator==(const KernelProperties &config0, const KernelProperties &config1)
{
    return config0.stage() == config1.stage();
}

/** Forward declaration */
class IGpuTemplateComponentWriter;

/** An abstract interface of a component. It enables manipulation by the component graph for purposes like fusion
 */
class IGpuKernelComponent
{
public:
    using Properties = KernelProperties;

public:
    /** Constructor
     *
     * @param[in] id         Component id
     * @param[in] properties Kernel component properties
     * @param[in] tensors    Tensor arguments to the components
     */
    IGpuKernelComponent(
        ComponentId                      id,
        const Properties                &properties,
        const ArgumentPack<ITensorInfo> &tensors)
        : _id{ id },
          _properties{ properties },
          _tensors{ tensors }
    {
    }
    /** Destructor */
    virtual ~IGpuKernelComponent()
    {
    }
    /** Get component id */
    ComponentId id() const
    {
        return _id;
    }
    /** Get tensor arguments */
    ArgumentPack<ITensorInfo> tensors() const
    {
        return _tensors;
    }
    /** Get properties */
    Properties properties() const
    {
        return _properties;
    }
    /** Get template writer for the component */
    virtual const IGpuTemplateComponentWriter *template_writer() const = 0;
    /** Get component type */
    virtual GpuComponentType type() const = 0;

private:
    ComponentId               _id{ -1 };
    Properties                _properties{};
    ArgumentPack<ITensorInfo> _tensors{};
};
} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
#endif /* SRC_DYNAMIC_FUSION_SKETCH_GPU_COMPONENTS_IGPUKERNELCOMPONENT */
