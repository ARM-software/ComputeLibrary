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
#ifndef SRC_DYNAMIC_FUSION_SKETCH_GPU_COMPONENTS_CL_CLCOMPONENTDEPTHWISECONV2D
#define SRC_DYNAMIC_FUSION_SKETCH_GPU_COMPONENTS_CL_CLCOMPONENTDEPTHWISECONV2D

#include "arm_compute/core/Error.h"
#include "src/dynamic_fusion/sketch/gpu/components/IGpuKernelComponent.h"
#include <memory>

namespace arm_compute
{
/** Forward declaration */
class ITensorInfo;
namespace experimental
{
namespace dynamic_fusion
{
/** Forward declaration */
template <typename T>
class ArgumentPack;
class DepthwiseConv2dAttributes;

/** Component specific settings
 */
class ClComponentDepthwiseConv2dSettings
{
public:
    /** Set export_input_to_cl_image flag */
    ClComponentDepthwiseConv2dSettings &export_input_to_cl_image(bool cl_image);
    /** Get export_input_to_cl_image flag */
    bool export_input_to_cl_image() const;

    /** Set export_weights_to_cl_image flag */
    ClComponentDepthwiseConv2dSettings &export_weights_to_cl_image(bool cl_image);
    /** Get export_weights_to_cl_image flag */
    bool export_weights_to_cl_image() const;

    /** Set fast_relaxed_math flag */
    ClComponentDepthwiseConv2dSettings &fast_relaxed_math(bool fast_relaxed_math);
    /** Get fast_relaxed_math flag */
    bool fast_relaxed_math() const;

    /** Set is_fma_available flag */
    ClComponentDepthwiseConv2dSettings &is_fma_available(bool is_fma_available);
    /** Get is_fma_available flag */
    bool is_fma_available() const;

    /** Set N0: number of columns processed by each thread */
    ClComponentDepthwiseConv2dSettings &n0(unsigned int n0);
    /** Get N0: number of columns processed by each thread */
    unsigned int n0() const;

    /** Set M0: number of rows processed by each thread */
    ClComponentDepthwiseConv2dSettings &m0(unsigned int m0);
    /** Set M0: number of rows processed by each thread */
    unsigned int m0() const;

private:
    bool         _export_input_to_cl_image{ false };   /**< Export input to cl_image */
    bool         _export_weights_to_cl_image{ false }; /**< Export the weights to cl_image */
    bool         _fast_relaxed_math{ true };           /**< Enable/disable -cl-fast-relaxed-math flag */
    bool         _is_fma_available{ false };           /**< Is fma instruction available */
    unsigned int _n0{ 0 };                             /**< Number of columns processed by each thread */
    unsigned int _m0{ 0 };                             /**< Number of rows processed by each thread */
};

/** Forward declaration */
class ClTemplateDepthwiseConv2d;

class ClComponentDepthwiseConv2d final : public IGpuKernelComponent
{
public:
    /** Attributes are a set of backend-agnostic parameters that define what a component does */
    using Attributes = DepthwiseConv2dAttributes;
    /** Settings are a set of backend-specific parameters that influence the implementation of a component */
    using Settings = ClComponentDepthwiseConv2dSettings;

public:
    /** Validate the component
     *
     * @param[in]     properties Component properties @ref Properties
     * @param[in,out] tensors    Tensor arguments to the component
     * @param[in]     attributes Component attributes @ref Attributes
     * @param[in]     settings   Component settings @ref Settings
     *
     * @return Status       Validation results
     *
     * Tensor argument names:
     * - ACL_SRC_0: Input
     * - ACL_SRC_1: Weight
     * - ACL_SRC_2: Bias (Optional)
     * - ACL_DST_0: Output
     *
     * Tensor argument constness:
     * - ACL_SRC_0: Const
     * - ACL_SRC_1: Const
     * - ACL_SRC_2: Const
     * - ACL_DST_0: Const
     *
     * Valid data layouts:
     * - NHWC
     *
     * Valid data type configurations:
     * |ACL_SRC_0      |ACL_SRC_1      |ACL_SRC_2      |ACL_DST_0      |
     * |:--------------|:--------------|:--------------|:--------------|
     * |F16            |F16            |F16            |F16            |
     * |F32            |F32            |F32            |F32            |
     */
    static Status validate(
        const Properties                &properties,
        const ArgumentPack<ITensorInfo> &tensors,
        const Attributes                &attributes,
        const Settings                  &settings);

    /** Constructor
     *
     * Similar to @ref ClComponentDepthwiseConv2d::validate()
     */
    ClComponentDepthwiseConv2d(
        ComponentId                      id,
        const Properties                &properties,
        const ArgumentPack<ITensorInfo> &tensors,
        const Attributes                &attributes,
        const Settings                  &settings);

    /** Destructor */
    ~ClComponentDepthwiseConv2d() override;
    /** Prevent instances of this class from being copy constructed */
    ClComponentDepthwiseConv2d(const ClComponentDepthwiseConv2d &component) = delete;
    /** Prevent instances of this class from being copied */
    ClComponentDepthwiseConv2d &operator=(const ClComponentDepthwiseConv2d &component) = delete;
    /** Allow instances of this class to be move constructed */
    ClComponentDepthwiseConv2d(ClComponentDepthwiseConv2d &&component) = default;
    /** Allow instances of this class to be moved */
    ClComponentDepthwiseConv2d &operator=(ClComponentDepthwiseConv2d &&component) = default;
    /** Get template writer for the component */
    const IGpuTemplateComponentWriter *template_writer() const override;
    /** Get component type */
    GpuComponentType type() const override
    {
        return GpuComponentType::Complex;
    }

private:
    std::unique_ptr<ClTemplateDepthwiseConv2d> _component_writer;
};
} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
#endif /* SRC_DYNAMIC_FUSION_SKETCH_GPU_COMPONENTS_CL_CLCOMPONENTDEPTHWISECONV2D */
