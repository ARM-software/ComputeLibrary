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
#if defined(ENABLE_EXPERIMENTAL_DYNAMIC_FUSION)

#ifndef ARM_COMPUTE_EXPERIMENTAL_CLKERNELBUILDINGAPI_H
#define ARM_COMPUTE_EXPERIMENTAL_CLKERNELBUILDINGAPI_H

#include "arm_compute/core/CL/CLCompileContext.h"
#include "arm_compute/core/Window.h"

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
using ArgumentID = int32_t;

static constexpr ArgumentID g_arg_placeholder = -1;

/** Verbose and explicit way to enumerate all the tensor arguments variants used by
 *  all kernel implementations. This avoids any ambiguity in what kernel arguments are passed
 */
enum class TensorArgType : int
{
    Scalar,

    Vector,

    Image,
    Image_Reinterpret_As_3D,
    Image_Export_To_ClImage2D,

    Image_3D, // 3D Tensor represented as a 2D Image + stride_z
    Image_3D_Export_To_ClImage2D,

    Tensor_3D,
    Tensor_4D,

    Tensor_4D_t_Buffer,
    Tensor_4D_t_Image
};
/** Describes all the info required to add a kernel argument at run time */
struct ClKernelArgRuntimeDescriptor
{
    ClKernelArgRuntimeDescriptor(int arg_id, TensorArgType type, bool slide_along_dimz = true)
        : arg_id{ arg_id }, tensor_arg_type{ type }, slide_along_dimz{ slide_along_dimz }
    {
    }
    ~ClKernelArgRuntimeDescriptor() = default;
    int           arg_id{ g_arg_placeholder }; // Arg ID in the blueprint
    TensorArgType tensor_arg_type{ TensorArgType::Image };
    bool          slide_along_dimz{ true };
};

using ClKernelArgList = std::vector<ClKernelArgRuntimeDescriptor>;

/** Intermediate representation of the final, complete kernel source. */
class ClKernelBlueprint
{
public:
    ClKernelBlueprint();
    ~ClKernelBlueprint();

private:
    struct Implementation;
    std::unique_ptr<Implementation> _impl;

public:
    Implementation       &impl();
    const Implementation &impl() const;
};

///// Kernel Components /////

/** Meta information about all Cl Kernel Components */
struct ClKernelComponentDescriptor
{
    int32_t version{ 1 }; /**< Operator version */
};

/** Component: Tensor Argument */
struct ClTensorDescriptor
{
    ClTensorDescriptor(ITensorInfo *info)
        : tensor_info(info)
    {
    }

    ITensorInfo *tensor_info;
};

Status add_tensor_argument(ClKernelBlueprint &, const ClTensorDescriptor &, ArgumentID &);
Status add_tensor_intermed(ClKernelBlueprint &, ArgumentID &);

/** Component: Gemm Native */
struct GemmNativeDescriptor
{
    float             alpha{};
    float             beta{};
    unsigned int      m{};
    unsigned int      n{};
    unsigned int      k{};
    unsigned int      depth_output_gemm3d{};
    bool              reinterpret_input_as_3d{};
    bool              broadcast_bias{};
    bool              fp_mixed_precision{};
    bool              has_pad_y{};
    int               nmult_transpose1xW_width{};
    int               mult_interleave4x4_height{};
    GEMMLHSMatrixInfo lhs_info{};
    GEMMRHSMatrixInfo rhs_info{};
    int32_t           a_offset{};
    int32_t           b_offset{};
};

Status add_kcomp_gemm_native(ClKernelBlueprint &, const ClKernelComponentDescriptor &, const GemmNativeDescriptor &,
                             ArgumentID lhs_id, ArgumentID rhs_id, ArgumentID bias_id, ArgumentID &dst_id);

/** Component: Eltwise Add */
struct EltwiseAddDescriptor
{
    ConvertPolicy convert_policy{ ConvertPolicy::SATURATE };
};
Status add_kcomp_eltwise_add(ClKernelBlueprint &, const ClKernelComponentDescriptor &, const EltwiseAddDescriptor &, ArgumentID src0_id,
                             ArgumentID src1_id, ArgumentID &dst_id);

/** Component: Activation */
struct ActivationDescriptor
{
};
Status add_kcomp_activation(ClKernelBlueprint &, const ClKernelComponentDescriptor &, const ActivationDescriptor &, ArgumentID src_id, ArgumentID &dst_id);

/** Component: Direct Convolution **/
struct DirectConvolutionDescriptor
{
    PadStrideInfo pad_stride_info{};
};
Status add_kcomp_direct_conv(ClKernelBlueprint &, const ClKernelComponentDescriptor &, const DirectConvolutionDescriptor &,
                             ArgumentID src_id, ArgumentID weight_id, ArgumentID bias_id, ArgumentID &dst_id);

enum class ClippingStrategy
{
    TOP_LEFT,
    TOP_RIGHT,
    BOTTOM_LEFT,
    BOTTOM_RIGHT,
};

/** Component: Store */
struct TileDescriptor
{
    Size2D           tile_dims{};
    Size2D           boundaries{};
    ClippingStrategy clipping{ ClippingStrategy::TOP_LEFT };

    TileDescriptor()
    {
    }

    TileDescriptor(Size2D dims, const Size2D &bound, const ClippingStrategy &clip)
        : tile_dims(dims), boundaries(bound), clipping(clip)
    {
    }

    bool empty() const
    {
        return (tile_dims.area() == 0) || (boundaries.area() == 0);
    }
};

enum class StoreType
{
    VStore,
    VStorePartial,
    StoreRow,
    ConvertStoreRow,
    StoreBlock,
    ConvertStoreBlock,
    StoreRowPartial,
    StoreBlockPartial,
    StoreBlockBoundaryAware,
    StoreVectorSelect,
    TStoreIndirectWidthSelect
};

Status add_kcomp_store(ClKernelBlueprint &, const ClKernelComponentDescriptor &, ArgumentID src_id, ArgumentID dst_id, const StoreType &store_type);

///// Kernel Components /////

///// Building /////

/** Information required for kernel compilation. The build results of KernelBlueprint */
struct ClKernelCode
{
    std::string     name{};          /**< Kernel name */
    std::string     code{};          /**< Kernel source code */
    std::string     config_id{};     /**< Generated from blueprint based on complex component */
    CLBuildOptions  build_options{}; /**< Kernel build options */
    Window          window{};        /**< Execution window */
    ClKernelArgList arguments{};     /**< Kernel argument specficiations */

    bool operator==(const ClKernelCode &other) const
    {
        return name == other.name && code == other.code && build_options == other.build_options;
    }
};

/** GPU information for building the @ref ClKernelCode */
struct GpuInfo
{
    GPUTarget target{ GPUTarget::UNKNOWN };
};

/** All information required for building the @ref ClKernelCode */
struct ClCodeBuilderContext
{
    GpuInfo gpu_info{};
};

Status set_tile_info(ClKernelBlueprint &, const TileDescriptor &);

/** Build final kernel source from KernelBlueprint */
Status build(ClKernelCode &code, const ClCodeBuilderContext &, ClKernelBlueprint &);

///// Building /////

///// Tuning /////
struct ClExecutionDescriptor
{
    cl::NDRange suggested_lws{};              /**< Suggested local work-group size for optimal performance if not zero */
    cl::NDRange gws{};                        /**< Global work-group to be used */
    bool        skip_sliding_window{ false }; /**< Skip sliding window slices during execution loop */
};

Status tune_static(ClExecutionDescriptor &, const ClKernelCode &);

///// Tuning /////

} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
#endif //ARM_COMPUTE_EXPERIMENTAL_CLKERNELBUILDINGAPI_H

#endif // defined(ENABLE_EXPERIMENTAL_DYNAMIC_FUSION)