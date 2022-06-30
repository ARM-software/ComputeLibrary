/*
 * Copyright (c) 2018-2022 Arm Limited.
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
#ifndef ARM_COMPUTE_CPU_INTERNAL_CPU_GEMM_ASSEMBLY_DISPATCH_H
#define ARM_COMPUTE_CPU_INTERNAL_CPU_GEMM_ASSEMBLY_DISPATCH_H

#include "src/core/common/Macros.h"
#include "src/cpu/ICpuOperator.h"

namespace arm_compute
{
namespace cpu
{
/* Convolution method supported by the assembly gemm interface */
enum class AsmConvMethod
{
    Im2Col,
    Indirect,
    Conv
};

struct AsmGemmInfo
{
    AsmConvMethod           method{ AsmConvMethod::Im2Col };
    PadStrideInfo           ps_info{};
    ActivationLayerInfo     activation_info{};
    GEMMLowpOutputStageInfo output_stage{};
    bool                    negated_offsets{ true };
    bool                    reinterpret_input_as_3d{ false };
    bool                    depth_output_gemm3d{ false };
    int64_t                 padding_top{ 0 };
    int64_t                 padding_left{ 0 };
    float                   padding_value{ 0.f };
    bool                    fast_mode{ false };
    bool                    fixed_format{ false };
    arm_gemm::WeightFormat  weight_format{ arm_gemm::WeightFormat::UNSPECIFIED };
};

/** Assembly kernel glue */
class CpuGemmAssemblyDispatch : public ICpuOperator
{
public:
    /** Constructor */
    CpuGemmAssemblyDispatch();
    /** Defautl destructor */
    ~CpuGemmAssemblyDispatch() = default;

    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(CpuGemmAssemblyDispatch);

    class IFallback
    {
    public:
        virtual void                             run(ITensorPack &tensors)     = 0;
        virtual void                             prepare(ITensorPack &tensors) = 0;
        virtual experimental::MemoryRequirements workspace() const             = 0;
        virtual bool                             is_configured() const         = 0;
        virtual bool                             isVarWeightsKernel() const    = 0;
        virtual ~IFallback()                                                   = default;
    };

public:
    /** If supported create a Compute Library function else fallback to the arm_gemm function.
     *
     * @param[in]  a    Input tensor (Matrix A)
     * @param[in]  b    Input tensor (Matrix B)
     * @param[in]  c    Input tensor (Matrix C) used to pass the bias for quantized calculations
     * @param[out] d    Output tensor to store the result of matrix multiplication. Data type supported: same as @p input0.
     * @param[in]  info GEMM meta-data
     */
    void configure(const ITensorInfo *a, const ITensorInfo *b, const ITensorInfo *c, ITensorInfo *d, const AsmGemmInfo &info);

    /** Indicates whether or not this function can be used to process the given parameters.
     *
     * @param[in] a    Input tensor info (Matrix A)
     * @param[in] b    Input tensor info (Matrix B)
     * @param[in] c    Input tensor info (Matrix C) used to pass the bias for quantized calculations
     * @param[in] d    Output tensor to store the result of matrix multiplication. Data type supported: same as @p input0.
     * @param[in] info GEMM meta-data
     *
     * @return a status.
     */
    static Status validate(const ITensorInfo *a, const ITensorInfo *b, const ITensorInfo *c, const ITensorInfo *d, const AsmGemmInfo &info);

    /** Indicates whether or not there is an optimal assembly implementation that can be used to process the given parameters.
     *
     * This method has the same use of @ref
     * NEGEMMConvolutionLayer::has_opt_impl, with the only caveat that
     * the value of arm_gemm::WeightFormat need to be passed via the
     * parameter info.
     *
     * @return a status.
     */
    static Status has_opt_impl(arm_gemm::WeightFormat &weight_format, const ITensorInfo *a, const ITensorInfo *b, const ITensorInfo *c, const ITensorInfo *d, const AsmGemmInfo &info);
    /** Checks if activation is supported by the gemm assembly dispatcher
     *
     * @param[in] activation Activation to check
     *
     * @return True if activation is supported else false
     */
    static bool is_activation_supported(const ActivationLayerInfo &activation);
    /** Was the function successfully configured ?
     *
     * @return True if the function is configured and ready to run
     */
    bool is_configured() const;
    /** Indicates if the convolution executes in variable weights mode.
     *
     * Similar to @ref CpuGemm::isVarWeightsKernel
     */
    bool isVarWeightsKernel() const
    {
        return _arm_gemm && _arm_gemm->isVarWeightsKernel();
    }

    // Inherited methods overridden:
    void                             prepare(ITensorPack &tensors) override;
    void                             run(ITensorPack &tensors) override;
    experimental::MemoryRequirements workspace() const override;

private:
    std::unique_ptr<IFallback> _arm_gemm; /**< Interface for the arm_gemm fallback */
};
} // namespace cpu
} // namespace arm_compute
#endif /* ARM_COMPUTE_CPU_INTERNAL_CPU_GEMM_ASSEMBLY_DISPATCH_H */
