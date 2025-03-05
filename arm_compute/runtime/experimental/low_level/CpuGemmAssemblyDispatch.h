/*
 * Copyright (c) 2018-2025 Arm Limited.
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

#ifndef ACL_ARM_COMPUTE_RUNTIME_EXPERIMENTAL_LOW_LEVEL_CPUGEMMASSEMBLYDISPATCH_H
#define ACL_ARM_COMPUTE_RUNTIME_EXPERIMENTAL_LOW_LEVEL_CPUGEMMASSEMBLYDISPATCH_H

#include "arm_compute/core/ITensorPack.h"
#include "arm_compute/function_info/GEMMInfo.h"
#include "arm_compute/runtime/IOperator.h"

/*
* A shallow wrapper class to expose CpuGemmAssemblyDispatch.
* New functionality should be added to src/cpu/operators/internal/CpuGemmAssemblyDispatch.h.
*/

namespace arm_compute
{
namespace experimental
{
namespace op
{
namespace ll
{
/** Wrapper class for CpuGemmAssemblyDispatch. For information on the functions,
 * see "src/cpu/operators/CpuGemmAssemblyDispatch.h".
 *
 * Following fields will be ignored if passed in through GEMMInfo in configure()
 * and has_opt_impl(). If these fields are set incorrectly, validate() will
 * return false:
 * GEMMInfo.method, GEMMInfo.reinterpret_input_as_3d, GEMMInfo.depth_output_gemm3d,
 * GEMMInfo.output_stage, GEMMInfo.reshape_b_only_on_first_run
*/
class CpuGemmAssemblyDispatch : arm_compute::experimental::IOperator
{
public:
    /** Constructor **/
    CpuGemmAssemblyDispatch();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CpuGemmAssemblyDispatch(const CpuGemmAssemblyDispatch &) = delete;
    /** Default move constructor */
    CpuGemmAssemblyDispatch(CpuGemmAssemblyDispatch &&) = default;
    /** Default destructor */
    ~CpuGemmAssemblyDispatch();

    /** If supported create a Compute Library function else fallback to the arm_gemm function.
     *
     * @note Configuring "batches"
     * The shapes of @p a @p b and @p d are arranged as follows:
     *     Lowest dimension <-> Highest dimension
     * a: [K, M, Batch, Multi]
     * b: [N, K, Multi]
     * d: [N, M, Batch, Multi]
     *
     * The "Batch" refers to where "Batch" number of MxK slices of tensor a multiplies with a single KxN slice of b
     * The "Multi" refers to where "Multi" number of individual multiplication of a with b
     *
     * E.g. the following are some example input shape configurations
     *
     * (1) Normal 2D gemm
     * a: [K=3, M=4]
     * b: [N=5, K=3]
     * d: [N=5, M=4]
     *
     * (2) Batches of a sharing b (e.g. gemm-based batched convolution where b is the shared )
     * a: [K=3, M=4, Batch=9]
     * b: [N=5, K=3]
     * d: [N=5, M=4, Batch=9]
     *
     * (3) "Batches" of independent gemm (e.g. batched matmul)
     * a: [K=3, M=4, Batch=1, Multi=7]
     * b: [N=5, K=3, Multi=7]
     * d: [N=5, M=4, Batch=1, Multi=7]
     *
     * (4) "Batches" of independent gemm where b is also shared
     * a: [K=3, M=4, Batch=4, Multi=7]
     * b: [N=5, K=3, Multi=7]
     * d: [N=5, M=4, Batch=4, Multi=7]
     *
     * @param[in]  a         Input tensor (Matrix A)
     * @param[in]  b         Input tensor (Matrix B)
     * @param[in]  c         Input tensor (Matrix C) used to pass the bias for quantized calculations
     * @param[out] d         Output tensor to store the result of matrix multiplication. Data type supported: same as @p input0.
     * @param[in]  gemm_info GEMM meta-data
     */
    void configure(const ITensorInfo *a,
                   const ITensorInfo *b,
                   const ITensorInfo *c,
                   ITensorInfo       *d,
                   const GEMMInfo    &gemm_info = GEMMInfo());

    /** Indicates whether or not this function can be used to process the given parameters.
     * Valid data type configurations:
     * |src0         |src1        |src2      |dst            |
     * |:------------|:-----------|:---------|:--------------|
     * |F32          |F32         |nullptr   |F32            |
     * |F16          |F16         |nullptr   |F16            |
     * |BFLOAT16     |BFLOAT16    |nullptr   |BFLOAT16       |
     * |BFLOAT16     |BFLOAT16    |nullptr   |BFLOAT32       |
     *
     * @param[in] a         Input tensor info (Matrix A)
     * @param[in] b         Input tensor info (Matrix B)
     * @param[in] c         Input tensor info (Matrix C) used to pass the bias for quantized calculations
     * @param[in] d         Output tensor to store the result of matrix multiplication. Data type supported: same as @p input0.
     * @param[in] gemm_info GEMM meta-data
     *
     * @return a status.
     */
    static Status validate(const ITensorInfo *a,
                           const ITensorInfo *b,
                           const ITensorInfo *c,
                           const ITensorInfo *d,
                           const GEMMInfo    &gemm_info = GEMMInfo());

    /** Indicates whether or not there is an optimal assembly implementation that can be used to process the given parameters.
     *
     * This method has the same use of @ref
     * NEGEMMConvolutionLayer::has_opt_impl, with the only caveat that
     * the value of arm_compute::WeightFormat need to be passed via the
     * parameter info.
     *
     * @return a status.
     */
    static Status has_opt_impl(arm_compute::WeightFormat &weight_format,
                               const ITensorInfo         *a,
                               const ITensorInfo         *b,
                               const ITensorInfo         *c,
                               const ITensorInfo         *d,
                               const GEMMInfo            &gemm_info = GEMMInfo());

    /** Indicates whether or not there is a implementation for the configured GEMM
     *
     * @deprecated All fixed-format kernels are now stateless.
     * For now this function will always return true, but it will be removed
     * completely in a future release.
     *
     * @return a bool: true if the implementation is stateless; false if not.
     */
    bool has_stateless_impl() const;

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

    // Inherited methods overridden:
    void                             run(ITensorPack &tensors);
    void                             prepare(ITensorPack &constants);
    experimental::MemoryRequirements workspace() const;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};
} // namespace ll
} // namespace op
} // namespace experimental
} // namespace arm_compute

#endif // ACL_ARM_COMPUTE_RUNTIME_EXPERIMENTAL_LOW_LEVEL_CPUGEMMASSEMBLYDISPATCH_H
