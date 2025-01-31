/*
 * Copyright (c) 2024-2025 Arm Limited.
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

#ifndef ACL_SRC_CPU_KERNELS_DYNAMIC_GEMM_HEURISTICS_CPUDYNAMICGEMMKERNELHEURISTICS_H
#define ACL_SRC_CPU_KERNELS_DYNAMIC_GEMM_HEURISTICS_CPUDYNAMICGEMMKERNELHEURISTICS_H

#include "arm_compute/core/CPP/ICPPKernel.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/ITensorInfo.h"
#include "arm_compute/core/Window.h"
#include "arm_compute/runtime/IScheduler.h"

#include "src/core/common/Macros.h"
#include "src/cpu/kernels/CpuKernelSelectionTypes.h"

#include <map>
#include <vector>

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
namespace heuristics
{

class CpuDynamicGemmKernelHeuristics
{
public:
    /** Run the micro-kernel
     *
     * @param[in] a      Tensor a
     * @param[in] b      Tensor b
     * @param[in] c      Tensor c
     * @param[in] d      Tensor d
     * @param[in] pack_b Packed tensor b
     * @param[in] window Window to run the kernel on
     */
    using KernelPtr = std::add_pointer<void(
        const ITensor *, const ITensor *, const ITensor *, ITensor *, ITensor *, const Window &)>::type;

    /** Pack RHS tensor
     *
     * @param[in]  rhs        Tensor b
     * @param[in]  bias       Bias data
     * @param[out] packed_rhs Destination buffer for packed RHS data
     */
    using PackRhsPtr = std::add_pointer<void(const ITensor *, const ITensor *, ITensor *)>::type;

    /** Size of packed RHS for data of given size
     *
     * @param[in] rows    Number of rows
     * @param[in] columns Number of columns
     *
     * @return Size of packed RHS data
     */
    using SizeOfPackedRhsPtr = std::add_pointer<size_t(const size_t, const size_t)>::type;

    /** Calculate window size
     *
     * @param[in] dst Destination tensor
     *
     * @return Window size for the micro-kernel
     */
    using GetWindowPtr = std::add_pointer<Window(const ITensorInfo *dst)>::type;

    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(CpuDynamicGemmKernelHeuristics);

    // Default constructor and destructor
    CpuDynamicGemmKernelHeuristics() noexcept {};
    ~CpuDynamicGemmKernelHeuristics() = default;

    /** Similar to @ref CpuDynamicGemmKernel::configure() */
    CpuDynamicGemmKernelHeuristics(const ITensorInfo *a,
                                   const ITensorInfo *b,
                                   const ITensorInfo *c,
                                   ITensorInfo       *d,
                                   float              alpha,
                                   float              beta,
                                   const GEMMInfo    &gemm_info = GEMMInfo());

    /** Return minimum workload size
     *
     * @return Minimum workload size for requested configuration in size_t
     */
    size_t mws() const;

    /** Prepare the micro-kernel for the run
     *
     * An example of an action that can be done here is b-tensor packing.
     *
     * @param[in] tensors              Tensors that will be used in the run
     * @param[in] run_packing          Whether b tensor should be packed
     * @param[in] pack_b_tensor_offset An offset of pack_rhs tensor in the tensors parameter
     */
    void prepare(ITensorPack &tensors, bool run_packing, const int pack_b_tensor_offset);

    /** Return the kernel to run
     *
     * @return The function pointer to the chosen kernel
     */
    KernelPtr kernel() const;

    /** Return the pack_rhs() function for the kernel
     *
     * @return The pointer to the pack_rhs() function
     */
    PackRhsPtr pack_rhs() const;

    /** Return the size_of_packed_rhs() function for the kernel
     *
     * @return The pointer to the size_of_packed_rhs() function
     */
    SizeOfPackedRhsPtr size_of_packed_rhs() const;

    /** Return the get_window() function for the kernel
     *
     * @return The pointer to the get_window() function
     */
    GetWindowPtr get_window() const;

    /** Return the name of the selected kernel
     *
     * @return Name of the selected kernel
     */
    const char *name() const;

    /** Return the scheduling hint e.g. dimension(s) to split
     *
     * @return an instance of @ref IScheduler::Hints to describe the scheduling hints
     */
    const IScheduler::Hints &scheduler_hint() const;

private:
    struct DynamicGemmKernel
    {
        const char                  *name{nullptr};
        const DataTypeISASelectorPtr is_selected{nullptr};

        KernelPtr          ukernel{nullptr};
        PackRhsPtr         pack_rhs{nullptr};
        SizeOfPackedRhsPtr size_of_packed_rhs{nullptr};
        GetWindowPtr       get_window{nullptr};
    };

    using KernelList = std::vector<DynamicGemmKernel>;
    using KernelMap  = std::map<DataType, KernelList>;

private:
    /** Chooses a kernel to run and saves it into _kernel data member
     *
     * @param[in] selector Selector object based on input and device configuration
     */
    void choose_kernel(const DataTypeISASelectorData &selector);

private:
    const static KernelList fp32_kernels;
    const static KernelMap  kernels;

    size_t                   _mws{ICPPKernel::default_mws};
    const DynamicGemmKernel *_kernel{nullptr};
    IScheduler::Hints        _hint{Window::DimY};
};

} // namespace heuristics
} // namespace kernels
} // namespace cpu
} // namespace arm_compute
#endif // ACL_SRC_CPU_KERNELS_DYNAMIC_GEMM_HEURISTICS_CPUDYNAMICGEMMKERNELHEURISTICS_H
