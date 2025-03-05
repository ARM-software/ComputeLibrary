/*
 * Copyright (c) 2018-2022, 2024-2025 Arm Limited.
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
#ifndef ACL_SRC_CPU_KERNELS_ASSEMBLY_CPUGEMMASSEMBLYWRAPPERKERNEL_H
#define ACL_SRC_CPU_KERNELS_ASSEMBLY_CPUGEMMASSEMBLYWRAPPERKERNEL_H

#include "arm_compute/core/Error.h"
#include "arm_compute/core/experimental/Types.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"

#include "src/core/NEON/INEKernel.h"
#include "src/cpu/kernels/assembly/arm_gemm_compute_iface.hpp"

#include "gemm_arrays.hpp"
#include "gemm_common.hpp"

namespace arm_compute
{
class ITensor;

namespace cpu
{
namespace kernel
{
/** This class is a wrapper for the assembly kernels.
  *
  * Some kernels were written in assembly and highly optimised for specific CPUs like A53 or A55.
  * This class works as a wrapper for these assembly kernels. The arm compute library creates an instance
  * of CpuGemmAssemblyWrapperKernel and other auxiliary data structures to execute a single assembly kernel
  * in the context of an NEFunctions.
  *
  * The type T is the type of the actual kernel implemented in assembly which is of type
  *         template<typename To, typename Tr> class GemmCommon
  *
  *
  */
template <typename TypeInput, typename TypeWeight, typename TypeOutput>
class CpuGemmAssemblyWrapperKernel final : public INEKernel
{
public:
    /** Constructor
     */
    CpuGemmAssemblyWrapperKernel() : _kernel(nullptr), _name("CpuGemmAssemblyWrapperKernel")
    {
    }

    CpuGemmAssemblyWrapperKernel(CpuGemmAssemblyWrapperKernel &)            = delete;
    CpuGemmAssemblyWrapperKernel(CpuGemmAssemblyWrapperKernel &&)           = default;
    CpuGemmAssemblyWrapperKernel &operator=(CpuGemmAssemblyWrapperKernel &) = delete;

    const char *name() const override
    {
        return _name.c_str();
    }

    void run(const Window &window, const ThreadInfo &info) override
    {
        ARM_COMPUTE_ERROR_ON_NULLPTR((reinterpret_cast<void *>(_kernel)));
        ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);

        auto win = arm_gemm::to_ndcoord(window);

        arm_gemm::ndcoord_t thread_locator{};

        _kernel->execute(win, thread_locator, info.thread_id);
    }

    // Inherited methods overridden:
    void run_nd(const Window &window, const ThreadInfo &info, const Window &thread_locator) override
    {
        ARM_COMPUTE_ERROR_ON_NULLPTR((reinterpret_cast<void *>(_kernel)));
        ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);

        //convert between arm_compute and arm_gemm types
        auto ndc_win = arm_gemm::to_ndcoord(window);
        auto ndc_tlc = arm_gemm::to_ndcoord(thread_locator);

        _kernel->execute(ndc_win, ndc_tlc, info.thread_id);
    }

    void run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info) override
    {
        ARM_COMPUTE_ERROR_ON_NULLPTR((reinterpret_cast<void *>(_kernel)));
        ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);

        const auto *Aptr      = reinterpret_cast<const TypeInput *>(tensors.get_tensor(ACL_SRC_0)->buffer());
        const auto *Bptr      = reinterpret_cast<const TypeWeight *>(tensors.get_tensor(ACL_SRC_1)->buffer());
        const auto *bias      = reinterpret_cast<const TypeOutput *>(tensors.get_tensor(ACL_SRC_2)->buffer());
        void       *workspace = tensors.get_tensor(ACL_SRC_3)->buffer();
        auto       *Cptr      = reinterpret_cast<TypeOutput *>(tensors.get_tensor(ACL_DST)->buffer());

        ARM_COMPUTE_ERROR_ON_NULLPTR(Aptr, Cptr);

        // We make a copy of the original gemm arrays and then update the
        // source, bias, and destination pointers with the packed values.
        arm_gemm::GemmArrays<TypeInput, TypeWeight, TypeOutput> ga = _kernel->get_gemm_arrays();

        ga._Aptr = Aptr;
        ga._Bptr = Bptr;
        ga._bias = bias;
        ga._Cptr = Cptr;
        ga.set_working_space(workspace);

        auto win = arm_gemm::to_ndcoord(window);

        arm_gemm::ndcoord_t thread_locator{};

        _kernel->execute_stateless(win, thread_locator, info.thread_id, ga);
    }

    /** Configure window of the kernel
     *
     * @param[in] window Region on which to execute the kernel
     */
    void configure_window(const Window &win)
    {
        INEKernel::configure(win);
    }

    /** Initialise the kernel's input and output.
     *
     * @param[in] kernel          Pointer to an assembly kernel implementation.
     * @param[in] kernel_name_tag Tag to be attacehd to the kernel's name.
     */
    void configure(arm_gemm::GemmCommon<TypeInput, TypeWeight, TypeOutput> *kernel, std::string kernel_name_tag)
    {
        ARM_COMPUTE_ERROR_ON_NULLPTR((reinterpret_cast<void *>(kernel)));
        _kernel = kernel;

        Window win = to_window(kernel->get_window_size());

        INEKernel::configure(win);

        if (!kernel_name_tag.empty())
        {
            _name += "/" + kernel_name_tag;
        }
    }
    /** Return minimum workload size of the relevant kernel
     *
     * @param[in] platform     The CPU platform used to create the context.
     * @param[in] thread_count Number of threads in the execution.
     *
     * @return[out] small_network_mws         Minimum workload size for requested configuration.
     */
    size_t get_mws(const CPUInfo &platform, size_t thread_count) const override
    {
        ARM_COMPUTE_UNUSED(thread_count);
        ARM_COMPUTE_UNUSED(platform);

        return ICPPKernel::default_mws;
    }

private:
    arm_gemm::GemmCommon<TypeInput, TypeWeight, TypeOutput> *_kernel;
    std::string                                              _name;
};
} // namespace kernel
} // namespace cpu
} // namespace arm_compute
#endif // ACL_SRC_CPU_KERNELS_ASSEMBLY_CPUGEMMASSEMBLYWRAPPERKERNEL_H
