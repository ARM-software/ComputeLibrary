/*
 * Copyright (c) 2021 Arm Limited.
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
#ifndef SRC_CPU_CPUCONTEXT_H
#define SRC_CPU_CPUCONTEXT_H

#include "src/common/AllocatorWrapper.h"
#include "src/common/IContext.h"
#include "src/common/cpuinfo/CpuInfo.h"

namespace arm_compute
{
namespace cpu
{
/** Structure that encodes the CPU capabilities to be used */
struct CpuCapabilities
{
    cpuinfo::CpuInfo cpu_info{};
    int32_t          max_threads{ -1 };
};

/** CPU context implementation class */
class CpuContext final : public IContext
{
public:
    /** Default Constructor
     *
     * @param[in] options Creational options
     */
    explicit CpuContext(const AclContextOptions *options);
    /** Cpu Capabilities accessor
     *
     * @return The ISA capabilities to be used by the CPU
     */
    const CpuCapabilities &capabilities() const;
    /** Backing memory allocator accessor
     *
     * @return Allocator that allocates CPU memory
     */
    AllocatorWrapper &allocator();

    // Inherrited methods overridden
    ITensorV2 *create_tensor(const AclTensorDescriptor &desc, bool allocate) override;
    IQueue *create_queue(const AclQueueOptions *options) override;
    std::tuple<IOperator *, StatusCode> create_activation(const AclTensorDescriptor &src,
                                                          const AclTensorDescriptor     &dst,
                                                          const AclActivationDescriptor &act,
                                                          bool                           is_validate) override;

private:
    AllocatorWrapper _allocator;
    CpuCapabilities  _caps;
};
} // namespace cpu
} // namespace arm_compute

#endif /* SRC_CPU_CPUCONTEXT_H */