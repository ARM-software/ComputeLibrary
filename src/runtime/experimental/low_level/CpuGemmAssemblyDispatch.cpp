/*
 * Copyright (c) 2024 Arm Limited.
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

#include "arm_compute/runtime/experimental/low_level/CpuGemmAssemblyDispatch.h"

#include "src/cpu/operators/internal/CpuGemmAssemblyDispatch.h"

/*
* Tests for a shallow wrapper class to expose CpuGemmAssemblyDispatch.
* Any new functionality should be added to
* src/cpu/operators/internal/CpuGemmAssemblyDispatch.h. and be tested separately.
*/
namespace arm_compute
{
namespace experimental
{
namespace op
{
namespace ll
{
namespace
{
// Create cpu::AsmGemmInfo from GEMMInfo.
// Unnecessary fields in GEMMInfo will be ignored.
cpu::AsmGemmInfo init_assembly_metadata(const GEMMInfo &info)
{
    cpu::AsmGemmInfo asm_info;
    asm_info.activation_info = info.activation_info();
    asm_info.fast_mode       = info.fast_math();
    asm_info.fixed_format    = info.fixed_format();
    asm_info.accumulate      = info.accumulate();
    asm_info.weight_format   = info.weight_format();
    asm_info.transpose_b =
        info.pretranspose_B(); // The "pretranspose_B" flag here is not the same as the pretranspose_B_array method. The flag here signals to pretranspose_B_array method if we want to perform additional transpose on B before the pretranspose_B_array method

    return asm_info;
}
} // namespace

struct CpuGemmAssemblyDispatch::Impl
{
    std::unique_ptr<cpu::CpuGemmAssemblyDispatch> cpu_gemm_assembly_dispatch{nullptr};
};

CpuGemmAssemblyDispatch::CpuGemmAssemblyDispatch() : _impl(std::make_unique<Impl>())
{
    _impl->cpu_gemm_assembly_dispatch = std::make_unique<cpu::CpuGemmAssemblyDispatch>();
}

CpuGemmAssemblyDispatch::~CpuGemmAssemblyDispatch() = default;

void CpuGemmAssemblyDispatch::configure(
    const ITensorInfo *a, const ITensorInfo *b, const ITensorInfo *c, ITensorInfo *d, const GEMMInfo &gemm_info)
{
    _impl->cpu_gemm_assembly_dispatch->configure(a, b, c, d, init_assembly_metadata(gemm_info));
}

Status CpuGemmAssemblyDispatch::validate(
    const ITensorInfo *a, const ITensorInfo *b, const ITensorInfo *c, const ITensorInfo *d, const GEMMInfo &gemm_info)
{
    if (gemm_info.reinterpret_input_as_3d() != false || gemm_info.depth_output_gemm3d() != false ||
        gemm_info.reshape_b_only_on_first_run() != true)
    {
        return Status(ErrorCode::RUNTIME_ERROR);
    }
    bool a_data_type_ok = a->data_type() == DataType::F32 || a->data_type() == DataType::F16;
    bool b_data_type_ok = b->data_type() == DataType::F32 || b->data_type() == DataType::F16;
    bool d_data_type_ok = d->data_type() == DataType::F32 || d->data_type() == DataType::F16;
    bool fixed_format_dtype_ok =
        (!gemm_info.fixed_format() ||
         (a->data_type() == DataType::F32 && b->data_type() == DataType::F32 && d->data_type() == DataType::F32));

    if (!(a_data_type_ok && b_data_type_ok && d_data_type_ok && fixed_format_dtype_ok))
    {
        return Status(ErrorCode::RUNTIME_ERROR);
    }

    return cpu::CpuGemmAssemblyDispatch::validate(a, b, c, d, init_assembly_metadata(gemm_info));
}

Status CpuGemmAssemblyDispatch::has_opt_impl(arm_compute::WeightFormat &weight_format,
                                             const ITensorInfo         *a,
                                             const ITensorInfo         *b,
                                             const ITensorInfo         *c,
                                             const ITensorInfo         *d,
                                             const GEMMInfo            &gemm_info)
{
    return cpu::CpuGemmAssemblyDispatch::has_opt_impl(weight_format, a, b, c, d, init_assembly_metadata(gemm_info));
}

bool CpuGemmAssemblyDispatch::is_activation_supported(const ActivationLayerInfo &activation)
{
    return cpu::CpuGemmAssemblyDispatch::is_activation_supported(activation);
}

bool CpuGemmAssemblyDispatch::is_configured() const
{
    return _impl->cpu_gemm_assembly_dispatch->is_configured();
}

void CpuGemmAssemblyDispatch::run(ITensorPack &tensors)
{
    _impl->cpu_gemm_assembly_dispatch->run(tensors);
}
void CpuGemmAssemblyDispatch::prepare(ITensorPack &constants)
{
    _impl->cpu_gemm_assembly_dispatch->prepare(constants);
}
experimental::MemoryRequirements CpuGemmAssemblyDispatch::workspace() const
{
    return _impl->cpu_gemm_assembly_dispatch->workspace();
}

} // namespace ll
} // namespace op
} // namespace experimental
} // namespace arm_compute
