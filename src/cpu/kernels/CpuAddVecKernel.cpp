#include "src/cpu/kernels/CpuAddVecKernel.h"

#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Validate.h"

#include "src/core/common/Registrars.h"
#include "src/core/CPP/Validate.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"
#include "src/cpu/kernels/add_vec/list.h"

#include <array>

#if defined(ENABLE_FP32_KERNELS)
namespace
{
static constexpr size_t default_mws_N1_fp32_neon = 24536;
static constexpr size_t default_mws_V1_fp32_neon = 40510;
} // namespace
#endif /* ENABLE_FP32_KERNELS */

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
namespace
{
static const std::vector<CpuAddVecKernel::AddKernel> available_kernels = {
    {"neon_fp32_add_vec", [](const CpuAddVecKernelDataTypeISASelectorData &data) { return (data.dt == DataType::F32); },
     REGISTER_FP32_NEON(arm_compute::cpu::add_vec_fp32_neon)},
    };

Status
validate_arguments(const ITensorInfo &src0, const ITensorInfo &src1, const ITensorInfo &dst, ConvertPolicy policy)
{
    ARM_COMPUTE_UNUSED(src0);
    ARM_COMPUTE_UNUSED(src1);
    ARM_COMPUTE_UNUSED(dst);
    ARM_COMPUTE_UNUSED(policy);
    
    return Status{};
}
} // namespace

void CpuAddVecKernel::configure(const ITensorInfo *src0, const ITensorInfo *src1, ITensorInfo *dst, size_t src0_target_dim, size_t src1_target_dim, ConvertPolicy policy)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src0, src1, dst);
    ARM_COMPUTE_UNUSED(src1);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(*src0, *src1, *dst, policy));
    
    _src0_target_dim = src0_target_dim;
    _src1_target_dim = src1_target_dim;

    const auto uk                 = CpuAddVecKernel::get_implementation<CpuAddVecKernelDataTypeISASelectorData>(
        CpuAddVecKernelDataTypeISASelectorData{src0->data_type(), CPUInfo::get().get_isa()});

    ARM_COMPUTE_ERROR_ON_NULLPTR(uk);

    _policy     = policy;
    _run_method = uk->ukernel;
    _name       = std::string("CpuAddVecKernel").append("/").append(uk->name);

    // Auto initialize dst if not initialized
    auto_init_if_empty(*dst, src0->clone()->set_tensor_shape(src0->tensor_shape()));

    // Explictly set tensor shape
    dst->set_tensor_shape(src0->tensor_shape());

    // Configure kernel window
    Window win;
    win.use_tensor_dimensions(src0->tensor_shape());
    ICpuKernel::configure(win);

}

Status
CpuAddVecKernel::validate(const ITensorInfo *src0, const ITensorInfo *src1, const ITensorInfo *dst, size_t src0_target_dim, size_t src1_target_dim, ConvertPolicy policy)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src0, src1, dst);
    ARM_COMPUTE_UNUSED(src0_target_dim);
    ARM_COMPUTE_UNUSED(src1_target_dim);

    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(*src0, *src1, *dst, policy));

    return Status{};
}

void CpuAddVecKernel::run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICpuKernel::window(), window);

    ARM_COMPUTE_ERROR_ON(tensors.empty());
    ARM_COMPUTE_ERROR_ON(_run_method == nullptr);

    const ITensor *src0 = tensors.get_const_tensor(TensorType::ACL_SRC_0);
    const ITensor *src1 = tensors.get_const_tensor(TensorType::ACL_SRC_1);
    ITensor       *dst  = tensors.get_tensor(TensorType::ACL_DST);

    _run_method(src0, src1, dst, _src0_target_dim, _src1_target_dim, _policy, window);
}

const char *CpuAddVecKernel::name() const
{
    return _name.c_str();
}

const std::vector<CpuAddVecKernel::AddKernel> &CpuAddVecKernel::get_available_kernels()
{
    return available_kernels;
}

size_t CpuAddVecKernel::get_mws(const CPUInfo &platform, size_t thread_count) const
{
    ARM_COMPUTE_UNUSED(thread_count);

#if defined(ENABLE_FP32_KERNELS)
    if (this->_run_method == &add_vec_fp32_neon)
    {
        size_t mws = ICPPKernel::default_mws;
        if (platform.get_cpu_model() == CPUModel::N1)
        {
            mws = default_mws_N1_fp32_neon;
        }
        else if (platform.get_cpu_model() == CPUModel::V1)
        {
            mws = default_mws_V1_fp32_neon;
        }
        else
        {
            return ICPPKernel::default_mws;
        }

        // tensor is 1D or was re-interpreted as 1D
        if (this->window().shape().num_dimensions() == 1)
        {
            return mws;
        }
        else
        {
            // scale mws down by the number of elements along all the dimensions (x, z, w, etc) except the one
            // that we parallelize along (the y dimension). This allows for parallelization when the Y_SIZE is small
            // but the other sizes are large, which boosts performance.
            mws = static_cast<size_t>(mws / (this->window().num_iterations_total() / this->window().num_iterations(1)));
            return std::max(static_cast<size_t>(1), mws);
        }
    }
#else  /* ENABLE_FP32_KERNELS */
    ARM_COMPUTE_UNUSED(platform);
#endif /* ENABLE_FP32_KERNELS */
    return ICPPKernel::default_mws;
}

} // namespace kernels
} // namespace cpu
} // namespace arm_compute
