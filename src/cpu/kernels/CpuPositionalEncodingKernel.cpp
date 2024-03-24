#include "src/cpu/kernels/CpuPositionalEncodingKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/core/Validate.h"

#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"

namespace arm_compute
{
namespace cpu
{
namespace kernels
{

namespace
{
template <typename T>
void run_positional_encoding(const Window &window, const ITensor *src, const ITensor *dst, const unsigned int d_model)
{
    ARM_COMPUTE_UNUSED(window);
    ARM_COMPUTE_UNUSED(src);
    ARM_COMPUTE_UNUSED(dst);
    std::cout << "src/cpu/kernels/CpuPositionalEncodingKernel.cpp" << std::endl;
    std::cout << d_model << std::endl;
}

}

void CpuPositionalEncodingKernel::configure(const ITensorInfo *src, ITensorInfo *dst, const unsigned int d_model)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, dst);

    _d_model = d_model;

    // Configure output tensor info.
    auto_init_if_empty(*dst, TensorInfo(*src->clone()));

    // Configure kernel window
    Window win = calculate_max_window(*src, Steps());
    ICpuKernel::configure(win);
}


Status CpuPositionalEncodingKernel::validate(const ITensorInfo *src, const ITensorInfo *dst, const unsigned int d_model)
{
    ARM_COMPUTE_UNUSED(src);
    ARM_COMPUTE_UNUSED(dst);
    ARM_COMPUTE_UNUSED(d_model);

    return Status{};
}

void CpuPositionalEncodingKernel::run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);

    const auto src = tensors.get_const_tensor(TensorType::ACL_SRC);
    auto       dst = tensors.get_tensor(TensorType::ACL_DST);

    run_positional_encoding<float>(window, src, dst, _d_model);
}

const char * CpuPositionalEncodingKernel::name() const
{
    return "CpuPositionalEncodingKernel";
}

} // namespace kernels
} // namespace cpu
} // namespace arm_compute
