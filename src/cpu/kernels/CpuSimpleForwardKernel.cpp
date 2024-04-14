#include "src/cpu/kernels/CpuSimpleForwardKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include "src/core/common/Registrars.h"
#include "src/core/CPP/Validate.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/Utils.h"
#include "src/core/helpers/WindowHelpers.h"
#include "src/cpu/kernels/softmax/list.h"

#include <vector>

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
namespace
{

} // namespace

void CpuSimpleForwardKernel::configure(const ITensorInfo *src1,
                                       const ITensorInfo *src2,
                                       const ITensorInfo *src3,
                                       ITensorInfo *dst1,
                                       ITensorInfo *dst2,
                                       ITensorInfo *dst3)
{
    auto_init_if_empty(*dst1, src1->clone()->set_tensor_shape(src1->tensor_shape()));
    auto_init_if_empty(*dst2, src2->clone()->set_tensor_shape(src2->tensor_shape()));
    auto_init_if_empty(*dst3, src3->clone()->set_tensor_shape(src3->tensor_shape()));

    Window win;

    win = calculate_max_window(*dst1, Steps());
    ICPPKernel::configure(win);
}


void CpuSimpleForwardKernel::run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_UNUSED(window);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);

    std::cout << "src/cpu/kernels/CpuSimpleForwardKernel.cpp" << std::endl;
    
}

const char *CpuSimpleForwardKernel::name() const
{
    return "CpuSimpleForwardKernel";
}

} // namespace kernels
} // namespace cpu
} // namespace arm_compute
