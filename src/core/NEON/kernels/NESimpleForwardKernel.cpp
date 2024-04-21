#include "src/core/NEON/kernels/NESimpleForwardKernel.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Validate.h"

#include "src/common/utils/Validate.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"

#include <arm_neon.h>

namespace arm_compute
{
namespace kernels
{
namespace
{

} // namespace
const char *NESimpleForwardKernel::name() const
{
    return "NESimpleForwardKernel";
}

void NESimpleForwardKernel::configure(ITensorPack& tensors, unsigned int total_nodes)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(tensors);
    _tensors = tensors;
    _total_nodes = total_nodes;
}

void NESimpleForwardKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    for(unsigned int idx = 0; idx < _total_nodes; idx++){
        ITensor *src = _tensors.get_tensor(TensorType::ACL_SRC_0+idx);
        ITensor *dst = _tensors.get_tensor(TensorType::ACL_DST_0+idx);
    }

}
} // namespace kernels
} // namespace arm_compute
