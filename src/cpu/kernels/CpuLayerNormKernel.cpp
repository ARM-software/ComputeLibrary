#include "src/cpu/kernels/CpuLayerNormKernel.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Validate.h"

#include "src/common/utils/Validate.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"

namespace arm_compute
{
namespace cpu
{
namespace kernels
{

void CpuLayerNormKernel::configure(const ITensorInfo *input,
                                    ITensorInfo       *output,
                                    LayerNormLayerInfo   info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_ERROR_THROW_ON(validate(input, output, info));

    _info = info;

    Window      win;
    size_t      split_dimension;
    std::tie(win,split_dimension) = calculate_squashed_or_max_window(*input);
    TensorShape out_shape = input->tensor_shape();

    ICPPKernel::configure(win);

    // Auto initialize if empty
    set_shape_if_empty(*output, out_shape);
    set_data_type_if_unknown(*output, input->data_type());
}

Status CpuLayerNormKernel::validate(const ITensorInfo *input,
                                    const ITensorInfo *output,
                                    LayerNormLayerInfo   info)
{
    ARM_COMPUTE_UNUSED(input);
    ARM_COMPUTE_UNUSED(output);
    ARM_COMPUTE_UNUSED(info);
    return Status{};
}

void CpuLayerNormKernel::run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info)
{
    std::cout << "src/cpu/kernels/CpuLayerNormKernel.cpp" << std::endl;
    ARM_COMPUTE_UNUSED(tensors);
    ARM_COMPUTE_UNUSED(window);
    ARM_COMPUTE_UNUSED(info);
}

const char *CpuLayerNormKernel::name() const
{
    return "NELayerNormLayerKernel";
}

} // namespace kernels
} // namespace cpu
} // namespace arm_compute
