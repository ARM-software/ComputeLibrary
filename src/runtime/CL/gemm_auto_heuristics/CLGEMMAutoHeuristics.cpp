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
#include "src/runtime/CL/gemm_auto_heuristics/CLGEMMAutoHeuristics.h"

#include "arm_compute/core/Log.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "arm_compute/runtime/CL/ICLGEMMKernelSelection.h"
#include "src/core/CL/ICLGEMMKernelConfiguration.h"
#include "src/core/CL/gemm/CLGEMMHelpers.h"
#include "src/core/CL/gemm/native/CLGEMMNativeKernelConfiguration.h"
#include "src/core/CL/gemm/reshaped/CLGEMMReshapedKernelConfiguration.h"
#include "src/core/CL/gemm/reshaped_only_rhs/CLGEMMReshapedOnlyRHSKernelConfiguration.h"
#include "src/runtime/CL/gemm/CLGEMMKernelSelection.h"
#include "src/runtime/CL/mlgo/MLGOHeuristics.h"
#include "src/runtime/CL/mlgo/Utils.h"
#include "utils/TypePrinter.h"

namespace arm_compute
{
namespace cl_gemm
{
namespace auto_heuristics
{
GEMMTypeResult select_mlgo_gemm_kernel(const CommonQuery &query, bool reshape_b_only_on_first_run)
{
    ARM_COMPUTE_UNUSED(reshape_b_only_on_first_run);
    bool             valid = false;
    CLGEMMKernelType gemm_type{};
    const auto       mlgo_heuristics = CLScheduler::get().gemm_heuristics();
    if(mlgo_heuristics != nullptr)
    {
        std::tie(valid, gemm_type) = mlgo_heuristics->get()->query_gemm_type(mlgo::Query{ string_from_target(query.gpu_target), query.data_type, query.m, query.n, query.k, query.b });
    }
    if(valid)
    {
        ARM_COMPUTE_LOG_INFO_MSG_WITH_FORMAT_CORE("MLGOHeuristics query returns gemm type: %s.", to_string(gemm_type).c_str());
    }
    else
    {
        ARM_COMPUTE_LOG_INFO_MSG_CORE("MLGOHeuristics query failed");
    }
    return GEMMTypeResult(valid, gemm_type);
}

GEMMTypeResult select_default_gemm_kernel(const CommonQuery &query, bool reshape_b_only_on_first_run)
{
    std::unique_ptr<ICLGEMMKernelSelection> default_heuristics = CLGEMMKernelSelectionFactory::create(query.gpu_target);
    ARM_COMPUTE_ERROR_ON_NULLPTR(default_heuristics.get());

    CLGEMMKernelSelectionParams params;
    params.m               = query.m;
    params.n               = query.n;
    params.k               = query.k;
    params.b               = query.b;
    params.is_rhs_constant = reshape_b_only_on_first_run;
    params.data_type       = query.data_type;

    const auto kernel_type = default_heuristics->select_kernel(params);
    return GEMMTypeResult(true, kernel_type);
}

GEMMConfigResult select_default_gemm_config_reshaped_only_rhs(const CommonQuery &query)
{
    GEMMLHSMatrixInfo                           lhs_info;
    GEMMRHSMatrixInfo                           rhs_info;
    std::unique_ptr<ICLGEMMKernelConfiguration> gemm_config = CLGEMMReshapedOnlyRHSKernelConfigurationFactory::create(query.gpu_target);
    ARM_COMPUTE_ERROR_ON_NULLPTR(gemm_config.get());
    std::tie(lhs_info, rhs_info) = gemm_config->configure(query.m, query.n, query.k, query.b, query.data_type);
    return GEMMConfigResult{ true, lhs_info, rhs_info };
}

GEMMConfigResult select_mlgo_gemm_config_reshaped_only_rhs(const CommonQuery &query)
{
    bool                            valid = false;
    GEMMLHSMatrixInfo               lhs_info;
    GEMMRHSMatrixInfo               rhs_info;
    mlgo::GEMMConfigReshapedOnlyRHS config{};
    const auto                      mlgo_heuristics = CLScheduler::get().gemm_heuristics();
    if(mlgo_heuristics != nullptr)
    {
        std::tie(valid, config) = mlgo_heuristics->get()->query_gemm_config_reshaped_only_rhs(mlgo::Query{ string_from_target(query.gpu_target), query.data_type, query.m, query.n, query.k, query.b });
    }
    if(valid)
    {
        ARM_COMPUTE_LOG_INFO_MSG_WITH_FORMAT_CORE("MLGOHeuristics query returns gemm config: %s.", to_string(config).c_str());
        // Setting irrelevant unsigned int parameters to 1 and bool parameters to false as they do no matter
        std::tie(lhs_info, rhs_info) = configure_lhs_rhs_info(query.m, query.n, config.m0, config.n0, config.k0, 1, config.h0, false, config.interleave_rhs, !config.transpose_rhs, config.transpose_rhs,
                                                              config.export_cl_image);
    }
    else
    {
        ARM_COMPUTE_LOG_INFO_MSG_CORE("MLGOHeuristics query failed");
    }
    return GEMMConfigResult{ valid, lhs_info, rhs_info };
}

GEMMConfigResult select_default_gemm_config_reshaped(const CommonQuery &query)
{
    GEMMLHSMatrixInfo                           lhs_info;
    GEMMRHSMatrixInfo                           rhs_info;
    std::unique_ptr<ICLGEMMKernelConfiguration> gemm_config = CLGEMMReshapedKernelConfigurationFactory::create(query.gpu_target);
    ARM_COMPUTE_ERROR_ON_NULLPTR(gemm_config.get());
    std::tie(lhs_info, rhs_info) = gemm_config->configure(query.m, query.n, query.k, query.b, query.data_type);
    return GEMMConfigResult{ true, lhs_info, rhs_info };
}

GEMMConfigResult select_mlgo_gemm_config_reshaped(const CommonQuery &query)
{
    bool                     valid = false;
    GEMMLHSMatrixInfo        lhs_info;
    GEMMRHSMatrixInfo        rhs_info;
    mlgo::GEMMConfigReshaped config{};
    const auto               mlgo_heuristics = CLScheduler::get().gemm_heuristics();
    if(mlgo_heuristics != nullptr)
    {
        std::tie(valid, config) = mlgo_heuristics->get()->query_gemm_config_reshaped(mlgo::Query{ string_from_target(query.gpu_target), query.data_type, query.m, query.n, query.k, query.b });
    }
    if(valid)
    {
        ARM_COMPUTE_LOG_INFO_MSG_WITH_FORMAT_CORE("MLGOHeuristics query returns gemm config: %s.", to_string(config).c_str());
        std::tie(lhs_info, rhs_info) = configure_lhs_rhs_info(query.m, query.n, config.m0, config.n0, config.k0, config.v0, config.h0, config.interleave_lhs, config.interleave_rhs, !config.transpose_rhs,
                                                              config.transpose_rhs, config.export_cl_image);
    }
    else
    {
        ARM_COMPUTE_LOG_INFO_MSG_CORE("MLGOHeuristics query failed");
    }
    return GEMMConfigResult{ valid, lhs_info, rhs_info };
}

GEMMConfigResult select_default_gemm_config_native(const CommonQuery &query)
{
    GEMMLHSMatrixInfo                           lhs_info;
    GEMMRHSMatrixInfo                           rhs_info;
    std::unique_ptr<ICLGEMMKernelConfiguration> gemm_config = CLGEMMNativeKernelConfigurationFactory::create(query.gpu_target);
    ARM_COMPUTE_ERROR_ON_NULLPTR(gemm_config.get());
    std::tie(lhs_info, rhs_info) = gemm_config->configure(query.m, query.n, query.k, query.b, query.data_type);
    return GEMMConfigResult{ true, lhs_info, rhs_info };
}

GEMMConfigResult select_mlgo_gemm_config_native(const CommonQuery &query)
{
    bool                   valid = false;
    GEMMLHSMatrixInfo      lhs_info;
    GEMMRHSMatrixInfo      rhs_info;
    mlgo::GEMMConfigNative config{};
    const auto             mlgo_heuristics = CLScheduler::get().gemm_heuristics();
    if(mlgo_heuristics != nullptr)
    {
        std::tie(valid, config) = mlgo_heuristics->get()->query_gemm_config_native(mlgo::Query{ string_from_target(query.gpu_target), query.data_type, query.m, query.n, query.k, query.b });
    }
    if(valid)
    {
        ARM_COMPUTE_LOG_INFO_MSG_WITH_FORMAT_CORE("MLGOHeuristics query returns gemm config: %s.", to_string(config).c_str());
        // Setting irrelevant unsigned int parameters to 1 and bool parameters to false as they do no matter
        std::tie(lhs_info, rhs_info) = configure_lhs_rhs_info(query.m, query.n, config.m0, config.n0, config.k0, 1, 1, false, false, false, false, false);
    }
    else
    {
        ARM_COMPUTE_LOG_INFO_MSG_CORE("MLGOHeuristics query failed");
    }
    return GEMMConfigResult{ valid, lhs_info, rhs_info };
}
} // namespace auto_heuristics

} // namespace cl_gemm
} // namespace arm_compute