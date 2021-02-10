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
#ifndef SRC_RUNTIME_CL_GEMM_AUTO_HEURISTICS_CL_GEMM_AUTO_HEURISTICS_H
#define SRC_RUNTIME_CL_GEMM_AUTO_HEURISTICS_CL_GEMM_AUTO_HEURISTICS_H

#include "arm_compute/core/GPUTarget.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/CLTypes.h"

namespace arm_compute
{
namespace cl_gemm
{
namespace auto_heuristics
{
/** A collection of adaptor functions that enable the auto selection between mlgo-based heuristics and default heuristics */

/** Common query */
struct CommonQuery
{
    GPUTarget    gpu_target; /**< Which @ref GPUTarget to query about */
    DataType     data_type;  /**< Data type */
    unsigned int m;          /**< Number of rows for the lhs matrix. Lhs matrix NOT transposed */
    unsigned int n;          /**< Number of columns for the rhs matrix. Rhs matrix NOT transposed */
    unsigned int k;          /**< Number of rows for the rhs matrix. Rhs matrix NOT transposed */
    unsigned int b;          /**< Batch size */
};

/** Result of querying about GEMM type ( @ref CLGEMMKernelType) */
struct GEMMTypeResult
{
    GEMMTypeResult(bool valid, CLGEMMKernelType gemm_type)
        : valid{ valid }, gemm_type{ gemm_type }
    {
    }
    /** Test if the result is valid */
    operator bool() const
    {
        return valid;
    }
    bool             valid;     /** If the result is valid */
    CLGEMMKernelType gemm_type; /** @ref CLGEMMKernelType */
};

/** Result of querying about GEMM config ( @ref GEMMLHSMatrixInfo and @ref GEMMRHSMatrixInfo) */
struct GEMMConfigResult
{
    GEMMConfigResult(bool valid, const GEMMLHSMatrixInfo &lhs_info, const GEMMRHSMatrixInfo &rhs_info)
        : valid{ valid }, lhs_info{ lhs_info }, rhs_info{ rhs_info }
    {
    }
    /** Test if the result is valid */
    operator bool() const
    {
        return valid;
    }
    bool              valid;    /** If the result is valid */
    GEMMLHSMatrixInfo lhs_info; /** @ref GEMMLHSMatrixInfo */
    GEMMRHSMatrixInfo rhs_info; /** @ref GEMMRHSMatrixInfo */
};

/** Select gemm type based on mlgo heuristics
 * @param query  Query
 * @param reshape_b_only_on_first_run Additional query parameter if reshape b only on first run
 * @return GEMMTypeResult. Result is valid if bool(GEMMTypeResult) == true and invalid otherwise
 */
GEMMTypeResult select_mlgo_gemm_kernel(const CommonQuery &query, bool reshape_b_only_on_first_run);

/** Select gemm type based on default heuristics
 * @param query  Query
 * @param reshape_b_only_on_first_run Additional query parameter if reshape b only on first run
 * @return GEMMTypeResult. Result is valid if bool(GEMMTypeResult) == true and invalid otherwise
 */
GEMMTypeResult select_default_gemm_kernel(const CommonQuery &query, bool reshape_b_only_on_first_run);

/** Select gemm config based on mlgo heuristics
 * @param query Query
 * @return GEMMConfigResult. Result is valid if bool(GEMMConfigResult) == true and invalid otherwise
 */
GEMMConfigResult select_mlgo_gemm_config_reshaped_only_rhs(const CommonQuery &query);

/** Select gemm config based on default heuristics
 * @param query Query
 * @return GEMMConfigResult. Result is valid if bool(GEMMConfigResult) == true and invalid otherwise
 */
GEMMConfigResult select_default_gemm_config_reshaped_only_rhs(const CommonQuery &query);

/** Select gemm config based on mlgo heuristics
 * @param query Query
 * @return GEMMConfigResult. Result is valid if bool(GEMMConfigResult) == true and invalid otherwise
 */
GEMMConfigResult select_mlgo_gemm_config_reshaped(const CommonQuery &query);

/** Select gemm config based on default heuristics
 * @param query Query
 * @return GEMMConfigResult. Result is valid if bool(GEMMConfigResult) == true and invalid otherwise
 */
GEMMConfigResult select_default_gemm_config_reshaped(const CommonQuery &query);

/** Select gemm config based on mlgo heuristics
 * @param query Query
 * @return GEMMConfigResult. Result is valid if bool(GEMMConfigResult) == true and invalid otherwise
 */
GEMMConfigResult select_mlgo_gemm_config_native(const CommonQuery &query);

/** Select gemm config based on default heuristics
 * @param query Query
 * @return GEMMConfigResult. Result is valid if bool(GEMMConfigResult) == true and invalid otherwise
 */
GEMMConfigResult select_default_gemm_config_native(const CommonQuery &query);

} // namespace auto_heuristics
} // namespace cl_gemm
} // namespace arm_compute

#endif // SRC_RUNTIME_CL_GEMM_AUTO_HEURISTICS_CL_GEMM_AUTO_HEURISTICS_H