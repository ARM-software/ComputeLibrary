/*
 * Copyright (c) 2022 Arm Limited.
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
#ifdef ENABLE_EXPERIMENTAL_DYNAMIC_FUSION
#ifndef ARM_COMPUTE_EXPERIMENTAL_DYNAMIC_FUSION_CLCOMPOSITEOPERATOR_H
#define ARM_COMPUTE_EXPERIMENTAL_DYNAMIC_FUSION_CLCOMPOSITEOPERATOR_H

#include "arm_compute/core/CL/CLCompileContext.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/IOperator.h"

#include "arm_compute/core/experimental/ClWorkload.h"

#include <memory>

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
/** Map OpTensor handles to their corresponding ITensor memory
 */
using OpTensorBinding = std::map<OpTensor, ITensor *>;

/** Map a kernel (as identified by its unit workload id) to its corresponding tensor pack
 *
 * @note External user should not use the add_tensor_pack method to alter this tensor pack map, and should only use the map returned by @ref bind_tensors
 */
class TensorPackMap
{
public:
    /** Find a tensor pack associated with the unit workload Id @p uwk_id
     *
     * @param[in] uwk_id unit workload Id associated with the tensor pack
     *
     * @return ITensorPack*
     */
    ITensorPack *find_tensor_pack(UnitWorkload::Id uwk_id);
    /** Get a tensor pack associated with @p uwk_id. Throws a exception if it cannot be found.
     *
     * @param[in] uwk_id unit workload Id associated with the tensor pack
     *
     * @return ITensorPack*
     */
    ITensorPack &get_tensor_pack(UnitWorkload::Id uwk_id);
    /** Add a tensor pack and associate it with unit workload Id @p uwk_id
     * @note Should not be used by external user
     *
     * @param[in] uwk_id      unit workload Id associated with the tensor pack
     * @param[in] tensor_pack Tensor Pack to be added
     */
    void add_tensor_pack(UnitWorkload::Id uwk_id, const ITensorPack &tensor_pack);

private:
    std::map<UnitWorkload::Id, ITensorPack> _tensor_packs{};
};

/** Holder of any auxiliary CLTensors required by a ClWorkload.
 *
 * @note The tensors are not allocated by default, and require the user to explicitly allocate them using the TensorInfo and AuxMemoryInfo
 *
 * @note This data holder must remain valid until the ClCompositeOperator that it's passed to is out of scope
 *
 * @note External user should not use the add_aux_tensor method, and should only use the data returned by @ref bind_tensors
 */
class ClAuxTensorData
{
public:
    /** A view of a single auxiliary data and the associated TensorInfo and AuxMemoryInfo
     */
    struct DataView
    {
        DataView() = default;
        DataView(CLTensor *tensor, const TensorInfo &tensor_info, const AuxMemoryInfo &memory_info)
            : tensor{ tensor }, tensor_info{ tensor_info }, memory_info{ memory_info }
        {
        }
        ~DataView()                     = default;
        DataView(const DataView &other) = default;
        DataView &operator=(const DataView &other) = default;
        DataView(DataView &&other)                 = default;
        DataView &operator=(DataView &&other) = default;
        CLTensor     *tensor{};      /**< Pointer to the auxiliary tensor */
        TensorInfo    tensor_info{}; /**< Associated TensorInfo */
        AuxMemoryInfo memory_info{}; /**< Memory requirement */
    };

    /** Add auxiliary tensor.
     *
     * @note Should not be used by external user
     *
     * @param[in] tensor_id   Any Id that can uniquely identify an auxiliary tensor. Usually ClWorkloadTensor Id
     * @param[in] tensor_info TensorInfo associated with the tensor
     * @param[in] memory_info Memory requirements
     *
     * @return CLTensor*  if successfully added, otherwise nullptr
     */
    CLTensor *add_aux_tensor(int tensor_id, const ITensorInfo &tensor_info, const AuxMemoryInfo &memory_info);

    /** Get views of all auxiliary tensors. This is mainly used for allocating the auxiliary tensors.
     *
     * @return std::vector<DataView>&
     */
    std::vector<DataView> &get_tensors();

private:
    std::map<int, std::unique_ptr<CLTensor>> _owned_tensors{};
    std::vector<DataView> _tensors{};
};

/** Bind tensor memory to packs used by prepare and run methods. Create auxiliary tensor objects and their memory requirements if needed
 *
 * @note This is the only method for external user to create ClAuxTensorData, and the prepare and run TensorPackMaps
 *
 * @param[out] aux_tensor_data  Auxiliary Tensors required by the workload
 * @param[out] prepare_pack_map TensorPackMap used by the prepare method
 * @param[out] run_pack_map     TensorPackMap used by the run method
 * @param[in]  workload         ClWorkload to bind the tensors to
 * @param[in]  op_tensors       CLTensor memory objects mapped from Core OpTensors
 *
 * @return Status
 */
Status bind_tensors(ClAuxTensorData &aux_tensor_data, TensorPackMap &prepare_pack_map, TensorPackMap &run_pack_map, const ClWorkload &workload, const OpTensorBinding &op_tensors);

/** Operator runtime to run a @ref ClWorkload
 *
 * @note User must explicitly call prepare before run otherwise run will fail.
 *
 */
class ClCompositeOperator
{
public:
    ClCompositeOperator();
    ~ClCompositeOperator();
    /** Configures a @ref ClCompositeOperator with a @ref ClWorkload
     * This includes the compilation of Cl kernels inside the @ref ClWorkload
     *
     * @param[in] ctx      CLCompileContext
     * @param[in] workload ClWorkload to configure with
     */
    void configure(const CLCompileContext &ctx, const ClWorkload &workload);
    /** Validate ClWorkload @p workload
     *
     * @param[in] workload ClWorkload to be validated
     *
     * @return Status
     */
    static Status validate(const ClWorkload &workload);
    /** Enqueue prepare workloads
     *
     * @param tensor_pack_map Tensors required by the prepare workloads
     */
    void prepare(TensorPackMap &tensor_pack_map);
    /** Enqueue run workloads
     *
     * @param tensor_pack_map Tensors required by the run workloads
     */
    void run(TensorPackMap &tensor_pack_map);

private:
    struct Implementation;
    std::unique_ptr<Implementation> _impl;
};

} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
#endif //ARM_COMPUTE_EXPERIMENTAL_DYNAMIC_FUSION_CLCOMPOSITEOPERATOR_H
#endif /* ENABLE_EXPERIMENTAL_DYNAMIC_FUSION */