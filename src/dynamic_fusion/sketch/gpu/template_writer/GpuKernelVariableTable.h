/*
 * Copyright (c) 2022-2023 Arm Limited.
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
#ifndef SRC_DYNAMIC_FUSION_SKETCH_GPU_TEMPLATE_WRITER_GPUKERNELVARIABLETABLE
#define SRC_DYNAMIC_FUSION_SKETCH_GPU_TEMPLATE_WRITER_GPUKERNELVARIABLETABLE

#include "arm_compute/core/ITensorInfo.h"
#include "src/dynamic_fusion/sketch/gpu/GpuKernelArgument.h"
#include "support/AclRequires.h"
#include "support/StringSupport.h"

#include <set>
#include <string>
#include <type_traits>

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
class GpuKernelComponentGroup;

/** A table of all the variables used in the kernel.
 * Each kernel has exactly one variable table.
 */
class GpuKernelVariableTable
{
public:
    /** A tensor variable whose main purposes are:
     *  - Hold the newly assigned @ref GpuKernelArgumentInfo for the associated tensor info
     *  - Hold the generated variable name for the associated tensor info
     */
    struct TensorVariable
    {
    public:
        TensorVariable()                       = default;
        TensorVariable(const TensorVariable &) = default;
        TensorVariable       &operator=(const TensorVariable &) = default;
        ITensorInfo::Id       id{ ITensorInfo::invalid_tensor_id };
        std::string           uniq_name{ "empty" }; // Unique name, also the final variable name used in the built code
        GpuKernelArgumentInfo kernel_argument_info{};
        bool                  has_valid_id() const
        {
            return id != ITensorInfo::invalid_tensor_id;
        }
    };
    using VariableList = std::vector<TensorVariable>;

public:
    /** Declare a @ref TensorVariable for a corresponding tensor info.
     *
     * @param[in] comp_group    Component group the tensor belongs to
     * @param[in] tensor        Tensor info with which the new variable is associated
     * @param[in] argument_info Kernel argument information
     * @param[in] alias         Alias for the variable. Will be used as part of the variable name
     */
    void declare_variable(const GpuKernelComponentGroup &comp_group, const ITensorInfo *tensor, GpuKernelArgumentInfo argument_info, const std::string &alias = "unnamed");
    /** Get the @ref TensorVariable associated with @p tensor
     *
     * @param[in] tensor Tensor info to be queried
     *
     * @return TensorVariable
     */
    TensorVariable get_variable(const ITensorInfo *tensor) const;
    /** Get the @ref TensorVariable list associated with @p tensors
     * @note Empty tensors are skipped
     *
     * @param[in] tensors List of tensor infos to be queried
     *
     * @return VariableList
     */
    VariableList get_variable_list(const std::vector<const ITensorInfo *> &tensors) const;

private:
    std::map<ITensorInfo::Id, TensorVariable> _vars{};
};

/** A tag value will substitute a tag in a string template during its instantiation */
struct TagVal
{
    /** Default constructor */
    TagVal() = default;
    /** Construct a @ref TagVal from a @ref GpuKernelVariableTable::TensorVariable */
    TagVal(const GpuKernelVariableTable::TensorVariable &var);
    /** Construct a @ref TagVal from an integral type */
    template <typename T, ARM_COMPUTE_REQUIRES_TA(std::is_integral<T>::value)>
    TagVal(T val)
        : value{ support::cpp11::to_string(val) }
    {
    }
    /** Construct a @ref TagVal from a string */
    TagVal(const std::string &val);
    /** Construct a @ref TagVal from a c-style string */
    TagVal(const char *val);
    /** Construct a @ref TagVal from a @ref DataType */
    TagVal(const DataType &data_type);
    /** Get the value of the TagVal as a converted string */
    std::string value{};
};

/** A tag used in a string template is a placeholder string to be substituted by real values during template instantiation */
using Tag = std::string;

/** Tag lookup table. It is used to instantiate a string template */
using TagLUT = std::unordered_map<Tag, TagVal>;

} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
#endif /* SRC_DYNAMIC_FUSION_SKETCH_GPU_TEMPLATE_WRITER_GPUKERNELVARIABLETABLE */
