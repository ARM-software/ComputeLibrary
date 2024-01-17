/*
 * Copyright (c) 2023 Arm Limited.
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
#ifndef CKW_SRC_CL_CLTENSORARGUMENT_H
#define CKW_SRC_CL_CLTENSORARGUMENT_H

#include "ckw/types/TensorComponentType.h"
#include "ckw/types/TensorStorageType.h"

#include "src/cl/CLTensorComponent.h"
#include "src/ITensor.h"

#include <memory>
#include <string>
#include <vector>

namespace ckw
{
// Forward declarations
class TensorInfo;

class ITensorComponent;

/** OpenCL specific tensor argument
 *  Internally, the object keeps track of the components and storages used to minimize the number
 *  of kernel arguments required. Therefore, if we create this object but we do not access any components
 *  or storages, the storages() and components() method will return an empty list.
*/
class CLTensorArgument : public ITensor
{
public:
    /** Constructor
     *
     * @param[in] name                 Tensor name
     * @param[in] info                 Tensor info
     * @param[in] return_dims_by_value Flag to return the dimensions by value whenever it is possible.
     *                                 True, if the dimensions should be returned as value instead as variable.
    */
    CLTensorArgument(const std::string &name, const TensorInfo &info, bool return_dims_by_value);

    /** Destructor. */
    ~CLTensorArgument();

    /** Get a tensor component of the given type.
     *
     * This function is for internal use as it returns a reference to @ref CLTensorComponent object.
     * It provides rich functionalities and doesn't require unnecessary casting
     * unlike @ref CLTensorComponent::component which is for the public API and only returns
     * a reference to a generic @ref ITile object.
     */
    CLTensorComponent &cl_component(TensorComponentType component_type);

    // Inherited method overridden
    TensorStorageVariable                &storage(TensorStorageType x) override;
    ITile                                &component(TensorComponentType x) override;
    std::vector<TensorStorageVariable>    storages() const override;
    std::vector<const ITensorComponent *> components() const override;

private:
    std::string create_storage_name(TensorStorageType x) const;

    bool                                            _return_dims_by_value{false};
    std::vector<TensorStorageVariable>              _storages_used{};
    std::vector<std::unique_ptr<CLTensorComponent>> _components_used{};
};

} // namespace ckw

#endif // CKW_SRC_CL_CLTENSORARGUMENT_H
