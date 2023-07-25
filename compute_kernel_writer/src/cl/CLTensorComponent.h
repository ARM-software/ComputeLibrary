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

#ifndef CKW_SRC_CL_CLTENSORCOMPONENT_H
#define CKW_SRC_CL_CLTENSORCOMPONENT_H

#include "ckw/types/TensorComponentType.h"
#include "src/ITensorComponent.h"
#include "src/cl/CLTile.h"

namespace ckw
{

class CLTensorArgument;

/** A tensor component object that can be used as a tile.
 *
 * The tensor component is created by @ref CLTensorArgument object when it is used
 * either by the user or internally by a kernel writer operation.
 * It allows the user to perform operation on tensor component just like any other tile.
 *
 * Because of the nature of tensor component, it's always a scalar tile of 32-bit integer.
 *
 * To find the list of all tensor components, see @ref TensorComponentType.
 */
class CLTensorComponent : public CLTile, public ITensorComponent
{
public:
    /** Initialize a new instance of @ref CLTensorComponent class for dynamic component.
     *
     * @param[in] tensor         The tensor to which this component belongs.
     * @param[in] component_type The tensor component type.
     */
    CLTensorComponent(const CLTensorArgument &tensor, TensorComponentType component_type);

    /** Initialize a new instance of @ref CLTensorComponent class for compile-time constant component.
     *
     * @param[in] tensor         The tensor to which this component belongs.
     * @param[in] component_type The tensor component type.
     * @param[in] value          The value of the component.
     */
    CLTensorComponent(const CLTensorArgument &tensor, TensorComponentType component_type, int32_t value);

    /** Destructor. */
    virtual ~CLTensorComponent();

    ITile &tile() override;

    const ITile &tile() const override;

    TensorComponentType component_type() const override;

private:
    TensorComponentType _component_type{ TensorComponentType::Unknown };
};

} // namespace ckw

#endif // CKW_SRC_CL_CLTENSORCOMPONENT_H
