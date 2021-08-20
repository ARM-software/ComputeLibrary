/*
 * Copyright (c) 2018-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_CLELEMENTWISEUNARYLAYER_H
#define ARM_COMPUTE_CLELEMENTWISEUNARYLAYER_H

#include "arm_compute/runtime/IFunction.h"

#include "arm_compute/core/Types.h"

#include <memory>

namespace arm_compute
{
class CLCompileContext;
class ICLTensor;
class ITensorInfo;

/** Basic function to perform inverse square root on an input tensor. */
class CLRsqrtLayer : public IFunction
{
public:
    /** Default Constructor */
    CLRsqrtLayer();
    /** Default Destructor */
    ~CLRsqrtLayer();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLRsqrtLayer(const CLRsqrtLayer &) = delete;
    /** Default move constructor */
    CLRsqrtLayer(CLRsqrtLayer &&);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLRsqrtLayer &operator=(const CLRsqrtLayer &) = delete;
    /** Default move assignment operator */
    CLRsqrtLayer &operator=(CLRsqrtLayer &&);
    /** Initialize the function
     *
     * Valid data layouts:
     * - All
     *
     * Valid data type configurations:
     * |src            |dst            |
     * |:--------------|:--------------|
     * |F16            |F16            |
     * |F32            |F32            |
     *
     * @param[in]  input  Input tensor. Data types supported: F16/F32.
     * @param[out] output Output tensor. Data types supported: same as @p input.
     */
    void configure(const ICLTensor *input, ICLTensor *output);
    /** Initialize the function
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  input           Input tensor. Data types supported: F16/F32.
     * @param[out] output          Output tensor. Data types supported: same as @p input.
     */
    void configure(const CLCompileContext &compile_context, const ICLTensor *input, ICLTensor *output);
    /** Static function to check if given info will lead to a valid configuration of @ref CLRsqrtLayer
     *
     * @param[in] input  First tensor input info. Data types supported: F16/F32.
     * @param[in] output Output tensor info. Data types supported: Same as @p input.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output);

    // Inherited methods overridden:
    void run() override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

/** Basic function to perform exponential on an input tensor. */
class CLExpLayer : public IFunction
{
public:
    /** Default Constructor */
    CLExpLayer();
    /** Default Destructor */
    ~CLExpLayer();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLExpLayer(const CLExpLayer &) = delete;
    /** Default move constructor */
    CLExpLayer(CLExpLayer &&);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLExpLayer &operator=(const CLExpLayer &) = delete;
    /** Default move assignment operator */
    CLExpLayer &operator=(CLExpLayer &&);
    /** Initialize the function
     *
     * Valid data layouts:
     * - All
     *
     * Valid data type configurations:
     * |src            |dst            |
     * |:--------------|:--------------|
     * |F16            |F16            |
     * |F32            |F32            |
     *
     * @param[in]  input  Input tensor. Data types supported: F16/F32.
     * @param[out] output Output tensor. Data types supported: same as @p input.
     */
    void configure(const ICLTensor *input, ICLTensor *output);
    /** Initialize the function
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  input           Input tensor. Data types supported: F16/F32.
     * @param[out] output          Output tensor. Data types supported: same as @p input.
     */
    void configure(const CLCompileContext &compile_context, const ICLTensor *input, ICLTensor *output);
    /** Static function to check if given info will lead to a valid configuration of @ref CLExpLayer
     *
     * @param[in] input  First tensor input info. Data types supported: F16/F32.
     * @param[in] output Output tensor info. Data types supported: Same as @p input.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output);

    // Inherited methods overridden:
    void run() override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

/** Basic function to negate an input tensor. */
class CLNegLayer : public IFunction
{
public:
    /** Default Constructor */
    CLNegLayer();
    /** Default Destructor */
    ~CLNegLayer();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLNegLayer(const CLNegLayer &) = delete;
    /** Default move constructor */
    CLNegLayer(CLNegLayer &&);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLNegLayer &operator=(const CLNegLayer &) = delete;
    /** Default move assignment operator */
    CLNegLayer &operator=(CLNegLayer &&);
    /** Initialize the function
     *
     * Valid data layouts:
     * - All
     *
     * Valid data type configurations:
     * |src            |dst            |
     * |:--------------|:--------------|
     * |F16            |F16            |
     * |F32            |F32            |
     * |S32            |S32            |
     *
     * @param[in]  input  Input tensor. Data types supported: F16/F32/S32
     * @param[out] output Output tensor. Data types supported: same as @p input.
     */
    void configure(const ICLTensor *input, ICLTensor *output);
    /** Initialize the function
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  input           Input tensor. Data types supported: F16/F32/S32
     * @param[out] output          Output tensor. Data types supported: same as @p input.
     */
    void configure(const CLCompileContext &compile_context, const ICLTensor *input, ICLTensor *output);
    /** Static function to check if given info will lead to a valid configuration of @ref CLNegLayer
     *
     * @param[in] input  First tensor input info. Data types supported: F16/F32/S32
     * @param[in] output Output tensor info. Data types supported: Same as @p input.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output);

    // Inherited methods overridden:
    void run() override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

/** Basic function to calculate sine of an input tensor. */
class CLSinLayer : public IFunction
{
public:
    /** Default Constructor */
    CLSinLayer();
    /** Default Destructor */
    ~CLSinLayer();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLSinLayer(const CLSinLayer &) = delete;
    /** Default move constructor */
    CLSinLayer(CLSinLayer &&);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLSinLayer &operator=(const CLSinLayer &) = delete;
    /** Default move assignment operator */
    CLSinLayer &operator=(CLSinLayer &&);
    /** Initialize the function
     *
     * Valid data layouts:
     * - All
     *
     * Valid data type configurations:
     * |src            |dst            |
     * |:--------------|:--------------|
     * |F16            |F16            |
     * |F32            |F32            |
     *
     * @param[in]  input  Input tensor. Data types supported: F16/F32.
     * @param[out] output Output tensor. Data types supported: same as @p input.
     */
    void configure(const ICLTensor *input, ICLTensor *output);
    /** Initialize the function
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  input           Input tensor. Data types supported: F16/F32.
     * @param[out] output          Output tensor. Data types supported: same as @p input.
     */
    void configure(const CLCompileContext &compile_context, const ICLTensor *input, ICLTensor *output);
    /** Static function to check if given info will lead to a valid configuration of @ref CLSinLayer
     *
     * @param[in] input  First tensor input info. Data types supported: F16/F32.
     * @param[in] output Output tensor info. Data types supported: Same as @p input.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output);

    // Inherited methods overridden:
    void run() override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

/** Basic function to perform elementwise log on an input tensor. */
class CLLogLayer : public IFunction
{
public:
    /** Default Constructor */
    CLLogLayer();
    /** Default Destructor */
    ~CLLogLayer();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLLogLayer(const CLLogLayer &) = delete;
    /** Default move constructor */
    CLLogLayer(CLLogLayer &&);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLLogLayer &operator=(const CLLogLayer &) = delete;
    /** Default move assignment operator */
    CLLogLayer &operator=(CLLogLayer &&);
    /** Initialize the function
     *
     * Valid data layouts:
     * - All
     *
     * Valid data type configurations:
     * |src            |dst            |
     * |:--------------|:--------------|
     * |F16            |F16            |
     * |F32            |F32            |
     *
     * @param[in]  input  Input tensor. Data types supported: F16/F32.
     * @param[out] output Output tensor. Data types supported: same as @p input.
     */
    void configure(const ICLTensor *input, ICLTensor *output);
    /** Initialize the function
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  input           Input tensor. Data types supported: F16/F32.
     * @param[out] output          Output tensor. Data types supported: same as @p input.
     */
    void configure(const CLCompileContext &compile_context, const ICLTensor *input, ICLTensor *output);
    /** Static function to check if given info will lead to a valid configuration of @ref CLLogLayer
     *
     * @param[in] input  First tensor input info. Data types supported: F16/F32.
     * @param[in] output Output tensor info. Data types supported: Same as @p input.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output);

    // Inherited methods overridden:
    void run() override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

/** Basic function to get the absolute value of an input tensor. */
class CLAbsLayer : public IFunction
{
public:
    /** Default Constructor */
    CLAbsLayer();
    /** Default Destructor */
    ~CLAbsLayer();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLAbsLayer(const CLAbsLayer &) = delete;
    /** Default move constructor */
    CLAbsLayer(CLAbsLayer &&);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLAbsLayer &operator=(const CLAbsLayer &) = delete;
    /** Default move assignment operator */
    CLAbsLayer &operator=(CLAbsLayer &&);
    /** Initialize the function
     *
     * Valid data layouts:
     * - All
     *
     * Valid data type configurations:
     * |src            |dst            |
     * |:--------------|:--------------|
     * |F16            |F16            |
     * |F32            |F32            |
     *
     * @param[in]  input  Input tensor. Data types supported: F16/F32.
     * @param[out] output Output tensor. Data types supported: same as @p input.
     */
    void configure(const ICLTensor *input, ICLTensor *output);
    /** Initialize the function
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  input           Input tensor. Data types supported: F16/F32.
     * @param[out] output          Output tensor. Data types supported: same as @p input.
     */
    void configure(const CLCompileContext &compile_context, const ICLTensor *input, ICLTensor *output);
    /** Static function to check if given info will lead to a valid configuration of @ref CLAbsLayer
     *
     * @param[in] input  First tensor input info. Data types supported: F16/F32.
     * @param[in] output Output tensor info. Data types supported: Same as @p input.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output);

    // Inherited methods overridden:
    void run() override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

/** Basic function to get the round (to the nearest even) value of an input tensor. */
class CLRoundLayer : public IFunction
{
public:
    /** Default Constructor */
    CLRoundLayer();
    /** Default Destructor */
    ~CLRoundLayer();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLRoundLayer(const CLRoundLayer &) = delete;
    /** Default move constructor */
    CLRoundLayer(CLRoundLayer &&);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLRoundLayer &operator=(const CLRoundLayer &) = delete;
    /** Default move assignment operator */
    CLRoundLayer &operator=(CLRoundLayer &&);
    /** Initialize the function
     *
     * Valid data layouts:
     * - All
     *
     * Valid data type configurations:
     * |src            |dst            |
     * |:--------------|:--------------|
     * |F16            |F16            |
     * |F32            |F32            |
     *
     * @param[in]  input  Input tensor. Data types supported: F16/F32.
     * @param[out] output Output tensor. Data types supported: same as @p input.
     */
    void configure(const ICLTensor *input, ICLTensor *output);
    /** Initialize the function
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  input           Input tensor. Data types supported: F16/F32.
     * @param[out] output          Output tensor. Data types supported: same as @p input.
     */
    void configure(const CLCompileContext &compile_context, const ICLTensor *input, ICLTensor *output);
    /** Static function to check if given info will lead to a valid configuration of @ref CLRoundLayer
     *
     * @param[in] input  First tensor input info. Data types supported: F16/F32.
     * @param[in] output Output tensor info. Data types supported: Same as @p input.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output);

    // Inherited methods overridden:
    void run() override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_CLELEMENTWISEUNARYLAYER_H */
