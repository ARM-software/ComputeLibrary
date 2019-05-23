/*
 * Copyright (c) 2019 ARM Limited.
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
#ifndef __ARM_COMPUTE_CL_LWS_LIST_H__
#define __ARM_COMPUTE_CL_LWS_LIST_H__

#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/runtime/CL/CLTunerTypes.h"
#include "support/ToolchainSupport.h"
#include <memory>

namespace arm_compute
{
namespace cl_tuner
{
constexpr unsigned int max_lws_supported_x{ 64u };
constexpr unsigned int max_lws_supported_y{ 32u };
constexpr unsigned int max_lws_supported_z{ 32u };

/** Interface for LWS lists */
class ICLLWSList
{
public:
    /** Constructor */
    ICLLWSList() = default;
    /** Copy Constructor */
    ICLLWSList(const ICLLWSList &) = default;
    /** Move Constructor */
    ICLLWSList(ICLLWSList &&) noexcept(true) = default;
    /** Assignment */
    ICLLWSList &operator=(const ICLLWSList &) = default;
    /** Move Assignment */
    ICLLWSList &operator=(ICLLWSList &&) noexcept(true) = default;
    /** Destructor */
    virtual ~ICLLWSList() = default;

    /** Return the LWS value at the given index.
     *
     * @return LWS value at the given index
     */
    virtual cl::NDRange operator[](size_t) = 0;

    /** LWS list size.
     *
     * @return LWS list size
     */
    virtual size_t size() = 0;
};

/** Non instantiable base class for LWS combinations that use Index2Cooard mapping */
class CLLWSList : public ICLLWSList
{
protected:
    /* Shape of 3-D search space */
    TensorShape search_space_shape{ 0, 0, 0 };

    /** Constructor */
    CLLWSList() = default;
    /** Copy Constructor */
    CLLWSList(const CLLWSList &) = default;
    /** Move Constructor */
    CLLWSList(CLLWSList &&) noexcept(true) = default;
    /** Assignment */
    CLLWSList &operator=(const CLLWSList &) = default;
    /** Move Assignment */
    CLLWSList &operator=(CLLWSList &&) noexcept(true) = default;
    /** Destructor */
    virtual ~CLLWSList() = default;

    // Inherited methods overridden:
    virtual size_t size() override;
};

/** Exhaustive list of all possible LWS values */
class CLLWSListExhaustive : public CLLWSList
{
public:
    /** Prevent default constructor calls */
    CLLWSListExhaustive() = delete;
    /** Constructor */
    CLLWSListExhaustive(const cl::NDRange &gws);
    /** Copy Constructor */
    CLLWSListExhaustive(const CLLWSListExhaustive &) = default;
    /** Move Constructor */
    CLLWSListExhaustive(CLLWSListExhaustive &&) noexcept(true) = default;
    /** Assignment */
    CLLWSListExhaustive &operator=(const CLLWSListExhaustive &) = default;
    /** Move Assignment */
    CLLWSListExhaustive &operator=(CLLWSListExhaustive &&) noexcept(true) = default;
    /** Destructor */
    ~CLLWSListExhaustive() = default;

    // Inherited methods overridden:
    cl::NDRange operator[](size_t) override;
};

/** A subset of LWS values that are either factors of gws when gws[2] < 16 or power of 2 */
class CLLWSListNormal : public CLLWSList
{
public:
    /** Constructor */
    CLLWSListNormal(const cl::NDRange &gws);
    /** Copy Constructor */
    CLLWSListNormal(const CLLWSListNormal &) = default;
    /** Move Constructor */
    CLLWSListNormal(CLLWSListNormal &&) noexcept(true) = default;
    /** Assignment */
    CLLWSListNormal &operator=(const CLLWSListNormal &) = default;
    /** Move Assignment */
    CLLWSListNormal &operator=(CLLWSListNormal &&) noexcept(true) = default;
    /** Destructor */
    ~CLLWSListNormal() = default;

    // Inherited methods overridden:
    cl::NDRange operator[](size_t) override;

protected:
    std::vector<unsigned int> _lws_x{};
    std::vector<unsigned int> _lws_y{};
    std::vector<unsigned int> _lws_z{};

    /** Prevent default constructor calls */
    CLLWSListNormal() = default;

private:
    /** Utility function used to initialize the LWS values to test.
     *  Only the LWS values which are power of 2 or satisfy the modulo conditions with GWS are taken into account by the CLTuner
     *
     * @param[in, out] lws         Vector of LWS to test
     * @param[in]      gws         Size of the specific GWS
     * @param[in]      lws_max     Max LWS value allowed to be tested
     * @param[in]      mod_let_one True if the results of the modulo operation between gws and the lws can be less than one.
     */
    void initialize_lws_values(std::vector<unsigned int> &lws, unsigned int gws, unsigned int lws_max, bool mod_let_one);
};

/** A minimal subset of LWS values that only have 1,2 and 4/8 */
class CLLWSListRapid : public CLLWSListNormal
{
public:
    /** Prevent default constructor calls */
    CLLWSListRapid() = delete;
    /** Constructor */
    CLLWSListRapid(const cl::NDRange &gws);
    /** Copy Constructor */
    CLLWSListRapid(const CLLWSListRapid &) = default;
    /** Move Constructor */
    CLLWSListRapid(CLLWSListRapid &&) noexcept(true) = default;
    /** Assignment */
    CLLWSListRapid &operator=(const CLLWSListRapid &) = default;
    /** Move Assignment */
    CLLWSListRapid &operator=(CLLWSListRapid &&) noexcept(true) = default;
    /** Destructor */
    virtual ~CLLWSListRapid() = default;

private:
    /** Utility function used to initialize the LWS values to test.
     *  Only the LWS values that have 1,2 and 4/8 for each dimension are taken into account by the CLTuner
     *
     * @param[in, out] lws     Vector of LWS to test
     * @param[in]      lws_max Max LWS value allowed to be tested
     */
    void initialize_lws_values(std::vector<unsigned int> &lws, unsigned int lws_max);
};

/** Factory to construct an ICLLWSList object based on the CL tuner mode */
class CLLWSListFactory final
{
public:
    /** Construct an ICLLWSList object for the given tuner mode and gws configuration.
     *
     * @return unique_ptr to the requested ICLLWSList implementation.
     */
    static std::unique_ptr<ICLLWSList> get_lws_list(CLTunerMode mode, const cl::NDRange &gws)
    {
        switch(mode)
        {
            case CLTunerMode::EXHAUSTIVE:
                return arm_compute::support::cpp14::make_unique<CLLWSListExhaustive>(gws);
            case CLTunerMode::NORMAL:
                return arm_compute::support::cpp14::make_unique<CLLWSListNormal>(gws);
            case CLTunerMode::RAPID:
                return arm_compute::support::cpp14::make_unique<CLLWSListRapid>(gws);
            default:
                return nullptr;
        }
    }
};
} // namespace cl_tuner
} // namespace arm_compute
#endif /*__ARM_COMPUTE_CL_LWS_LIST_H__ */
