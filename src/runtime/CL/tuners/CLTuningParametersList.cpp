/*
 * Copyright (c) 2019-2021, 2023 Arm Limited.
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
#include "arm_compute/runtime/CL/tuners/CLTuningParametersList.h"

namespace arm_compute
{
namespace cl_tuner
{
constexpr unsigned int max_lws_supported_x{64u};
constexpr unsigned int max_lws_supported_y{32u};
constexpr unsigned int max_lws_supported_z{32u};

/** Non instantiable base class for Tuning parameters combinations that use Index2Coord mapping */
class CLTuningParametersList : public ICLTuningParametersList
{
protected:
    /* Shape of 4-D search space */
    TensorShape               search_space_shape{0, 0, 0, 0};
    std::vector<unsigned int> _lws_x{0};
    std::vector<unsigned int> _lws_y{0};
    std::vector<unsigned int> _lws_z{0};
    std::vector<int>          _wbsm{0}; /* Modify the batches size of workgroups distributed to compute units.
                                             The value is in the range [-31,+31].
                                             When 0, the runtime-selected wbs used is unmodified. */

    /** Constructor */
    CLTuningParametersList() = default;
    /** Copy Constructor */
    CLTuningParametersList(const CLTuningParametersList &) = default;
    /** Move Constructor */
    CLTuningParametersList(CLTuningParametersList &&) noexcept(true) = default;
    /** Assignment */
    CLTuningParametersList &operator=(const CLTuningParametersList &) = default;
    /** Move Assignment */
    CLTuningParametersList &operator=(CLTuningParametersList &&) noexcept(true) = default;
    /** Destructor */
    virtual ~CLTuningParametersList() = default;

    // Inherited methods overridden:
    virtual size_t size() override;
};

/** Exhaustive list of all possible Tuning parameters (lws) values */
class CLTuningParametersListExhaustive : public CLTuningParametersList
{
public:
    /** Prevent default constructor calls */
    CLTuningParametersListExhaustive() = delete;
    /** Constructor */
    CLTuningParametersListExhaustive(const cl::NDRange &gws, CLTuningInfo tuning_info);
    /** Copy Constructor */
    CLTuningParametersListExhaustive(const CLTuningParametersListExhaustive &) = default;
    /** Move Constructor */
    CLTuningParametersListExhaustive(CLTuningParametersListExhaustive &&) noexcept(true) = default;
    /** Assignment */
    CLTuningParametersListExhaustive &operator=(const CLTuningParametersListExhaustive &) = default;
    /** Move Assignment */
    CLTuningParametersListExhaustive &operator=(CLTuningParametersListExhaustive &&) noexcept(true) = default;
    /** Destructor */
    ~CLTuningParametersListExhaustive() = default;

    // Inherited methods overridden:
    CLTuningParams operator[](size_t) override;
};

/** A subset of LWS values that are either factors of gws when gws[2] < 16 or power of 2 */
class CLTuningParametersListNormal : public CLTuningParametersList
{
public:
    /** Constructor */
    CLTuningParametersListNormal(const cl::NDRange &gws, CLTuningInfo tuning_info);
    /** Copy Constructor */
    CLTuningParametersListNormal(const CLTuningParametersListNormal &) = default;
    /** Move Constructor */
    CLTuningParametersListNormal(CLTuningParametersListNormal &&) noexcept(true) = default;
    /** Assignment */
    CLTuningParametersListNormal &operator=(const CLTuningParametersListNormal &) = default;
    /** Move Assignment */
    CLTuningParametersListNormal &operator=(CLTuningParametersListNormal &&) noexcept(true) = default;
    /** Destructor */
    ~CLTuningParametersListNormal() = default;

    // Inherited methods overridden:
    CLTuningParams operator[](size_t) override;

    /** Prevent default constructor calls */
    CLTuningParametersListNormal() = default;

private:
    /** Utility function used to initialize the LWS values to test.
     *  Only the LWS values which are power of 2 or satisfy the modulo conditions with GWS are taken into account by the CLTuner
     *
     * @param[in, out] lws         Vector of LWS to test
     * @param[in]      gws         Size of the specific GWS
     * @param[in]      lws_max     Max LWS value allowed to be tested
     * @param[in]      mod_let_one True if the results of the modulo operation between gws and the lws can be less than one.
     */
    void
    initialize_lws_values(std::vector<unsigned int> &lws, unsigned int gws, unsigned int lws_max, bool mod_let_one);
};

/** A minimal subset of LWS values that only have 1,2 and 4/8 */
class CLTuningParametersListRapid : public CLTuningParametersListNormal
{
public:
    /** Prevent default constructor calls */
    CLTuningParametersListRapid() = delete;
    /** Constructor */
    CLTuningParametersListRapid(const cl::NDRange &gws, CLTuningInfo tuning_info);
    /** Copy Constructor */
    CLTuningParametersListRapid(const CLTuningParametersListRapid &) = default;
    /** Move Constructor */
    CLTuningParametersListRapid(CLTuningParametersListRapid &&) noexcept(true) = default;
    /** Assignment */
    CLTuningParametersListRapid &operator=(const CLTuningParametersListRapid &) = default;
    /** Move Assignment */
    CLTuningParametersListRapid &operator=(CLTuningParametersListRapid &&) noexcept(true) = default;
    /** Destructor */
    virtual ~CLTuningParametersListRapid() = default;

private:
    /** Utility function used to initialize the LWS values to test.
     *  Only the LWS values that have 1,2 and 4/8 for each dimension are taken into account by the CLTuner
     *
     * @param[in, out] lws     Vector of LWS to test
     * @param[in]      lws_max Max LWS value allowed to be tested
     */
    void initialize_lws_values(std::vector<unsigned int> &lws, unsigned int lws_max);
};

size_t CLTuningParametersList::size()
{
    return search_space_shape.total_size();
}

CLTuningParams CLTuningParametersListExhaustive::operator[](size_t index)
{
    ARM_COMPUTE_ERROR_ON(index >= size());
    auto coords = index2coords(search_space_shape, index);
    return CLTuningParams(coords[0] + 1U, coords[1] + 1U, coords[2] + 1U, static_cast<int>(coords[3]));
}

CLTuningParametersListExhaustive::CLTuningParametersListExhaustive(const cl::NDRange &gws, CLTuningInfo tuning_info)
{
    const auto lws_x_max = std::min(static_cast<unsigned int>(gws[0]), max_lws_supported_x);
    const auto lws_y_max = std::min(static_cast<unsigned int>(gws[1]), max_lws_supported_y);
    const auto lws_z_max = std::min(static_cast<unsigned int>(gws[2]), max_lws_supported_z);

    search_space_shape[0] = lws_x_max;
    search_space_shape[1] = lws_y_max;
    search_space_shape[2] = lws_z_max;
    search_space_shape[3] = 1;
    if (tuning_info.tune_wbsm)
    {
        _wbsm                 = {-3, -2, -1, 0, 1, 2, 3};
        search_space_shape[3] = _wbsm.size();
    }
}

CLTuningParams CLTuningParametersListNormal::operator[](size_t index)
{
    ARM_COMPUTE_ERROR_ON(index >= size());
    auto coords = index2coords(search_space_shape, index);
    return CLTuningParams(_lws_x[coords[0]], _lws_y[coords[1]], _lws_z[coords[2]], _wbsm[coords[3]]);
}

CLTuningParametersListNormal::CLTuningParametersListNormal(const cl::NDRange &gws, CLTuningInfo tuning_info)
{
    const auto lws_x_max = std::min(static_cast<unsigned int>(gws[0]), max_lws_supported_x);
    const auto lws_y_max = std::min(static_cast<unsigned int>(gws[1]), max_lws_supported_y);
    const auto lws_z_max = std::min(static_cast<unsigned int>(gws[2]), max_lws_supported_z);

    // Initialize the tuning parameters values to test
    _lws_x = {};
    _lws_y = {};
    _lws_z = {};
    initialize_lws_values(_lws_x, gws[0], lws_x_max,
                          gws[2] > 16); // Explore lws that are not factors of gws only when gws[2] > 16
    initialize_lws_values(_lws_y, gws[1], lws_y_max,
                          gws[2] > 16); // Explore lws that are not factors of gws only when gws[2] > 16
    initialize_lws_values(_lws_z, gws[2], lws_z_max, false);

    search_space_shape[0] = _lws_x.size();
    search_space_shape[1] = _lws_y.size();
    search_space_shape[2] = _lws_z.size();
    search_space_shape[3] = 1;
    if (tuning_info.tune_wbsm)
    {
        _wbsm                 = {-2, -1, 0, 1, 2};
        search_space_shape[3] = _wbsm.size();
    }
}

void CLTuningParametersListNormal::initialize_lws_values(std::vector<unsigned int> &lws,
                                                         unsigned int               gws,
                                                         unsigned int               lws_max,
                                                         bool                       mod_let_one)
{
    lws.push_back(1);

    for (unsigned int i = 2; i <= lws_max; ++i)
    {
        // Power of two condition
        const bool is_power_of_two = (i & (i - 1)) == 0;

        // Condition for the module accordingly with the mod_let_one flag
        const bool mod_cond = mod_let_one ? (gws % i) <= 1 : (gws % i) == 0;

        if (mod_cond || is_power_of_two)
        {
            lws.push_back(i);
        }
    }
}

CLTuningParametersListRapid::CLTuningParametersListRapid(const cl::NDRange &gws, CLTuningInfo tuning_info)
{
    const auto lws_x_max = std::min(static_cast<unsigned int>(gws[0]), 8u); // Limit exploration to 1 - 8
    const auto lws_y_max = std::min(static_cast<unsigned int>(gws[1]), 4u); // Limit exploration to 1 - 4
    const auto lws_z_max = std::min(static_cast<unsigned int>(gws[2]), 4u); // Limit exploration to 1 - 4

    // Initialize the LWS values to test
    _lws_x = {};
    _lws_y = {};
    _lws_z = {};
    initialize_lws_values(_lws_x, lws_x_max);
    initialize_lws_values(_lws_y, lws_y_max);
    initialize_lws_values(_lws_z, lws_z_max);

    search_space_shape[0] = _lws_x.size();
    search_space_shape[1] = _lws_y.size();
    search_space_shape[2] = _lws_z.size();
    search_space_shape[3] = 1;
    if (tuning_info.tune_wbsm)
    {
        _wbsm                 = {-1, 0, 1};
        search_space_shape[3] = _wbsm.size();
    }
}

void CLTuningParametersListRapid::initialize_lws_values(std::vector<unsigned int> &lws, unsigned int lws_max)
{
    lws.push_back(1);

    for (unsigned int i = 2; i <= lws_max; i *= 4)
    {
        lws.push_back(i);
    }
}

std::unique_ptr<ICLTuningParametersList> get_tuning_parameters_list(CLTuningInfo tuning_info, const cl::NDRange &gws)
{
    switch (tuning_info.tuner_mode)
    {
        case CLTunerMode::EXHAUSTIVE:
            return std::make_unique<CLTuningParametersListExhaustive>(gws, tuning_info);
        case CLTunerMode::NORMAL:
            return std::make_unique<CLTuningParametersListNormal>(gws, tuning_info);
        case CLTunerMode::RAPID:
            return std::make_unique<CLTuningParametersListRapid>(gws, tuning_info);
        default:
            return nullptr;
    }
}
} // namespace cl_tuner
} // namespace arm_compute
