/*
 * Copyright (c) 2020-2022 Arm Limited.
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
#ifndef ARM_COMPUTE_CLTUNING_PARAMS_H
#define ARM_COMPUTE_CLTUNING_PARAMS_H

#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/runtime/CL/CLTunerTypes.h"
#include "support/StringSupport.h"

#include <ostream>

namespace arm_compute
{
/**< OpenCL tuner parameters */
class CLTuningParams
{
public:
    CLTuningParams(const CLTuningParams &tuning_params)
        : _lws(tuning_params._lws), _wbsm(tuning_params._wbsm)
    {
    }

    CLTuningParams(unsigned int lws_x = 0, unsigned int lws_y = 0, unsigned int lws_z = 0, int wbsm = 0)
        : _lws(lws_x, lws_y, lws_z), _wbsm(wbsm)
    {
    }
    CLTuningParams(cl::NDRange lws, cl_int wbsm = 0)
        : _lws(lws), _wbsm(wbsm)
    {
    }

    CLTuningParams(cl_int wbsm)
        : CLTuningParams(cl::NullRange, wbsm)
    {
    }
    CLTuningParams& operator=(const CLTuningParams &other)
    {
        _lws = other._lws;
        _wbsm = other._wbsm;
        return *this;
    }

    void set_lws(cl::NDRange lws)
    {
        _lws = lws;
    }

    cl::NDRange get_lws() const
    {
        return _lws;
    }

    void set_wbsm(cl_int wbsm)
    {
        _wbsm = wbsm;
    }

    cl_int get_wbsm() const
    {
        return _wbsm;
    }

    std::string to_string(CLTuningInfo tuning_info)
    {
        std::string tuning_params_string = "";
        tuning_params_string += ";" + support::cpp11::to_string(_lws[0]) + ";" + support::cpp11::to_string(_lws[1]) + ";" + support::cpp11::to_string(_lws[2]);
        if(tuning_info.tune_wbsm)
        {
            tuning_params_string += ";" + support::cpp11::to_string(_wbsm);
        }
        return tuning_params_string;
    }

    bool from_string(CLTuningInfo tuning_info, std::string tuning_params_string)
    {
        std::replace(tuning_params_string.begin(), tuning_params_string.end(), ';', ' ');
        std::vector<std::string> array;
        std::stringstream        ss(tuning_params_string);
        std::string              temp;
        while(ss >> temp)
        {
            array.push_back(temp);
        }
        // Read 3 values for lws
        if(array.size() < 3)
        {
            return false;
        }
        const unsigned int lws_0 = support::cpp11::stoi(array[0]);
        const unsigned int lws_1 = support::cpp11::stoi(array[1]);
        const unsigned int lws_2 = support::cpp11::stoi(array[2]);
        if(lws_0 == 0 && lws_1 == 0 && lws_2 == 0)
        {
            // If lws values are 0, cl::NullRange has to be used
            // otherwise the lws object will be badly created
            _lws = cl::NullRange;
        }
        else
        {
            _lws = cl::NDRange(lws_0, lws_1, lws_2);
        }
        array.erase(array.begin(), array.begin() + 3);
        if(tuning_info.tune_wbsm)
        {
            if(array.size() < 1)
            {
                return false;
            }
            _wbsm = support::cpp11::stoi(array[0]);
            array.erase(array.begin());
        }
        return true;
    }

private:
    cl::NDRange _lws;
    cl_int      _wbsm;
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_CLTUNING_PARAMS_H */
