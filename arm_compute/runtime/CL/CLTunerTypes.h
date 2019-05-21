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
#ifndef __ARM_COMPUTE_CLTUNER_TYPES_H__
#define __ARM_COMPUTE_CLTUNER_TYPES_H__

#include "arm_compute/core/Error.h"
#include "arm_compute/core/utils/misc/Utility.h"

#include <map>

namespace arm_compute
{
/**< OpenCL tuner modes */
enum class CLTunerMode
{
    EXHAUSTIVE, /**< Searches all possible LWS configurations while tuning */
    NORMAL,     /**< Searches a subset of LWS configurations while tuning */
    RAPID       /**< Searches a minimal subset of LWS configurations while tuning */
};

/** Converts a string to a strong types enumeration @ref CLTunerMode
 *
 * @param[in] name String to convert
 *
 * @return Converted CLTunerMode enumeration
 */
inline CLTunerMode tuner_mode_from_name(const std::string &name)
{
    static const std::map<std::string, CLTunerMode> tuner_modes =
    {
        { "exhaustive", CLTunerMode::EXHAUSTIVE },
        { "normal", CLTunerMode::NORMAL },
        { "rapid", CLTunerMode::RAPID },
    };

#ifndef ARM_COMPUTE_EXCEPTIONS_DISABLED
    try
    {
#endif /* ARM_COMPUTE_EXCEPTIONS_DISABLED */
        return tuner_modes.at(arm_compute::utility::tolower(name));

#ifndef ARM_COMPUTE_EXCEPTIONS_DISABLED
    }
    catch(const std::out_of_range &)
    {
        throw std::invalid_argument(name);
    }
#endif /* ARM_COMPUTE_EXCEPTIONS_DISABLED */
}

/** Input Stream operator for @ref CLTunerMode
 *
 * @param[in]  stream     Stream to parse
 * @param[out] tuner_mode Output tuner mode
 *
 * @return Updated stream
 */
inline ::std::istream &operator>>(::std::istream &stream, CLTunerMode &tuner_mode)
{
    std::string value;
    stream >> value;
    tuner_mode = tuner_mode_from_name(value);
    return stream;
}
} // namespace arm_compute
#endif /*__ARM_COMPUTE_CLTUNER_TYPES_H__ */
