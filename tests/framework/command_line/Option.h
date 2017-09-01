/*
 * Copyright (c) 2017 ARM Limited.
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
#ifndef ARM_COMPUTE_TEST_OPTIONBASE
#define ARM_COMPUTE_TEST_OPTIONBASE

#include <string>

namespace arm_compute
{
namespace test
{
namespace framework
{
/** Abstract base class for a command line option. */
class Option
{
public:
    /** Constructor.
     *
     * @param[in] name Name of the option.
     */
    Option(std::string name);

    /** Constructor.
     *
     * @param[in] name        Name of the option.
     * @param[in] is_required Is the option required?
     * @param[in] is_set      Has a value been assigned to the option?
     */
    Option(std::string name, bool is_required, bool is_set);

    /** Default destructor. */
    virtual ~Option() = default;

    /** Parses the given string.
     *
     * @param[in] value String representation as passed on the command line.
     *
     * @return True if the value could be parsed by the specific subclass.
     */
    virtual bool parse(std::string value) = 0;

    /** Help message for the option.
     *
     * @return String representing the help message for the specific subclass.
     */
    virtual std::string help() const = 0;

    /** Name of the option.
     *
     * @return Name of the option.
     */
    std::string name() const;

    /** Set whether the option is required.
     *
     * @param[in] is_required Pass true if the option is required.
     */
    void set_required(bool is_required);

    /** Set the help message for the option.
     *
     * @param[in] help Option specific help message.
     */
    void set_help(std::string help);

    /** Is the option required?
     *
     * @return True if the option is required.
     */
    bool is_required() const;

    /** Has a value been assigned to the option?
     *
     * @return True if a value has been set.
     */
    bool is_set() const;

protected:
    std::string _name;
    bool        _is_required{ false };
    bool        _is_set{ false };
    std::string _help{};
};
} // namespace framework
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_OPTIONBASE */
