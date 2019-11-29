/*
 * Copyright (c) 2016-2019 ARM Limited.
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
#ifndef VALIDATE_EXAMPLE_H
#define VALIDATE_EXAMPLE_H

#include "utils/Utils.h"
namespace arm_compute
{
namespace test
{
namespace framework
{
class Printer;
} // namespace framework
} // namespace test
namespace utils
{
/** Abstract ValidateExample class.
 *
 * All examples with a validation stage have to inherit from this class.
 */
class ValidateExample
{
public:
    /** Setup the example.
     *
     * @param[in] argc Argument count.
     * @param[in] argv Argument values.
     */
    virtual bool do_setup(int argc, char **argv)
    {
        ARM_COMPUTE_UNUSED(argc, argv);
        return true;
    };
    /** Run the example. */
    virtual void do_run() {};
    /** Run reference implementation and validate against the target output
     */
    virtual void do_validate()
    {
    }
    /** Teardown the example. */
    virtual void do_teardown() {};
    /** Print the example parameters
     *
     * @param[in,out] printer Printer to use to print the parameters
     */
    virtual void print_parameters(test::framework::Printer &printer)
    {
        ARM_COMPUTE_UNUSED(printer);
    }

    /** Default destructor */
    virtual ~ValidateExample() = default;
};
/** Run an example and handle the potential exceptions it throws
 *
 * @param[in] argc    Number of command line arguments
 * @param[in] argv    Command line arguments
 * @param[in] example Example to run
 */
int run_example(int argc, char **argv, std::unique_ptr<ValidateExample> example);

} // namespace utils
} // namespace arm_compute
#endif /* VALIDATE_EXAMPLE_H */
