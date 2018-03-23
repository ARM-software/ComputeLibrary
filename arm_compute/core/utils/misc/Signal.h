/*
 * Copyright (c) 2018 ARM Limited.
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
#ifndef __ARM_COMPUTE_MISC_SIGNAL_H__
#define __ARM_COMPUTE_MISC_SIGNAL_H__

#include <functional>

namespace arm_compute
{
namespace utils
{
namespace signal
{
namespace detail
{
/** Base signal class */
template <typename SignalType>
class SignalImpl;

/** Signal class function specialization */
template <typename ReturnType, typename... Args>
class SignalImpl<ReturnType(Args...)>
{
public:
    using Callback = std::function<ReturnType(Args...)>;

public:
    /** Default Constructor */
    SignalImpl() = default;

    /** Connects signal
     *
     * @param[in] cb Callback to connect the signal with
     */
    void connect(const Callback &cb)
    {
        _cb = cb;
    }

    /** Disconnects the signal */
    void disconnect()
    {
        _cb = nullptr;
    }

    /** Checks if the signal is connected
     *
     * @return True if there is a connection else false
     */
    bool connected() const
    {
        return (_cb != nullptr);
    }

    /** Calls the connected callback
     *
     * @param[in] args Callback arguments
     */
    void operator()(Args &&... args)
    {
        if(_cb)
        {
            _cb(std::forward<Args>(args)...);
        }
    }

private:
    Callback _cb{}; /**< Signal callback */
};
} // namespace detail

/** Signal alias */
template <class T>
using Signal = detail::SignalImpl<T>;
} // namespace signal
} // namespace utils
} // namespace arm_compute
#endif /* __ARM_COMPUTE_MISC_SIGNAL_H__ */
