/*
 * Copyright (c) 2022 Arm Limited.
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

#ifndef ARM_COMPUTE_MISC_ITERABLE_H
#define ARM_COMPUTE_MISC_ITERABLE_H
namespace arm_compute
{
namespace utils
{
namespace memory
{
namespace
{
/**  Default polymorphic deep copy function, used by deep_unique_ptr
 *
 * @param ptr  Potentially polymorphic object to be deep copied
 * @return template <typename Base, typename Derived>*
 */
template <typename Base, typename Derived>
Base *default_polymorphic_copy(const Base *ptr)
{
    static_assert(std::is_base_of<Base, Derived>::value,
                  "Derived is not a specialization of Base");
    if(ptr == nullptr)
    {
        return nullptr;
    }
    return new Derived(*static_cast<const Derived *>(ptr));
}
} // namespace

/** A deep-copying unique pointer that also supports polymorphic cloning behavior
 *
 * @note The == operator compares the dereferenced value instead of the pointer itself.
 *
 * @tparam Base Base type
 */
template <typename Base>
class deep_unique_ptr
{
public:
    using CopyFunc = std::function<Base *(const Base *)>;

    deep_unique_ptr(std::nullptr_t val = nullptr) noexcept
        : _val{ val },
    _copy{}
    {
    }
    template <typename Derived, typename CopyFuncDerived>
    deep_unique_ptr(Derived *value, const CopyFuncDerived &copy) noexcept
        : _val{ value },
    _copy{ std::move(copy) }
    {
        static_assert(std::is_base_of<Base, Derived>::value,
                      "Derived is not a specialization of Base");
        static_assert(
            std::is_constructible<CopyFunc, CopyFuncDerived>::value,
            "CopyFuncDerived is not valid for a copy functor");
    }

    deep_unique_ptr(const deep_unique_ptr<Base> &ptr)
        : deep_unique_ptr(ptr.clone())
    {
    }
    deep_unique_ptr &operator=(const deep_unique_ptr<Base> &ptr)
    {
        deep_unique_ptr<Base> tmp(ptr);
        swap(*this, tmp);
        return *this;
    }

    deep_unique_ptr(deep_unique_ptr<Base> &&ptr) = default;
    deep_unique_ptr &operator=(deep_unique_ptr<Base> &&ptr) = default;
    ~deep_unique_ptr()                                      = default;
    friend void swap(deep_unique_ptr &ptr0, deep_unique_ptr<Base> &ptr1) noexcept
    {
        using std::swap;
        swap(ptr0._val, ptr1._val);
        swap(ptr0._copy, ptr1._copy);
    }
    Base &operator*() noexcept
    {
        return *_val;
    }

    const Base &operator*() const noexcept
    {
        return *_val;
    }

    Base *operator->() noexcept
    {
        return _val.operator->();
    }

    const Base *operator->() const noexcept
    {
        return _val.operator->();
    }

    Base *get() noexcept
    {
        return _val.get();
    }
    const Base *get() const noexcept
    {
        return _val.get();
    }

    explicit operator bool() const noexcept
    {
        return static_cast<bool>(_val);
    }

    bool operator==(const deep_unique_ptr<Base> &rhs) const
    {
        if(rhs.get() == nullptr && _val == nullptr)
        {
            return true;
        }
        else if(rhs.get() == nullptr || _val == nullptr)
        {
            return false;
        }
        else
        {
            return (*_val == *rhs);
        }
    }

private:
    deep_unique_ptr clone() const
    {
        return { _copy(_val.get()), CopyFunc(_copy) };
    }
    std::unique_ptr<Base> _val{ nullptr };
    CopyFunc              _copy{};
};

/** Utility function to create a polymorphic deep-copying unique pointer
 *
 * @tparam Base
 * @tparam Derived
 * @tparam CopyFunc
 * @param temp
 * @param copy
 * @return deep_unique_ptr<Base>
 */
template <typename Base, typename Derived, typename CopyFunc>
deep_unique_ptr<Base> make_deep_unique(Derived &&temp, CopyFunc copy)
{
    return
    {
        new Derived(std::move(temp)),
        CopyFunc{ std::move(copy) }
    };
}

template <typename Base, typename Derived>
deep_unique_ptr<Base> make_deep_unique(Derived &&temp)
{
    static_assert(std::is_base_of<Base, Derived>::value,
                  "Derived is not a specialization of Base");

    return make_deep_unique<Base, Derived>(
               std::move(temp), default_polymorphic_copy<Base, Derived>);
}

template <typename Base, typename Derived, typename... Args>
deep_unique_ptr<Base> make_deep_unique(Args &&... args)
{
    static_assert(std::is_constructible<Derived, Args...>::value,
                  "Cannot instantiate Derived from arguments");

    return make_deep_unique<Base, Derived>(
               std::move(Derived{ std::forward<Args>(args)... }));
}

} // namespace memory
} // namespace utils
} // namespace arm_compute
#endif // ARM_COMPUTE_MISC_ITERABLE_H