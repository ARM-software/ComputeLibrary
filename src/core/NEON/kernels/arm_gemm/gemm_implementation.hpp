/*
 * Copyright (c) 2018-2019 ARM Limited.
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

#include <arm_gemm.hpp>

#include <functional>

namespace arm_gemm {

template<typename Top, typename Tret>
struct GemmImplementation {
    const GemmMethod                                               method;
    const char *                                                   name;
    std::function<bool(const GemmArgs<Tret> &)>                    is_supported;
    std::function<bool(const GemmArgs<Tret> &)>                    is_recommended;
    std::function<GemmCommon<Top, Tret> *(const GemmArgs<Tret> &)> instantiate;
};

/* "Master" function implemented for each valid combination of types.
 * Returns a list of GEMM implementation descriptors for processing by the
 * other functions, terminated by an implementation with
 * method==GemmMethod::DEFAULT.  */
template<typename Top, typename Tret>
const GemmImplementation<Top, Tret> *gemm_implementation_list();

/*
 * Select a GEMM implementation for the given arguments.
 *
 * The logic here returns the first method on the list which supports the
 * requested problem parameters, matches the provided filters (method and/or
 * name string match) and recommends itself.
 *
 * If there is no such method, it will return the first method which
 * supports the requested parameters and passes the filters, regardless of
 * recommendation.
 *
 * If no method supports the requested parameters and passes the filters,
 * this function returns false and doesn't touch the provided pointer
 * reference.
 */
template<typename Top, typename Tret>
bool find_implementation(const GemmArgs<Tret> &args, const GemmImplementation<Top, Tret> * &impl) {
    auto gemms = gemm_implementation_list<Top, Tret>();
    const GemmConfig *cfg = args._cfg;

    const GemmImplementation<Top, Tret> *saved_impl = nullptr;

    for (auto i = gemms; i->method != GemmMethod::DEFAULT; i++) {
        /* Skip if this implementation doesn't support these args. */
        if (i->is_supported != nullptr && !i->is_supported(args)) {
            continue;
        }

        /* Skip if a specific method is requested and this is a different one. */
        if (cfg && cfg->method != GemmMethod::DEFAULT && i->method != cfg->method) {
            continue;
        }

        /* Skip if a filter is to be applied and it doesn't match. */
        if (cfg && cfg->filter != "" && !strstr(i->name, cfg->filter.c_str())) {
            continue;
        }

        /* At this point, if we don't have a saved implementation, save this
         * one.  This is so that we always return something if a filter
         * matches, even if it doesn't recommend itself.
         */
        if (saved_impl == nullptr) {
            saved_impl=i;
        }

        /* Check that this method recommends itself. */
        if (i->is_recommended != nullptr && !i->is_recommended(args)) {
            continue;
        }

        impl=i;

        return true;
    }

    /* We didn't find an option matching the filters that recommended
     * itself.  But if we found something earlier that matched the filters
     * but wasn't recommended, return it here.  */
    if (saved_impl != nullptr) {
        impl = saved_impl;
        return true;
    }

    return false;
}

template<typename Top, typename Tret>
std::vector<std::string> get_compatible_kernels(const GemmArgs<Tret> &args) {
    std::vector<std::string> res;

    auto gemms = gemm_implementation_list<Top, Tret>();

    for (auto i = gemms; i->method != GemmMethod::DEFAULT; i++) {
        /* Check that this implementation supports the presented problem. */
        if (i->is_supported != nullptr && !i->is_supported(args)) {
            continue;
        }

        res.push_back(i->name);
    }

    return res;
}

template<typename Top, typename Tret>
UniqueGemmCommon<Top, Tret> gemm(const GemmArgs<Tret> &args) {
    const GemmImplementation<Top, Tret> *impl;

    if (find_implementation<Top, Tret>(args, impl)) {
        return UniqueGemmCommon<Top, Tret>(impl->instantiate(args));
    }

    return UniqueGemmCommon<Top, Tret>(nullptr);
}

template<typename Top, typename Tret>
KernelDescription get_gemm_method(const GemmArgs<Tret> &args) {
    const GemmImplementation<Top, Tret> *impl;

    if (find_implementation<Top, Tret>(args, impl)) {
        return KernelDescription(impl->method, impl->name);
    }

    /* This shouldn't happen - there should always be at least one valid implementation. */
    return KernelDescription();
}

template<typename Top, typename Tret>
bool method_is_compatible(GemmMethod method, const GemmArgs<Tret> &args) {
    /* Determine if the method is valid by attempting to obtain an implementation specifying this method. */
    GemmConfig       cfg(method);
    GemmArgs<Tret>   myargs = args;

    myargs._cfg = &cfg;

    const GemmImplementation<Top, Tret> *impl;

    return find_implementation<Top, Tret>(myargs, impl);
}

} // namespace arm_gemm