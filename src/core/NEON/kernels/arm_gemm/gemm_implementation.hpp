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

/* Structure describing an implementation.  For each supported combination
 * of types, a static list of these structures is built up to describe the
 * implementations available.
 */
template<typename Top, typename Tret, class OutputStage = Nothing>
struct GemmImplementation {
    const GemmMethod                                                                     method;
    const char *                                                                         name;
    std::function<bool(const GemmArgs<Tret> &, const OutputStage &)>                     is_supported;
    std::function<bool(const GemmArgs<Tret> &, const OutputStage &)>                     is_recommended;
    std::function<GemmCommon<Top, Tret> *(const GemmArgs<Tret> &, const OutputStage &)>  instantiate;

    bool do_is_supported(const GemmArgs<Tret> &args, const OutputStage &os) const {
        if (is_supported != nullptr) {
            return is_supported(args, os);
        } else {
            return true;
        }
    }

    bool do_is_recommended(const GemmArgs<Tret> &args, const OutputStage &os) const {
        if (is_recommended != nullptr) {
            return is_recommended(args, os);
        } else {
            return true;
        }
    }

    GemmCommon<Top, Tret> *do_instantiate(const GemmArgs<Tret> &args, const OutputStage &os) const {
        return instantiate(args, os);
    }
};

/* Slightly different version of above for straightforward GEMMs with no
 * output stage, so the std::functions there don't have to deal with the
 * unnecessary second argument.  */
template<typename Top, typename Tret>
struct GemmImplementation<Top, Tret, Nothing> {
    const GemmMethod                                               method;
    const char *                                                   name;
    std::function<bool(const GemmArgs<Tret> &)>                    is_supported;
    std::function<bool(const GemmArgs<Tret> &)>                    is_recommended;
    std::function<GemmCommon<Top, Tret> *(const GemmArgs<Tret> &)> instantiate;

    bool do_is_supported(const GemmArgs<Tret> &args, const Nothing &) const {
        if (is_supported != nullptr) {
            return is_supported(args);
        } else {
            return true;
        }
    }

    bool do_is_recommended(const GemmArgs<Tret> &args, const Nothing &) const {
        if (is_recommended != nullptr) {
            return is_recommended(args);
        } else {
            return true;
        }
    }

    GemmCommon<Top, Tret> *do_instantiate(const GemmArgs<Tret> &args, const Nothing &) const {
        return instantiate(args);
    }
};

/* "Master" function implemented for each valid combination of types.
 * Returns a list of GEMM implementation descriptors for processing by the
 * other functions, terminated by an implementation with
 * method==GemmMethod::DEFAULT.  */
template<typename Top, typename Tret, class OutputStage = Nothing>
const GemmImplementation<Top, Tret, OutputStage> *gemm_implementation_list();

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
template<typename Top, typename Tret, class OutputStage>
bool find_implementation(const GemmArgs<Tret> &args, const OutputStage &os, const GemmImplementation<Top, Tret, OutputStage> * &impl) {
    auto gemms = gemm_implementation_list<Top, Tret, OutputStage>();
    const GemmConfig *cfg = args._cfg;

    const GemmImplementation<Top, Tret, OutputStage> *saved_impl = nullptr;

    for (const GemmImplementation<Top, Tret, OutputStage> *i = gemms; i->method != GemmMethod::DEFAULT; i++) {
        /* Skip if this implementation doesn't support these args. */
        if (!i->do_is_supported(args, os)) {
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
        if (!i->do_is_recommended(args, os)) {
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

template<typename Top, typename Tret, class OutputStage>
std::vector<KernelDescription> get_compatible_kernels(const GemmArgs<Tret> &args, const OutputStage &os) {
    std::vector<KernelDescription> res;

    /* Find out what the default implementation in so we can set the flag accordingly later. */
    const GemmImplementation<Top, Tret, OutputStage> *default_impl;
    find_implementation(args, os, default_impl);

    auto gemms = gemm_implementation_list<Top, Tret, OutputStage>();

    for (const GemmImplementation<Top, Tret, OutputStage> *i = gemms; i->method != GemmMethod::DEFAULT; i++) {
        /* Check that this implementation supports the presented problem. */
        if (!i->do_is_supported(args, os)) {
            continue;
        }

        res.push_back(KernelDescription(i->method, i->name, i==default_impl));
    }

    return res;
}

template<typename Top, typename Tret, class OutputStage>
UniqueGemmCommon<Top, Tret> gemm(const GemmArgs<Tret> &args, const OutputStage &os) {
    const GemmImplementation<Top, Tret, OutputStage> *impl;

    if (find_implementation<Top, Tret, OutputStage>(args, os, impl)) {
        return UniqueGemmCommon<Top, Tret>(impl->do_instantiate(args, os));
    }

    return UniqueGemmCommon<Top, Tret>(nullptr);
}

template<typename Top, typename Tret, class OutputStage>
KernelDescription get_gemm_method(const GemmArgs<Tret> &args, const OutputStage &os) {
    const GemmImplementation<Top, Tret, OutputStage> *impl;

    if (find_implementation<Top, Tret>(args, os, impl)) {
        return KernelDescription(impl->method, impl->name);
    }

    /* This shouldn't happen - there should always be at least one valid implementation. */
    return KernelDescription();
}

} // namespace arm_gemm
