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

#include "gemv_batched.hpp"

namespace arm_gemm {

template<typename Top, typename Tret>
class GemmImplementation {
public:
    /* Is this implementation compatible with the args as provided? */
    virtual bool is_supported(const GemmArgs<Tret> &args)   { return true; }
    /* Is this implementation "recommended" for these args (heuristic)? */
    virtual bool is_recommended(const GemmArgs<Tret> &args) { return true; }
    /* Instantiate this method please. */
    virtual UniqueGemmCommon<Top, Tret> instantiate(const GemmArgs<Tret> &args) = 0;

    /* Indicate the "GemmMethod" for use as a selector */
    const GemmMethod method;

    virtual ~GemmImplementation() { }

    GemmImplementation(GemmMethod method) : method(method) { }
};

/* "gemv_batched" implementation is type-agnostic, so template it here. */
template<typename Top, typename Tret>
class GemmImpl_gemv_batched : public GemmImplementation<Top, Tret> {
public:
    bool is_supported(const GemmArgs<Tret> &args) override {
        return (args._Msize==1 && args._nbatches > 1);
    }

    UniqueGemmCommon<Top, Tret> instantiate(const GemmArgs<Tret> &args) override {
        return UniqueGemmCommon<Top, Tret> (new GemvBatched<Top, Tret>(args));
    }

    GemmImpl_gemv_batched() : GemmImplementation<Top, Tret>(GemmMethod::GEMV_BATCHED) { }
};

/* "Master" function implemented for each valid combination of types.
 * Returns a list of GEMM implementation descriptors for processing by the
 * other functions.  */
template<typename Top, typename Tret>
std::vector<GemmImplementation<Top, Tret> *> &gemm_implementation_list();

template<typename Top, typename Tret>
GemmImplementation<Top, Tret> *find_implementation(GemmArgs<Tret> &args, GemmConfig *cfg) {
    auto gemms = gemm_implementation_list<Top, Tret>();

    for(auto &&i : gemms) {
        /* Skip if this implementation doesn't support these args. */
        if (!i->is_supported(args)) {
            continue;
        }

        /* Skip if a specific method is requested and this is a different one. */
        if (cfg && cfg->method != GemmMethod::DEFAULT && i->method != cfg->method) {
            continue;
        }

        /* If no specific method is requested, check that this method recommends itself. */
        if ((!cfg || cfg->method == GemmMethod::DEFAULT) && !i->is_recommended(args)) {
            continue;
        }

        return i;
    }

    return nullptr;
}

template<typename Top, typename Tret>
UniqueGemmCommon<Top, Tret> gemm(GemmArgs<Tret> &args, GemmConfig *cfg) {
    auto impl = find_implementation<Top, Tret>(args, cfg);

    if (impl) {
        return impl->instantiate(args);
    }

    return UniqueGemmCommon<Top, Tret>(nullptr);
}

template<typename Top, typename Tret>
GemmMethod get_gemm_method(GemmArgs<Tret> &args) {
    auto impl = find_implementation<Top, Tret>(args, nullptr);

    if (impl) {
        return impl->method;
    }

    /* This shouldn't happen - there should always be at least one valid implementation. */
    return GemmMethod::DEFAULT;
}

template<typename Top, typename Tret>
bool method_is_compatible(GemmMethod method, GemmArgs<Tret> &args) {
    /* Determine if the method is valid by attempting to obtain an implementation specifying this method. */
    GemmConfig cfg(method);

    auto impl = find_implementation<Top, Tret>(args, &cfg);

    if (impl) {
        return true;
    }

    return false;
}

} // namespace arm_gemm
