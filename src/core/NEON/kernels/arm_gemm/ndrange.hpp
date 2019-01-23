/*
 * Copyright (c) 2019 Arm Limited.
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
#pragma once

#include <algorithm>
#include <initializer_list>

namespace arm_gemm {

template<unsigned int D>
class NDRange {
private:
    unsigned int m_sizes[D];
    unsigned int m_totalsizes[D];

    class NDRangeIterator {
    private:
        const NDRange &m_parent;
        unsigned int m_pos = 0;
        unsigned int m_end = 0;

    public:
        NDRangeIterator(const NDRange &p, unsigned int s, unsigned int e) : m_parent(p), m_pos(s), m_end(e) { }

        bool done() const {
            return (m_pos >= m_end);
        }

        unsigned int dim(unsigned int d) const {
            unsigned int r = m_pos;

            if (d < (D - 1)) {
                r %= m_parent.m_totalsizes[d];
            }

            if (d > 0) {
                r /= m_parent.m_totalsizes[d-1];
            }

            return r;
        }

        bool next_dim0() {
            m_pos++;

            return !done();
        }

        bool next_dim1() {
            m_pos += m_parent.m_sizes[0] - dim(0);

            return !done();
        }

        unsigned int dim0_max() const {
            unsigned int offset = std::min(m_end - m_pos, m_parent.m_sizes[0] - dim(0));

            return dim(0) + offset;
        }
    };

public:
    template <typename... T>
    NDRange(T... ts) : m_sizes{ts...} {
        unsigned int t=1;

        for (unsigned int i=0; i<D; i++) {
            t *= m_sizes[i];

            m_totalsizes[i] = t;
        }
    }

    NDRangeIterator iterator(unsigned int start, unsigned int end) const {
        return NDRangeIterator(*this, start, end);
    }

    unsigned int total_size() const {
        return m_totalsizes[D - 1];
    }

    unsigned int get_size(unsigned int v) const {
        return m_sizes[v];
    }
};

} // namespace arm_gemm
