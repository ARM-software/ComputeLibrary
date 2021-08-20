/*
 * Copyright (c) 2021 Arm Limited.
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
#ifndef ARM_COMPUTE_ACL_VERSION_H_
#define ARM_COMPUTE_ACL_VERSION_H_

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/** Semantic versioning information */
typedef struct AclVersion
{
    int         major;      /**< Major version, is increased on API incompatible changes */
    int         minor;      /**< Minor version, is increased on adding back-ward compatible functionality */
    int         patch;      /**< Patch version, is increased when doing backward compatible fixes */
    const char *build_info; /**< Build related information */
} AclVersion;

/**< Major version, is increased on API incompatible changes */
#define ARM_COMPUTE_LIBRARY_VERSION_MAJOR 0
/**< Minor version, is increased on adding back-ward compatible functionality */
#define ARM_COMPUTE_LIBRARY_VERSION_MINOR 1
/**< Patch version, is increased when doing backward compatible fixes */
#define ARM_COMPUTE_LIBRARY_VERSION_PATCH 0

/** Get library's version meta-data
 *
 * @return Version information
 */
const AclVersion *AclVersionInfo();

#ifdef __cplusplus
}
#endif /* __cplusplus */
#endif /* ARM_COMPUTE_ACL_VERSION_H_ */
