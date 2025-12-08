#
# SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

"""Build definitions for KleidiAI"""

load(
    "@bazel_tools//tools/build_defs/repo:utils.bzl",
    "update_attrs",
    "workspace_and_buildfile",
)

# Extra warnings for GCC/CLANG C/C++
def kai_gcc_warn_copts():
    return [
        "-Wall",
        "-Wdisabled-optimization",
        "-Wextra",
        "-Wformat-security",
        "-Wformat=2",
        "-Winit-self",
        "-Wstrict-overflow=2",
        "-Wswitch-default",
        "-Wno-vla",
        "-Wcast-qual",
    ]

# GCC/CLANG C only warning options
def kai_gcc_warn_conlyopts():
    return [
        "-Wmissing-prototypes",
        "-Wstrict-prototypes",
        "-Wpedantic",
    ]

# GCC/CLANG C++ only warning options
def kai_gcc_warn_cxxopts():
    return [
        "-Wctor-dtor-privacy",
        "-Woverloaded-virtual",
        "-Wsign-promo",
        "-Wmissing-declarations",
    ]

# GCC/CLANG compiler options
def kai_gcc_std_copts():
    return ["-std=c99"] + kai_gcc_warn_copts() + kai_gcc_warn_conlyopts()

# GCC/CLANG compiler options
def kai_gcc_std_cxxopts():
    return ["-std=c++17"] + kai_gcc_warn_copts() + kai_gcc_warn_cxxopts()

def kai_cpu_select(cpu_uarch):
    if len(cpu_uarch) == 0:
        return "armv8-a"
    else:
        return "armv8.2-a+" + "+".join(cpu_uarch)

def kai_cpu_i8mm():
    return ["i8mm"]

def kai_cpu_dotprod():
    return ["dotprod"]

def kai_cpu_bf16():
    return ["bf16"]

def kai_cpu_fp16():
    return ["fp16"]

def kai_cpu_neon():
    return []

def kai_cpu_sve():
    return ["sve"]

def kai_cpu_sve2():
    return ["sve2"]

def kai_cpu_sme():
    return ["sme"]

def kai_cpu_sme2():
    return ["sme2"]

# MSVC compiler options
def kai_msvc_std_copts():
    return ["/Wall"]

def kai_msvc_std_cxxopts():
    return ["/Wall"]

def kai_copts(ua_variant):
    return select({
        "//:windows": kai_msvc_std_copts(),
        # Assume default to use GCC/CLANG compilers. This is a fallback case to make it
        # easier for KleidiAI library users
        "//conditions:default": kai_gcc_std_copts() + ["-march=" + kai_cpu_select(ua_variant)],
    })

def kai_cxxopts(ua_variant):
    return select({
        "//:windows": kai_msvc_std_cxxopts(),
        # Assume default to use GCC/CLANG compilers. This is a fallback case to make it
        # easier for KleidiAI library users
        "//conditions:default": kai_gcc_std_cxxopts() + ["-march=" + kai_cpu_select(ua_variant)],
    })

def _kai_list_check(predicate, sub_list, super_list):
    """ Allow to check of any or all elements of first list are in second one

    Args:
        predicate (function): predicate to check. For example 'all' or 'any'
        sub_list (list): first list
        super_list (list): second list
    """
    return predicate([item in super_list for item in sub_list])

def _kai_c_cxx_common(name, copts_def_func, **kwargs):
    """Common C/C++ native cc_library wrapper with custom parameters and defaults

    Args:
        name (string): name of target library
        copts_def_func (function): function to get C or C++ respective defaults
        **kwargs (dict): other arguments like srcs, hdrs, deps
    """

    # Convert CPU uarch to list of features
    cpu_uarch = kwargs.get("cpu_uarch", kai_cpu_neon())
    extra_copts = []

    # Indicate if SME flags should be replaced since a toolchain may not support it
    replace_sme_flags = _kai_list_check(any, kai_cpu_sme() + kai_cpu_sme2(), cpu_uarch)

    if replace_sme_flags:
        if _kai_list_check(all, kai_cpu_sme(), cpu_uarch):
            for uarch in kai_cpu_sme():
                cpu_uarch.remove(uarch)

        if _kai_list_check(all, kai_cpu_sme2(), cpu_uarch):
            for uarch in kai_cpu_sme2():
                cpu_uarch.remove(uarch)

        # Replace SME/SME2 with SVE+SVE2, but disable compiler vectorization
        cpu_uarch.extend(kai_cpu_sve())
        cpu_uarch.extend(kai_cpu_sve2())

        extra_copts.append("-fno-tree-vectorize")

    kwargs["copts"] = kwargs.get("copts", []) + copts_def_func(cpu_uarch) + extra_copts
    kwargs["deps"] = ["//:common"] + kwargs.get("deps", [])
    kwargs["linkstatic"] = kwargs.get("linkstatic", True)

    # Remove custom cpu_uarch paramter before passing it to cc_library
    if "cpu_uarch" in kwargs:
        kwargs.pop("cpu_uarch")

    # Add kernels source files
    if "kernels" in kwargs:
        kwargs["srcs"] = kwargs.get("srcs", []) + [ukernel + ".c" for ukernel in kwargs["kernels"]]
        kwargs["textual_hdrs"] = kwargs.get("textual_hdrs", []) + [ukernel + ".h" for ukernel in kwargs["kernels"]]
        kwargs.pop("kernels")

    # Add assembly kernels source files
    if "kernels_asm" in kwargs:
        kwargs["srcs"] = kwargs.get("srcs", []) + [ukernel + "_asm.S" for ukernel in kwargs["kernels_asm"]] + [ukernel + ".c" for ukernel in kwargs["kernels_asm"]]
        kwargs["textual_hdrs"] = kwargs.get("textual_hdrs", []) + [ukernel + ".h" for ukernel in kwargs["kernels_asm"]]
        kwargs.pop("kernels_asm")

    native.cc_library(
        name = name,
        **kwargs
    )

def kai_c_library(name, **kwargs):
    """C native cc_library wrapper with custom parameters and defaults

    Args:
        name (string): name of target library
        **kwargs (dict): other arguments like srcs, hdrs, deps
    """
    _kai_c_cxx_common(name, kai_copts, **kwargs)

def kai_cxx_library(name, **kwargs):
    """C++ native cc_library wrapper with custom parameters and defaults

    Args:
        name (string): name of target library
        **kwargs (dict): other arguments like srcs, hdrs, deps
    """
    _kai_c_cxx_common(name, kai_cxxopts, **kwargs)

def _kai_local_archive_impl(ctx):
    """Implementation of the kai_local_archive rule."""
    ctx.extract(
        ctx.attr.archive,
        stripPrefix = ctx.attr.strip_prefix,
    )
    workspace_and_buildfile(ctx)

    return update_attrs(ctx.attr, _kai_local_archive_attrs.keys(), {})

_kai_local_archive_attrs = {
    "archive": attr.label(mandatory = True, allow_single_file = True, doc = "Path to local archive relative to workspace"),
    "strip_prefix": attr.string(doc = "Strip prefix from archive internal content"),
    "build_file": attr.label(allow_single_file = True, doc = "Name of BUILD file for extracted repository"),
    "build_file_content": attr.string(doc = "Content of BUILD file for extracted repository"),
    "workspace_file": attr.label(doc = "Name of WORKSPACE file for extracted repository"),
    "workspace_file_content": attr.string(doc = "Content of WORKSPACE file for extracted repository"),
}

kai_local_archive = repository_rule(
    implementation = _kai_local_archive_impl,
    attrs = _kai_local_archive_attrs,
    local = True,
    doc = "Rule to use repository from compressed local archive",
)
