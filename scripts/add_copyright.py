#!/usr/bin/env python3
import glob
import os.path
import sys

mit_copyright = open("scripts/copyright_mit.txt",'r').read()

def add_cpp_copyright( f, content):
    global mit_copyright
    out = open(f,'w')
    out.write("/*\n")
    for line in mit_copyright.split('\n')[:-1]:
        out.write(" *");
        if line.strip() != "":
            out.write(" %s" %line)
        out.write("\n")
    out.write(" */\n")
    out.write(content.strip())
    out.write("\n")
    out.close()

def add_python_copyright( f, content):
    global mit_copyright
    out = open(f,'w')
    for line in mit_copyright.split('\n')[:-1]:
        out.write("#");
        if line.strip() != "":
            out.write(" %s" %line)
        out.write("\n")
    out.write(content.strip())
    out.write("\n")
    out.close()

def remove_comment( content ):
    comment=True
    out=""
    for line in content.split('\n'):
        if comment:
            if line.startswith(' */'):
                comment=False
            elif line.startswith('/*') or line.startswith(' *'):
                #print(line)
                continue
            else:
                raise Exception("ERROR: not a comment ? '%s'"% line)
        else:
            out += line + "\n"
    return out
def remove_comment_python( content ):
    comment=True
    out=""
    for line in content.split('\n'):
        if comment and line.startswith('#'):
            continue
        else:
            comment = False
            out += line + "\n"
    return out

def check_file( path ):
    root, f = os.path.split(path)
    if f in ['.clang-tidy', '.clang-format']:
        print("Skipping file: {}".format(path))
        return

    with open(path, 'r', encoding='utf-8') as fd:
        content = fd.read()
        _, extension = os.path.splitext(f)

        if extension in ['.cpp', '.h', '.hpp', '.inl', '.cl', '.in', '.cs']:
            if not content.startswith('/*'):
                add_cpp_copyright(path, content)
        elif extension == '.py' or f in ['SConstruct', 'SConscript']:
            if not content.startswith('# Copyright'):
                add_python_copyright(path, content)
        elif f == 'CMakeLists.txt':
            if not content.startswith('# Copyright'):
                add_python_copyright(path, content)
        else:
            raise Exception("Unhandled file: {}".format(path))

if len(sys.argv) > 1:
    for path in sys.argv[1:]:
        check_file(path)
else:
    for top in ['./arm_compute', './tests','./src','./examples','./utils/','./opencl-1.2-stubs/','./opengles-3.1-stubs/','./support']:
        for root, _, files in os.walk(top):
            for f in files:
                path = os.path.join(root, f)
                check_file(path)
