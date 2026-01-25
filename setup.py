import os
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11


class BuildExt(build_ext):
    c_opts = {
        "msvc": ["/O2", "/std:c++17"],
        "unix": ["-O3", "-DNDEBUG", "-funroll-loops"],
    }


    l_opts = {
        "msvc": [],
        "unix": [],
    }

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, []).copy()
        link_opts = self.l_opts.get(ct, []).copy()

        if ct == "unix":
            opts.append("-std=c++17")
            opts.append("-fPIC")

        aggressive = os.environ.get("AGGRESSIVE", "").lower() in ("1", "true", "yes")

        if aggressive and ct == "unix":
            opts += [
                "-march=core-avx-i",
                "-mtune=generic",
                "-flto",
            ]
            link_opts += [
                "-flto",
                "-lm",
            ]
            print("** Building PRAXIS with additional flags")
        elif aggressive and ct != "unix":
            print("Non-unix compiler detected; using safe flags.")

        for ext in self.extensions:
            ext.extra_compile_args = opts
            ext.extra_link_args = link_opts

        build_ext.build_extensions(self)


ext_modules = [
    Extension(
        "praxis._core",
        sources=[
            "src/praxis/_core.cpp",
            # _core.cpp includes relevant things in cpp folder anyway
        ],
        include_dirs=[
            pybind11.get_include(),
            "src/praxis/cpp",
        ],
        language="c++",
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExt},
)
