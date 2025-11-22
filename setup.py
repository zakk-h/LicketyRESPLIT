import os
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11


class BuildExt(build_ext):
    c_opts = {
        "msvc": ["/O2"],
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

        # optional fast mode via env var
        fast = os.environ.get("LICKETY_FAST", "").lower() in ("1", "true", "yes")

        if fast and ct == "unix":
            opts += [
                "-march=core-avx-i",
                "-mtune=generic",
                "-flto",
            ]
            link_opts += [
                "-flto",
                "-lm",
            ]
            print("** Building LicketyRESPLIT in FAST mode (LICKETY_FAST=1) **")
        elif fast and ct != "unix":
            print("WARNING: LICKETY_FAST is set, but non-unix compiler detected; ignoring fast flags.")

        for ext in self.extensions:
            ext.extra_compile_args = opts
            ext.extra_link_args = link_opts

        build_ext.build_extensions(self)


ext_modules = [
    Extension(
        "licketyresplit._core",
        sources=[
            "src/licketyresplit/_core.cpp",
            # _core.cpp includes cpp/lickety_resplit.cpp directly
        ],
        include_dirs=[
            pybind11.get_include(),
            "src/licketyresplit/cpp",
        ],
        language="c++",
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExt},
)
