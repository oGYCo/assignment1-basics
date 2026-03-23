from pathlib import Path

from Cython.Build import cythonize
from setuptools import Extension, setup


ROOT = Path(__file__).parent


setup(
    name="train_bpe_ext",
    packages=[],
    py_modules=[],
    ext_modules=cythonize(
        [
            Extension(
                "cs336_basics._train_bpe_cython",
                [str(ROOT / "cs336_basics" / "_train_bpe_cython.pyx")],
            )
        ],
        compiler_directives={
            "language_level": 3,
            "boundscheck": False,
            "wraparound": False,
            "initializedcheck": False,
            "nonecheck": False,
            "infer_types": True,
        },
    ),
    script_args=["build_ext", "--inplace"],
    zip_safe=False,
)
