from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")


setup(
    name="zplane",
    version="0.1.2",
    description="Wrapper for scipy's signal module, simplifying a handful of commonly used, discrete system plots.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Attrup/zplane",
    author="Jonas Attrup",
    author_email="attrup.jonas@gmail.com",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering"
    ],
    keywords="matplotlib, scipy, digital signal processing, discrete, bode, polezero, impulse, frequency",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10, <4",
    install_requires=["matplotlib", "scipy", "numpy"],
)