import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="iwin",
    version="2.0.0.dev1",
    author="Ernesto Giron Echeverry",
    author_email="e.giron.e@gmail.com",
    description="Library for analyzing IWIN dataset",
    keywords="IWIN, META, wheat, iPAR, photoperiod, RUE, NDVI, BLUEs, BLUPs, Heritability, crop modeling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/egiron/CIMMYT/IWIN",
    download_url = "https://github.com/egiron/CIMMYT/tree/1004057bf06204120455702e1adef2ff0abb71b9/IWIN",
    packages=setuptools.find_packages(),
    include_package_data=True,
    # packages=['iparyield'],
    # scripts=['bin/iparyield'],
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: Free for non-commercial use",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Development Status :: 5 - Production/Stable",
        "Environment :: Web Environment",
        "Environment :: Console",
        "Framework :: Jupyter",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: End Users/Desktop",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Natural Language :: Spanish",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Machine Learning"
        "Topic :: Scientific/Engineering :: GIS"
        "Topic :: Scientific/Engineering :: Agriculture"
    ],
)