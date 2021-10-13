import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gbtsnowcast",
    version="0.1.0",
    author="Jussi Leinonen",
    author_email="jussi.leinonen@meteoswiss.ch",
    description="Code package of 'Nowcasting thunderstorm hazards using machine learning: the impact of data sources on performance'",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/meteoswiss-mdr/ts-nowcast-datasources",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'lightgbm',
        'sklearn',
        'netCDF4'
    ]
)
