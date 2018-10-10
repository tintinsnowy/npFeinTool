import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="npFeintool",
    version="0.1.1",
    author="Sherry Yang",
    author_email="yangxiaoli1994@gmail.com",
    description="npFeintool is used for the time series data processing",
    long_description=long_description,
    #long_description_content_type="text/markdown",
    url="https://github.com/Senseering/npFeinTool",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Unix"
    ),
)
