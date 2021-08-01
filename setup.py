# importing relevant packages
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="osl",
    version="0.0.1",
    author="Milad Sadeghi DM",
    author_email="EverLookNeverSee@ProtonMail.ch",
    description="Object Segmenting and Labeling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/EverLookNeverSee/osl",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operation System :: OS Independent",
    ],
    python_requires=">=3.8"
)
