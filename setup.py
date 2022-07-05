import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="streamtools",
    version="0.1.dev2",
    author="Jo Bovy",
    author_email="bovy@astro.utoronto.ca",
    description="Tools for modeling and analyzing stellar streams",
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='MIT',
    packages=["streamtools","streamtools/df"],
    install_requires=['numpy>=1.7','scipy','galpy']
    )
