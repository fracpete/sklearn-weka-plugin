from setuptools import setup, find_namespace_packages


def _read(f):
    """
    Reads in the content of the file.
    :param f: the file to read
    :type f: str
    :return: the content
    :rtype: str
    """
    return open(f, 'rb').read()


setup(
    name="scikit-weka",
    description="Library for making Weka algorithms available within scikit-learn. Relies on the python-weka-wrapper3 library.",
    long_description=(
        _read('DESCRIPTION.rst') + b'\n' +
        _read('CHANGES.rst')).decode('utf-8'),
    url="https://github.com/fracpete/scikit-weka",
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: GNU General Public License (GPL)',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Programming Language :: Python :: 3',
    ],
    license='GNU General Public License version 3.0 (GPLv3)',
    package_dir={
        '': 'src'
    },
    packages=find_namespace_packages(where='src'),
    namespace_packages=[
        "ufdl",
    ],
    version="0.0.1",
    author='Peter "fracpete" Reutemann',
    author_email='scikit-weka@fracpete.org',
    install_requires=[
        "numpy",
        "python-weka-wrapper3",
        "sklearn",
    ],
)