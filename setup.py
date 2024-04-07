from setuptools import setup


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
    name="sklearn-weka-plugin",
    description="Library for making Weka algorithms available within scikit-learn. Relies on the python-weka-wrapper3 library.",
    long_description=(
        _read('DESCRIPTION.rst') + b'\n' +
        _read('CHANGES.rst')).decode('utf-8'),
    url="https://github.com/fracpete/sklearn-weka-plugin",
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
    packages=[
        "sklweka",
    ],
    version="0.0.8",
    author='Peter "fracpete" Reutemann',
    author_email='sklweka@fracpete.org',
    install_requires=[
        "numpy",
        "python-weka-wrapper3>=0.2.5",
        "scikit-learn",
    ],
)
