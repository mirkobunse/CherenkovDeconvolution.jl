from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()
with open('LICENSE.md') as f:
    license = f.read()

setup(
    name='CherenkovDeconvolution_jl',
    version='0.0.1',
    description='Python wrapper for CherenkovDeconvolution.jl',
    long_description=readme,
    author='Mirko Bunse',
    author_email='mirko.bunse@cs.tu-dortmund.de',
    url='https://github.com/mirkobunse/CherenkovDeconvolution.jl',
    license=license,
    packages=find_packages(exclude=('tests', 'docs')),
    tests_require=['pytest'],
    setup_requires=['pytest-runner'],
    install_requires=[
        'julia >= 0.5.6'
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Machine Learning'
    ]
)
