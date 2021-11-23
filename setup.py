from importlib import reload
from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop

def julia_backend(project):
    print("CherenkovDeconvolution_jl (step 1/2): julia.install()")
    import julia
    julia.install()
    reload(julia) # reload required between julia.install and Main.eval
    print("CherenkovDeconvolution_jl (step 2/2): install package")
    try:
        from julia import Main
    except Exception as e:
        from julia.api import Julia # retry with compiled_modules=False
        jl = Julia(compiled_modules=False)
        from julia import Main
    Main.eval(f'import Pkg; Pkg.activate("{project}")')
    Main.eval(f'Pkg.add(url="https://github.com/mirkobunse/CherenkovDeconvolution.jl.git", rev="main")')

class JuliaBackendInstall(install):
    def run(self):
        install.run(self)
        julia_backend(self.install_lib + "CherenkovDeconvolution_jl")

class JuliaBackendDevelop(develop):
    def run(self):
        develop.run(self)
        julia_backend("CherenkovDeconvolution_jl")

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
        'julia >= 0.5.6',
        'scikit-learn >= 1'
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
    ],
    cmdclass={
        'install': JuliaBackendInstall,
        'develop': JuliaBackendDevelop
    }
)
