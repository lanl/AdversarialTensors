from setuptools import setup, find_packages


__version__ = "1.0"



# add readme
with open('README.md', 'r') as f:
    LONG_DESCRIPTION = f.read()

# add dependencies
with open('requirements.txt', 'r') as f:
    INSTALL_REQUIRES = f.read().strip().split('\n')

setup(
    name='AdversarialTensors',
    version=__version__,
    author='Manish Bhattarai, Mehmet Cagri, Ben Nebgen, Kim Rasmussen, Boian Alexandrov',
    author_email='ceodspspectrum@lanl.gov',
    long_description=LONG_DESCRIPTION,
    url='https://github.com/lanl/AdversarialTensors',  # change this to GitHub once published
    description='AdversarialTensors: Tensors-based framework for adversarial robustness',
    setup_requires=['numpy', 'scipy', 'matplotlib', 'torch', 'torchvision', 'tensorly','scikit-learn', 'pytest'],
    install_requires=INSTALL_REQUIRES,
    packages=find_packages(),
    python_requires='>=3.9',
    classifiers=[
        'Development Status :: ' + str(__version__) + ' - Beta',
        'Programming Language :: Python :: 3.9',
        'Topic :: Machine Learning :: Libraries'
    ],
    license='License :: BSD3 License',
    zip_safe=False,
    dependency_links=[
        'git+https://github.com/fra31/auto-attack.git#egg=auto-attack'
    ]
)