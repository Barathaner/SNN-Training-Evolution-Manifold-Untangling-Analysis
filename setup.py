# setup.py
from setuptools import setup, find_packages

setup(
    # Metadaten
    name='snn-manifold-analysis',
    version='0.1.0',
    description='The learning process in Spiking Neural Networks can be geometrically explained as the untangling of the layer manifolds over epochs into linearly separable classes. This library provides a pipeline and an example to analyse your spiking data from the input and your snntorch model. It is based on Manifold Capacity, and dimensionalityreduction (UMAP)',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Barathaner/SNN-Training-Evolution-Manifold-Untangling-Analysis',
    author='Karl-Augustin Jahnel (Barathaner)', 
    license='MIT',
    install_requires=open('requirements.txt').read().splitlines(),
    # Der wichtigste Teil: Code-Struktur
    # find_packages() sucht automatisch nach Ordnern mit __init__.py
    packages=find_packages(),


    # Klassifikatoren helfen Nutzern, das Paket zu finden
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence :: Spiking Neural Networks',
    ],
    python_requires='>=3.12',
)