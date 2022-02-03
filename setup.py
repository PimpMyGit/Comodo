from setuptools import setup, find_packages
setup(
    name='comodo',
    description='VGGFace implementation with Keras framework',
    url='https://github.com/PimpMyGit/Comodo',
    author='Tommà',
    license='MIT',
    packages=find_packages(exclude=["tools", "training", "temp", "test", "data", "visualize","image",".venv",".github"]),
    zip_safe=False,
    install_requires=[
        'numpy>=1.9.1', 'scipy>=0.14', 'h5py', 'pillow', 'keras',
        'six>=1.9.0', 'pyyaml'
    ]
)