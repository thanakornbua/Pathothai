from setuptools import setup, find_packages

setup(
    name='large-image-viewer',
    version='0.1',
    description='A large image viewing program that handles 1GB PNG or JPG files with channel manipulation features.',
    author='Your Name',
    author_email='your.email@example.com',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'Pillow',
        'PyQt5',
        'numpy',
        'opencv-python'
    ],
    entry_points={
        'console_scripts': [
            'large-image-viewer=main:main',
        ],
    },
    python_requires='>=3.6',
)