# setup.py
from setuptools import setup, find_packages

setup(
    name="nave",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.21.0',
        'torch>=1.12.0',
        'sounddevice>=0.4.0',
        'soundfile>=0.12.0',
        'matplotlib>=3.5.0',
        'librosa>=0.8.0',
        'scipy>=1.7.0',
        'imageio>=2.10.0',         # Added for video writing
        'imageio[ffmpeg]>=2.10.0'  # Optional: often needed for mp4 support
    ],
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'nave-gui=nave.__main__:main', # Point to main function in __main__
        ],
    },
)