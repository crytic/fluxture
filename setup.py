from setuptools import setup, find_packages


setup(
    name='blockscraper',
    description='A scraper for blockchains',
    url='https://github.com/trailofbits/blockscraper',
    author='Trail of Bits',
    version='0.0.1',
    packages=find_packages(),
    python_requires='>=3.7',
    install_requires=[
    ],
    entry_points={
        'console_scripts': [
            'blockscraper = blockscraper.__main__:main'
        ]
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Utilities'
    ]
)
