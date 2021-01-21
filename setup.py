from setuptools import setup, find_packages


setup(
    name='fluxture',
    description='A crawling framework for blockchains and peer-to-peer systems',
    url='https://github.com/trailofbits/fluxture',
    author='Trail of Bits',
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    python_requires='>=3.7',
    install_requires=[
        'fastkml~=0.11',
        'geoip2~=4.1.0',
        'graphviz~=0.14.1',
        'great-circle-calculator~=1.1.0',
        'lxml~=4.6.2',
        'networkx~=2.4',
        'shapely~=1.7.1',
        'six>=1.5',
        'tqdm~=4.48.0',
    ],
    entry_points={
        'console_scripts': [
            'fluxture = fluxture.__main__:main'
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
