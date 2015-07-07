#!/usr/bin/env python

from setuptools import find_packages, setup

VERSION = '0.6'

if __name__ == '__main__':
    setup(
        name = 'saloon',
        packages = find_packages(),
        version = VERSION,
        description = 'Multi-armed bandit library',
        author = 'Derek Bennewies',
        author_email = 'derekrb@gmail.com',
        license = 'MIT',
        url = 'https://github.com/derekrb/saloon',
        download_url = 'https://github.com/derekrb/saloon/tarball/{}'.format(VERSION),
        keywords = ['bandit', 'mab', 'multi-armed bandit', 'ab test'],
        install_requires=['numpy', 'psycopg2']
    )
