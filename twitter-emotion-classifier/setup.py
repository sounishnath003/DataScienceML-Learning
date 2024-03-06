
from setuptools import find_packages, setup


setup(
    name='trainer',
    version='v1.0.0',
    description='a simple twitter emotion dataset classifier',
    author='Sounish Nath',
    author_email='sounish.nath@gmail.com',
    packages=find_packages(exclude=['logs', 'github', '.workflows']),
    install_requires=[
        open('requirements.txt', 'r+').read().split("\n")
    ],
    license='BSD',
    keywords="sounish twitter-emotion kaggle lightningai pytorch deeplearning aiml ml",
)

