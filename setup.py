from setuptools import find_packages,setup

setup(
    name='Emotion_Detector',
    version='0.0.1',
    author='Shivam Kumar',
    author_email='shivamkumarsingh1064@gmail.com',
    install_requires=["matplotlib","numpy","pandas","seaborn","tensorflow","keras","opencv-python"],
    packages=find_packages()
)