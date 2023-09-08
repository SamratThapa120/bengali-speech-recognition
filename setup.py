from setuptools import setup, find_packages

setup(
    name='bengali_asr',
    version='0.1',
    description='Bengali Automatic Speech Recognition',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/SamratThapa120/bengali-speech-recognition',
    packages=find_packages(),
    install_requires=[],
    keywords='asr bengali speech-recognition',
    package_data={
        'bengali_asr': ['audio/assets/*'],
    },
    include_package_data=True,
)

