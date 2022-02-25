from setuptools import setup

setup(
    name='cream_ai',
    version='0.0.1',
    description='cream ai package',
    url='https://github.com/CreamDeLaCream/creme-ai.git',
    author='cream',
    author_email='immutable000@gmail.com',
    license='MIT',
    packages=['dogeemotion','humanemotion'],
    zip_safe=False,
    install_requires=[
        'keras==2.8.0',
        'tensorflow==2.8.0',
        'dlib',
    ]
) 