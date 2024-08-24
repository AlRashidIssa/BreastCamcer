from setuptools import setup, find_packages

setup(
    name='breast-cancer-pipeline',
    version='1.0.0',
    author='Al Rashid Issa',
    author_email='alrashidissa2001@hotmail.com',
    description='A comprehensive pipeline for processing breast cancer data, training a machine learning model, and evaluating predictions.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/AlRashidIssa/BreastCancer',
    packages=find_packages(include=['src', 'src.*']),
    install_requires=[
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'scikit-learn>=0.24.0',
        'matplotlib>=3.4.0',
        'seaborn>=0.11.0',
        'mlflow>=1.20.0',
        'django>=3.2.0'
    ],
    extras_require={
        'dev': [
            'pytest>=6.2.0',
            'flake8>=3.9.0'
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Creative Commons Attribution-NonCommercial 4.0 International License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
    entry_points={
        'console_scripts': [
            'run-pipeline=src.cli.run:main',  # 
            'start-api=src.cli.start_api:main' 
        ],
    },
)
