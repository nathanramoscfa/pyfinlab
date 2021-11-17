from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
    name='pyfinlab',  # Required
    version='0.0.27',  # Required
    description='Financial applications for portfolio management',  # Optional
    long_description=long_description,  # Optional
    long_description_content_type='text/markdown',  # Optional (see note above)
    url='https://github.com/nathanramoscfa/pyfinlab',  # Optional
    author='Nathan Ramos, CFA©',  # Optional
    author_email='nathan.ramos.cfa@gmail.com',  # Optional
    classifiers=[  # Optional
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Financial and Insurance Industry',
        'Intended Audience :: Science/Research',
        'Topic :: Office/Business :: Financial :: Investment',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
    ],
    keywords='python, finance, portfolio-optimization, quantitative-finance, portfolio-management',  # Optional
    package_dir={'': 'src'},  # Optional
    packages=find_packages(where='src'),  # Required
    python_requires='>=3.7',
    project_urls={  # Optional
        'Bug Reports': 'https://github.com/nathanramoscfa/pyfinlab/issues',
        'Source': 'https://github.com/nathanramoscfa/pyfinlab',
    },
)
