from setuptools import setup, find_packages

setup(
    name='umlaut',
    version='0.1.0',
    description='End-to-end machine learning benchmark',
    author='Willi Rieck, Christian Jacob, Jonas Schulze, Jost Morgenstern',
    packages=find_packages(include=['umlaut']),
    install_requires=[
        'matplotlib',
        'numpy',
        'pandas',
        'plotly',
        'psutil',
        'pyRAPL',
        'seaborn',
        'scikit-learn',
        'sqlalchemy',
        'PyInquirer'
    ],
    entry_points = {
        'console_scripts': ['umlaut-cli=umlaut.visualization_cli:main'],
    }
)