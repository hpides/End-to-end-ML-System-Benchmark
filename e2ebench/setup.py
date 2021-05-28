from setuptools import setup, find_packages

setup(
    name='e2ebench',
    version='0.1.0',
    description='End-to-end machine learning benchmark',
    author='Willi Rieck, Christian Jacob, Jonas Schulze, Jost Morgenstern',
    packages=find_packages(include=['e2ebench']),
    install_requires=[
        'matplotlib',
        'numpy',
        'pandas',
        'plotly',
        'psutil',
        # 'pyRAPL',
        'seaborn',
        'scikit-learn',
        'sqlalchemy',
        'PyInquirer'
    ],
    entry_points = {
        'console_scripts': ['e2ebench-cli=e2ebench.visualization_cli:main'],
    }
)