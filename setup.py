from setuptools import setup, find_packages

setup(
    name="stress_monitor",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas>=2.0.3",
        "matplotlib>=3.7.2",
        "seaborn>=0.12.2",
        "scikit-learn>=1.3.0",
        "streamlit>=1.28.0",
        "pydantic>=2.5.0",
        "rich>=13.7.0",
        "catboost>=1.2.2",
        "plotly>=5.17.0",
    ],
    python_requires=">=3.8",
)