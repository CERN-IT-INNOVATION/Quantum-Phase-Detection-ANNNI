from setuptools import setup, find_packages

setup(
    name="PhaseEstimation",
    version="0.1",
    description="Package Built on top of Pennylane for Phase Mapping of the ANNNI Model",
    url="https://github.com/SaverioMonaco/Phase-Estimation-through-QML",
    author="Saverio Monaco",
    author_email="saveriomonaco97@gmail.com",
    license="MIT",
    packages=find_packages(),
    install_requires=[
    	"ipykernel==6.15.0",
    	"joblib==1.1.0",
    	"plotly==5.8.2",
    	"matplotlib==3.5.2",
        "PennyLane==0.23.1",
        "numpy",
        "tqdm==4.64.0",
        "jax==0.3.13",
        "jaxlib==0.3.10",
    ],
    zip_safe=False,
)
