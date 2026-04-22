from setuptools import find_packages, setup

setup(
    name="sd-lora-attention-segmentation",
    version="0.1.0",
    description="LoRA training, generation, evaluation, and attention segmentation toolkit",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[],
    python_requires=">=3.9",
)
