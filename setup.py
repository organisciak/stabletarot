from setuptools import setup

setup(
    name='stabletarot',
    version='0.0.1',
    install_requires=[
        'diffusers', 'transformers', 'ftfy', 'Pillow>=9.0', 'openai',
    ],
)