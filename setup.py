from setuptools import setup, find_packages
from typing import List
HYPHEN_E_DOT = "-e."
def get_requirements(file_path: str) -> List[str]:
    """Read requirements file and return list of requirements."""
    requirements=[]
    with open(file_path) as file:
        requirements=file.readlines() ##[line.strip() for line in file], as the file read line by line it will have \n at the end of each line
        requirements=[req.replace("\n","") for req in requirements] ##[r.strip() for r in requirements]
    
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    return requirements

setup(
    name='mlproject',
    version='0.0.1',
   author='Saroj',
   author_email='me.sarojrai@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)