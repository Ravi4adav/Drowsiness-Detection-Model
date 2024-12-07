from setuptools import find_packages, setup 
from typing import List


HYPON_E_DOT = '-e .'
def get_requirements(file_path: str) -> List[str]:
    
    """This function will return the list of elements"""

    requirements=[]
    with open(file_path) as f:
        requirements=f.readlines()
        requirements=[req.replace('\n','') for req in requirements]

        if HYPON_E_DOT in requirements:
            requirements.remove(HYPON_E_DOT)
    
    return requirements


setup(
    name='DDR',
    author='Ravi Yadav',
    author_email='ry794396@gmail.com',
    version='1.0.0',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)