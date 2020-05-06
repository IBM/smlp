'''
Copyright 2020, IBM CORP.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Authors      : Nikolas Ioannou,

End Copyright
'''
from setuptools import setup, find_packages

setup(name='smlp',
      version='1.0',
      packages=find_packages(),
      license='Apache License 2.0',
      python_requires='>=3.7',
      package_data={'': ['test/data/*.csv']})
