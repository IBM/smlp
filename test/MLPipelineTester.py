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
import argparse

from smlp import MLPipeline
from smlp import MLPipelineParams
from smlp import MLPipelineUtil

if __name__ == '__main__' :
    parser = argparse.ArgumentParser(
        description='ML Pipeline Tester',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    MLPipelineUtil.AddCommonParserArgs(parser)

    args = parser.parse_args()
    args_dict = vars(args)

    mlp_params=MLPipelineParams(**args_dict)
    mlp = MLPipeline(mlp_params)

    models = mlp.run_pipeline()
