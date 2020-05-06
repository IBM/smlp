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

import os
import sys
import time
import argparse
import traceback

from smlp import MLPipelineUtil
from smlp import MLPipeline
from smlp import MLPipelineParams

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize

import numpy as np

if __name__ == '__main__' :
    parser = argparse.ArgumentParser(
        description='ML Pipeline Tester',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    MLPipelineUtil.AddCommonParserArgs(parser)

    args = parser.parse_args()
    args_dict = vars(args)

    mlp_params=MLPipelineParams(**args_dict)
    mlp = MLPipeline(mlp_params)

    try:
        t0 = time.time()
        models = mlp.run_pipeline()
        t_fit = time.time()-t0

        print ('t_fit={:.2f}s'.format(t_fit))

        args.ml_model_lst = models
        args.dataset_path=args.dataset_test_path
        args.ml_op='predict'

        args_dict = vars(args)

        mlp_params=MLPipelineParams(**args_dict)
        mlp = MLPipeline(mlp_params)

        t0 = time.time()
        score = mlp.run_pipeline()
        t_predict = time.time()-t0

        print ('test score={:.8f} t_fit={:.2f}s t_predict={:.2f}s'.format(score, t_fit, t_predict))
    except Exception:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback,
                                  limit=2, file=sys.stdout)

