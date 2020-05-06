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
import logging

from smlp import MLPipelineUtil
from smlp import MLPipeline
from smlp import MLPipelineParams

if __name__ == '__main__' :
    parser = argparse.ArgumentParser(
        description='ML Pipeline Tester',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    MLPipelineUtil.AddCommonParserArgs(parser)
    parser.add_argument('--verify', dest='verify', action='store_true', help='verify against Snap Booster')
    parser.add_argument('--fail_on_mismatch', dest='verify', action='store_true', help='abort if does not match')

    args = parser.parse_args()
    args_dict = vars(args)

    mlp_params=MLPipelineParams(**args_dict)
    mlp = MLPipeline(mlp_params)

    try:
        t0 = time.time()
        models = mlp.run_pipeline()
        t_fit = time.time()-t0

        args.ml_model_lst = models
        args.ml_op='predict'

        args_dict = vars(args)

        mlp_params=MLPipelineParams(**args_dict)
        mlp = MLPipeline(mlp_params)

        t0 = time.time()
        score = mlp.run_pipeline()
        t_predict = time.time()-t0

        logging.debug ('ml_params={}'.format(args.ml_opts_dict))
        logging.debug ('mlpipe test score={:.8f} t_fit={:.2f}s t_predict={:.2f}s'.format(score, t_fit, t_predict))
        if args.verify:
            from pai4sk import BoostingMachine as Booster
            import numpy as np
            bl = Booster(**args.ml_opts_dict)
            from numpy import genfromtxt

            # import pandas as pd
            t0=time.time()
            X_train = genfromtxt(args.dataset_path, delimiter=',') #pd.read_csv(args.dataset_path, delimiter=",").values
            X_test = genfromtxt(args.dataset_test_path, delimiter=',') #pd.read_csv(args.dataset_test_path, delimiter=",").values
            snap_load_time=time.time() - t0
            # ignore first row
            y_train=X_train[1:, 0]
            X_train=X_train[1:, 1:]
            y_test=X_test[1:, 0]
            X_test=X_test[1:, 1:]

            #print(X_test)
            #print(y_test)
            # data_path = os.path.abspath('/'.join(args.dataset_path.split('/')[:-1 or None]))
            # dname = args.dataset_path.split('/')[-1].split('.')[0]
            # X_train = np.load(data_path + "/{}.{}".format(dname,"X_train.npy"))
            # y_train = np.load(data_path + "/{}.{}".format(dname,"y_train.npy"))
            # X_test  = np.load(data_path + "/{}.{}".format(dname,"X_test.npy"))
            # y_test  = np.load(data_path + "/{}.{}".format(dname,"y_test.npy"))

            t0=time.time()
            bl.fit(X_train, y_train)
            snap_t_fit=time.time() - t0
            t0=time.time()
            yhat_test = bl.predict(X_test)
            snap_t_predict=time.time() - t0
            y_test[y_test==-1]=0
            from sklearn.metrics import auc, mean_squared_error, f1_score, accuracy_score, roc_auc_score, average_precision_score
            snap_score=accuracy_score(y_test, yhat_test > 0)
            logging.debug('params={}'.format(args.ml_opts_dict))
            logging.debug('snap test score={:.8f} t_fit={:.2f}s t_predict={:.2f}s t_load={:.2f}s'.format(snap_score, snap_t_fit, snap_t_predict, snap_load_time))
            assert score == snap_score
    except Exception:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback,
                                  limit=2, file=sys.stdout)

