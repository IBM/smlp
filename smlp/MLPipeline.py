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
               Adrian Schüpbach,
               Thomas Parnell

End Copyright
'''

import logging
import mlio
import sys
import time
import numpy as np
from mlio.integ.numpy import as_numpy
import asyncio
import atexit
from pathlib import Path
import traceback

class MLPipeline(object):
    def __init__(self, params):
        self.params = params
        self.aio_loop = asyncio.get_event_loop()
        atexit.register(self.__cleanup)

        log_level = logging.DEBUG if self.params.debug else logging.INFO
        # basicConfing doesn't work due to import ordering and other module(s) calling this first
        # logging.basicConfig(format='%(asctime)s %(message)s', level=log_level,
        # datefmt='%Y-%m-%d %H:%M:%S')

        # alter root logger directly
        l = logging.getLogger()
        l.setLevel(log_level)
        h=l.handlers[0]
        h.setFormatter(logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
        logging.debug('params={}'.format(self.params.pretty_print_str()))

    def __cleanup(self):
        self.aio_loop.close()

    def run_pipeline(self):
        # TODO:
        # check and proceed based on parameters, e.g., dataset_type, etc.
        # use proper parameters for the model training and prediction
        valid = self.__param_check()
        if not valid:
            return None

        if not self.params.override_chunk:
            valid = self.__sanitize_model_size()
            if not valid:
                return None
        try:
            if self.params.ml_op == 'train':
                return self.aio_loop.run_until_complete(self.__async_train())
            elif self.params.ml_op == 'predict':
                return self.aio_loop.run_until_complete(self.__async_predict())
            else:
                logging.error("invalid")
        except Exception:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_traceback,
                                      limit=2, file=sys.stdout)
            sys.exit(1)
        return None

    def __param_check(self):
        if self.params.dataset_type != 'csv':
            logging.error('not supported dataset file type')
            return False
        if self.params.dataset_binary_type != 'np':
            logging.error('not supported dataset binary represenation type')
            return False
        if self.params.ml_op != 'train' and self.params.ml_op != 'predict':
            logging.error('unsupported ML operation {}'.format(self.params.ml_op))
            return False
        # TODO all checks
        return True

    def __sanitize_model_size(self):
        # choose the biggest chunk size that fits in memory -> minimize number of models trained
        MAX_MODEL_NR=1000

        # get file size in bytes
        fpath = self.params.dataset_path if self.params.ml_op == 'train' else self.params.dataset_test_path
        fsize_b = Path(fpath).stat().st_size

        old_chunk_size = self.params.chunk_size
        tot_line_size_b=1
        max_lines = 500
        with open(fpath) as f:
            for i, l in enumerate(f):
                line_size_b=max(1, len(l))
                tot_line_size_b+=line_size_b
                if i >= max_lines:
                    break;
        line_nr=min(max_lines, i)
        logging.debug('l_nr={} tot_l_s={} f_s={}'.format(line_nr, tot_line_size_b, fsize_b))
        avg_line_size_b = tot_line_size_b / line_nr
        if self.params.ml_op == 'predict':
            import psutil
            model_size_b = sys.getsizeof(self.params.ml_model_lst)
            available_mem_b = psutil.virtual_memory().available
            # 1 GiB + (2 x model_size) + (fsize with 5% inefficiency)
            tot_mem_req = (pow(2,30) + model_size_b * 2.0 + 1.05 * fsize_b)
            if tot_mem_req < available_mem_b:
                self.params.chunk_size = int(1.05 * fsize_b / avg_line_size_b)  # whole file
            else:
                if available_mem_b < pow(2,30) + model_size_b * 2.0:
                    logging.error('not enough memory {} to hold the models {}'.format(available_mem_b, pow(2,30) + model_size_b * 2.0))
                    return False
                self.params.chunk_size = int((1.05 * fsize_b) / (available_mem_b - pow(2,30) - model_size_b * 2.0))
                self.params.chunk_size = int ((self.params.chunk_size + avg_line_size_b - 1) / avg_line_size_b)

        elif self.params.ml_op == 'train':
            if fsize_b < avg_line_size_b * self.params.chunk_size:
                # chunk size > than file size
                self.params.chunk_size = int(1.05 * fsize_b / avg_line_size_b) + 1  # whole file
            tot_chunk_nr_est = int(fsize_b / (avg_line_size_b * self.params.chunk_size))
            num_rounds = self.params.num_epochs if self.params.ml_lib != 'snap' else self.params.ml_opts_dict['num_round']
            logging.debug('avg_l_s={} tot_chunk_nr_est={}'.format(avg_line_size_b, tot_chunk_nr_est))

            # try to sanitize
            while tot_chunk_nr_est * num_rounds > MAX_MODEL_NR and self.params.chunk_size * avg_line_size_b < fsize_b:
                self.params.chunk_size += 1
                tot_chunk_nr_est = int(fsize_b / (avg_line_size_b * self.params.chunk_size))

            if tot_chunk_nr_est * num_rounds > MAX_MODEL_NR:
                logging.error('Estimate of total model size {} exceeds max {}.'.format(tot_chunk_nr_est *num_rounds, MAX_MODEL_NR))
                return False
        else:
            assert False
            return False

        if old_chunk_size != self.params.chunk_size:
            logging.debug('changed chunk size from {} to {}.'.format(old_chunk_size, self.params.chunk_size))

        return True

    async def __train_old(self):
        chunk_size = self.params.chunk_size # getattr(self.params, "chunk_size")
        dataset = mlio.list_files(getattr(self.params, "dataset_path"), pattern='*.csv')
        logging.debug('mlio dataset={}'.format(dataset))
        preproc_fn=self.params.preproc_fn
        reader_params = mlio.DataReaderParams(dataset=dataset, batch_size=chunk_size, num_prefetched_batches=self.params.num_prefetched_chunks)
        reader = mlio.CsvReader(reader_params)
        logging.debug('mlio reader={}'.format(reader))
        num_epochs = self.params.num_epochs  # Number of times to read the full dataset.
        # use eta parameteres
        eta = 0.01
        if self.params.ml_lib == 'snap':
            eta = 0.1
            from pai4sk import BoostingMachine as Booster
        else:
            from sklearn.tree import DecisionTreeRegressor

        logging.debug('starting training')
        models=[]
        # RecordIOProtobufReader is simply an iterator over mini-batches of data.
        for chunk_idx,chunk in enumerate(reader):
            rand_state = self.params.rand_state
            # Alternatively, transform the mini-batch into a NumPy array.
            chunk_train_Xy = np.column_stack([as_numpy(feature) for feature in chunk])
            chunk_train_X, chunk_train_y = preproc_fn(chunk_train_Xy, self.params.label_col_idx)
            #print(chunk_train_X)
            if self.params.ml_lib == 'snap':
                bl = Booster(**self.params.ml_opts_dict)
                bl.fit(chunk_train_X, chunk_train_y)
                models.append(bl)
            else:
                z_train = np.zeros(chunk_train_X.shape[0])
                for epoch in range(num_epochs):
                    #logging.debug('chunk idx={} chunk={}'.format(chunk_idx, chunk))

                    target = chunk_train_y - z_train
                    bl = DecisionTreeRegressor(max_depth=3, max_features='sqrt', random_state=rand_state)
                    bl.fit(chunk_train_X, target)
                    u_train = bl.predict(chunk_train_X)
                    z_train = z_train + eta*u_train
                    models.append(bl)
        return models

    async def __preprocess_chunk(self, chunk):
        t0 = time.time()
        preproc_fn = self.params.preproc_fn
        Xy = np.column_stack([as_numpy(feature) for feature in chunk])
        X, y = preproc_fn(Xy, self.params.label_col_idx)
        logging.debug('t_preproc_chunk={:.2f}'.format(time.time() - t0))
        return X,y

    async def __train_chunk(self, X, y):
        if self.params.ml_lib == 'snap':
            from pai4sk import BoostingMachine as Booster
        else:
            from sklearn.tree import DecisionTreeRegressor
        preproc_fn=self.params.preproc_fn
        num_epochs = self.params.num_epochs  # Number of times to read the full dataset.
        # use eta parameteres
        eta = 0.01
        if self.params.ml_lib == 'snap':
            eta = 0.1
        else:
            rand_state = self.params.rand_state

        logging.debug('starting training')
        models=[]

        t0 = time.time()
        if self.params.ml_lib == 'snap':
            bl = Booster(**self.params.ml_opts_dict)
            bl.fit(X, y)
            models.append(bl)
        else:
            z_train = np.zeros(X.shape[0])
            for epoch in range(num_epochs):
                #logging.debug('chunk idx={} chunk={}'.format(chunk_idx, chunk))

                target = y - z_train
                bl = DecisionTreeRegressor(max_depth=3, max_features='sqrt', random_state=rand_state)
                bl.fit(X, target)
                u_train = bl.predict(X)
                z_train = z_train + eta*u_train
                models.append(bl)
        logging.debug('train chunk fit={:.2f}'.format(time.time() - t0))

        return models

    async def __train(self):
        chunk_size = self.params.chunk_size # getattr(self.params, "chunk_size")
        dataset = mlio.list_files(getattr(self.params, "dataset_path"), pattern='*.csv')
        logging.debug('mlio dataset={}'.format(dataset))
        reader_params = mlio.DataReaderParams(dataset=dataset, batch_size=chunk_size, num_prefetched_batches=self.params.num_prefetched_chunks)
        reader = mlio.CsvReader(reader_params)
        logging.debug('mlio reader={}'.format(reader))
        num_epochs = self.params.num_epochs  # Number of times to read the full dataset.
        # use eta parameteres
        eta = 0.01
        if self.params.ml_lib == 'snap':
            eta = 0.1
            from pai4sk import BoostingMachine as Booster
        else:
            from sklearn.tree import DecisionTreeRegressor

        logging.debug('starting training')
        models=[]
        # preample
        chunkim1 = reader.read_example()
        if chunkim1 != None:
            X_im1, y_im1 = await self.__preprocess_chunk(chunkim1)
        chunki = reader.read_example()
        i=1
        logging.debug('chunk{}={}'.format(0, chunkim1))
        logging.debug('chunk{}={}'.format(i, chunki))
        while chunki != None:
            logging.debug('chunk{}={}'.format(i, chunki))
            task_preprocess = asyncio.create_task(self.__preprocess_chunk(chunki))
            task_train = asyncio.create_task(self.__train_chunk(X_im1, y_im1))
            X_i, y_i = await task_preprocess
            models.extend(await task_train)
            X_im1 = X_i
            y_im1 = y_i
            chunkim1 = chunki
            chunki = reader.read_example()
            i += 1
        # postample
        if chunkim1 != None:
            logging.debug('y{}m1={}'.format(i, y_im1))
            models.extend(await self.__train_chunk(X_im1, y_im1))
        return models

    async def __predict_chunk(self, X, y):
        num_epochs = self.params.num_epochs  # Number of times to read the full dataset.
        # use eta parameteres
        eta = 0.01
        if self.params.ml_lib == 'snap':
            eta = 0.1
        ml_obj = self.params.ml_obj
        models = self.params.ml_model_lst
        if self.params.ml_lib == 'snap':
            num_train_chunks = len(models)
        else:
            num_train_chunks = int(len(models) / num_epochs)
        z_test_mean = np.zeros(X.shape[0])
        for tr_chunk in range(num_train_chunks):
            if self.params.ml_lib == 'snap':
                bl = models[tr_chunk]
                # predictions from model trained on tr_chunk
                yhat_test = bl.predict(X)
                z_test_mean += yhat_test
            else:
                # predictions from model trained on tr_chunk
                for epoch in range(num_epochs):
                    bl = models[epoch * num_train_chunks + epoch]
                    z_test_mean += eta * bl.predict(X)
        # average predictions across all models
        z_test_mean /= num_train_chunks
        logging.debug('z_test_mean shape={}'.format(z_test_mean.shape))
        if ml_obj == 'mse':
            score = np.sum(np.power(y - z_test_mean, 2))
        elif ml_obj == 'logloss':
            score = np.sum((y > 0) == (z_test_mean > 0))
        score_norm = y.shape[0]
        return score, score_norm

    async def __predict(self):
        if self.params.ml_lib == 'snap':
            from pai4sk import BoostingMachine as Booster
        else:
            from sklearn.tree import DecisionTreeRegressor
        chunk_size = self.params.chunk_size # getattr(self.params, "chunk_size")
        dataset = mlio.list_files(getattr(self.params, "dataset_test_path"), pattern='*.csv')
        logging.debug('mlio dataset={}'.format(dataset))
        reader_params = mlio.DataReaderParams(dataset=dataset, batch_size=chunk_size, num_prefetched_batches=self.params.num_prefetched_chunks)
        reader = mlio.CsvReader(reader_params)
        logging.debug('mlio reader={}'.format(reader))


        logging.debug('starting inference')
        score_norm = 0.0
        score = 0.0
        # preample
        chunkim1 = reader.read_example()
        if chunkim1 != None:
            X_im1, y_im1 = await self.__preprocess_chunk(chunkim1)
        chunki = reader.read_example()
        i=1
        logging.debug('chunk{}={}'.format(0, chunkim1))
        logging.debug('chunk{}={}'.format(i, chunki))
        while chunki != None:
            logging.debug('chunk{}={}'.format(i, chunki))
            task_predict = asyncio.create_task(self.__predict_chunk(X_im1, y_im1))
            task_preprocess = asyncio.create_task(self.__preprocess_chunk(chunki))
            X_i, y_i = await task_preprocess
            s,n = await task_predict
            score += s
            score_norm += n
            X_im1 = X_i
            y_im1 = y_i
            chunkim1 = chunki
            chunki = reader.read_example()
            i += 1
        # postample
        if chunkim1 != None:
            logging.debug('y{}m1={}'.format(i, y_im1))
            s,n = await self.__predict_chunk(X_im1, y_im1)
            score += s
            score_norm += n
        score /= score_norm
        return score

    async def __async_predict(self):
        return await self.__predict()

    async def __async_train(self):
        return await self.__train()
