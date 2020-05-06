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
import logging

from smlp import DefaultPreprocFns

class MLPipelineParams(object):
    def __init__(self, preproc_fn=None, **kwargs):
        self.dataset_type = kwargs.get('dataset_type', 'csv')
        self.dataset_path = kwargs.get('dataset_path', 'nil')
        self.dataset_test_path = kwargs.get('dataset_test_path', 'nil')
        self.dataset_binary_type = kwargs.get('dataset_binary_type', 'np').lower()
        self.chunk_size = int(kwargs.get('chunk_size', 10))
        self.ml_lib = kwargs.get('ml_lib', 'snap').lower()
        self.ml_op = kwargs.get('ml_op', 'train').lower()
        self.ml_obj = kwargs.get('ml_obj', 'mse').lower()
        self.ml_opts_dict = kwargs.get('ml_opts_dict', {})
        self.ml_model_lst = kwargs.get('ml_model_lst', [])
        self.preprocessing = kwargs.get('preprocessing', 'nil').lower()
        self.label_col_idx = kwargs.get('label_col_idx', 0)
        self.num_epochs = int(kwargs.get('num_epochs', 1))
        self.num_prefetched_chunks = int(kwargs.get('num_prefetched_chunks', 2))
        self.rand_state = kwargs.get('rand_state', 42)
        self.debug = kwargs.get('debug', False)
        self.override_chunk = kwargs.get('override_chunk', False)
        self.preproc_fn = preproc_fn if preproc_fn != None else DefaultPreprocFns.default_np_preproc_fn
        self.__sanitize()

    def __sanitize(self):
        # convert values to int when possible
        for k in self.ml_opts_dict:
            try:
                if self.ml_opts_dict[k] == 'True':
                    self.ml_opts_dict[k] = True
                elif self.ml_opts_dict[k] == 'False':
                    self.ml_opts_dict[k] = False
                else:
                    self.ml_opts_dict[k] = int(self.ml_opts_dict[k])
            except:
                pass
        logging.debug(self.ml_opts_dict)
        # make sure objective is consistent
        for k in self.ml_opts_dict:
            if k == 'objective':
                if self.ml_opts_dict[k] != self.ml_obj:
                    logging.error('objective parameter mismatch ml_obj={} ml_opts_dict["objective"]={}'.format(self.ml_obj, self.ml_opts_dict[k]))
                assert self.ml_opts_dict[k] == self.ml_obj

    def pretty_print_str(self):
        ret=''
        for attr in dir(self):
            ret += '{} = {}'.format(attr, getattr(self, attr))
        return ret
