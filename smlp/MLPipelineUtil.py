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

def AddCommonParserArgs(parser):
    parser.add_argument('--dataset_type', default = 'csv', help='Dataset type')
    parser.add_argument('--dataset_path', dest='dataset_path', default = 'test/data/train.csv', help='Path to dataset')
    parser.add_argument('--dataset_test_path', dest='dataset_test_path', default = 'test/data/test.csv', help='Path to test dataset')
    parser.add_argument('--dataset_binary_type', default = 'np', help='Output type after loading and parsing')
    parser.add_argument('--label_col_idx', dest='label_col_idx', default = 0, help='Dataset label column idx')
    parser.add_argument('--chunk_size', dest='chunk_size', default=10, help='Chunk size per pipeline stage')
    parser.add_argument('--ml_lib', dest='ml_lib', default='', help='ML lib to use')
    parser.add_argument('--ml_model', dest='ml_model', default='boosting', help='ML model to use')
    parser.add_argument('--ml_model_options', dest='ml_opts_dict', default="objective=mse,num_round=1,min_max_depth=4,max_max_depth=6,n_threads=40,random_state=42", help='List of options for ML model', type=lambda s: dict(x.split('=') for x in s.split(','))) # lambda s: [x for x in s.split(',')])
    # TODO support for loading binary models
    parser.add_argument('--ml_models', dest='ml_model_lst', default=[], help="model in binary format")
    parser.add_argument('--ml_op', dest='ml_op', default='train', help='ML operation')
    parser.add_argument('--ml_obj', dest='ml_obj', default='mse', help='ML objective function')
    parser.add_argument('--random_state', dest='rand_state', default=42, help='Random state seed')
    parser.add_argument('--num_epochs', dest='num_epochs', default=1, help='Number of training epochs')
    parser.add_argument('--num_prefetched_chunks', dest='num_prefetched_chunks', default=4, help='Number of chunks to prefetch in background')
    parser.add_argument('--debug', dest='debug', action='store_true', help='print debug information')
    parser.add_argument('--override_chunk_sanity_check', dest='override_chunk', action='store_true', help='')
    # TODO: add pre-processing parameters
