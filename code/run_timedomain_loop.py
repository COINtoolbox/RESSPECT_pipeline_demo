# Copyright 2023 resspect software
# Author: Emille E. O. Ishida
#
# created on 28 February 2023
#
# Licensed MIT License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/license/mit/
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from resspect import time_domain_loop
    
days = [42, 407]
training = 20
strategy = 'RandomSampling'
n_estimators = 1000
batch = None

sep_files = True
save_full_query= False
    
output_diag_file = '/media/RESSPECT/data/PLAsTiCC/for_pipeline/DDF/learn_loop_results/metrics/' + \
                    'metrics_' + strategy + '_' + str(training) + '_batch' + str(batch) + '.csv'

output_query_file = '/media/RESSPECT/data/PLAsTiCC/for_pipeline/DDF/learn_loop_results/queried/' + \
                    strategy + '/queried_' + strategy + '_' + str(training) + '_batch' + str(batch) + '.csv'

path_to_features_dir = '/media/RESSPECT/data/PLAsTiCC/for_pipeline/DDF/features/pool/'
  

budgets = (2. * 3600, 1. * 3600)
classifier = 'RandomForest'
clf_bootstrap = False #True
feature_method = 'Bazin'
screen = False
fname_pattern = ['day_', '.csv']
canonical = False
queryable= True
    
path_to_ini_files = {}
path_to_ini_files['train'] = '/media/RESSPECT/data/PLAsTiCC/for_pipeline/DDF/features/PLAsTiCC_Bazin_train.csv'
path_to_ini_files['test'] = '/media/RESSPECT/data/PLAsTiCC/for_pipeline/DDF/features/PLAsTiCC_Bazin_test.csv'
path_to_ini_files['validation'] = '/media/RESSPECT/data/PLAsTiCC/for_pipeline/DDF/features/PLAsTiCC_Bazin_validation.csv'

survey='LSST'
    
# run time domain loop
time_domain_loop(days=days, output_metrics_file=output_diag_file,
                 output_queried_file=output_query_file,
                 path_to_features_dir=path_to_features_dir,
                 budgets=budgets, clf_bootstrap=clf_bootstrap,
                 strategy=strategy, fname_pattern=fname_pattern, batch=batch, classifier=classifier,
                 canonical=canonical, sep_files=sep_files,
                 screen=screen, initial_training=training, path_to_ini_files=path_to_ini_files,
                 survey=survey, queryable=queryable, n_estimators=n_estimators, save_full_query=save_full_query)