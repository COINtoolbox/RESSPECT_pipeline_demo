# Copyright 2023 resspect software
# Author: Emille E. O. Ishida
#
# created on 17 January 2023
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


from resspect.time_domain_plasticc import PLAsTiCCPhotometry
from resspect.lightcurves_utils import PLASTICC_TARGET_TYPES
import time

field = 'DDF'                           # DDF or WFD
output_dir = '/media/emille/git/COIN/COINtoolbox/RESSPECT_pipeline_paper/code/' #'/media/RESSPECT/data/PLAsTiCC/for_pipeline/' + field + '/features/pool/'
raw_data_dir = '/media/RESSPECT/data/PLAsTiCC/for_pipeline/' + field + '/initial_samples/'     # path to PLAsTiCC zenodo files
photo_file = field + '_pool_lightcurves_01.csv'

create_daily_files =  True             # create 1 file for each day of survey (do this only once!)

get_cost = True                         # calculate cost of each observation
ask_user = False                        # only use False if you are not in interactive mode
queryable_criteria = 2                  # if 2, estimate brightness at time of query
sample = 'test'                         # original plasticc sample
vol = 1                                 # index of plasticc zenodo file for test sample
spec_SNR = 10                           # minimum SNR required for spec follow-up
tel_names = ['4m', '8m']                # name of telescopes considered for spec follow-up
tel_sizes = [4, 8]                      # size of primay mirrors, in m
number_of_processors = 30               # Number of cpu processes to use.

# start PLAsTiCCPhotometry object
feature_class = PLAsTiCCPhotometry()
feature_class.build(config='RESSPECT', sample='test', photo_file=photo_file)

feature_class.read_metadata(
                     path_to_data_dir=raw_data_dir,
                     classes=PLASTICC_TARGET_TYPES.keys(),
                     field=field,
                     meta_data_file_name= field + '_pool_metadata.csv')

# Extract features for specific days
time_window = [460,461]


# Fit first N available snid light curves
snids = feature_class.metadata['object_id'].values

start_time = time.time()

feature_class.fit_all_snids_lc(
           raw_data_dir=raw_data_dir, snids=snids,
           output_dir=output_dir,
           vol=vol, queryable_criteria=queryable_criteria,
           get_cost=get_cost,
           tel_sizes=tel_sizes,
           tel_names=tel_names,
           spec_SNR=spec_SNR,
           time_window=time_window, sample=sample,
           create_daily_files=create_daily_files,
           number_of_processors=number_of_processors,
           ask_user=ask_user)

print("--- %s hours ---" % ((time.time() - start_time)/3600))
