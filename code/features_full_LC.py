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


from resspect import fit_plasticc_bazin

orig_sample = 'train'


output_sample = 'train'
header_file = '/media/RESSPECT/data/PLAsTiCC/for_pipeline/DDF/initial_samples/DDF_' + output_sample + '_metadata.csv'
photo_file = '/media/RESSPECT/data/PLAsTiCC/for_pipeline/DDF/initial_samples/DDF_' + output_sample + '_lightcurves_01.csv'

output_file = '/media/RESSPECT/data/PLAsTiCC/for_pipeline/DDF/features/PLAsTiCC_Bazin_' + output_sample + '.csv'

ncores = 35

fit_plasticc_bazin(photo_file, header_file, output_file, sample=orig_sample,
                   number_of_processors=ncores)