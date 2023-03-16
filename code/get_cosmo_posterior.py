# Copyright 2023 resspect software
# Author: Emille E. O. Ishida
# 
# slightly modified from Malz et al., 2023
# https://github.com/COINtoolbox/RESSPECT_metric/blob/main/code/get_cosmo_posterior.py
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

import pandas as pd
import numpy as np
import stan as pystan
import os
import pickle
import arviz
import matplotlib.pyplot as plt
from cmdstanpy import CmdStanModel

from shutil import move


def create_directories(dir_output: str):
    """Create directory structure. 
    
    Parameters
    ----------
    dir_output: str
        Full path to main output directory. 
    """
    
    dirname_list = [dir_output ,
                dir_output + 'fitres/',
                dir_output + 'M0DIF/',
                dir_output + 'posteriors/',
                dir_output + 'posteriors/trace/',
                dir_output + 'posteriors/pkl/',
                dir_output + 'posteriors/csv/',
                dir_output + 'COV/',
                dir_output + 'stan_input/',
                dir_output + 'stan_summary/',
                dir_output + 'SALT2mu_input/']


    for fname in dirname_list:
        if not os.path.isdir(fname):
            os.makedirs(fname)


def read_fitres(fname_zenodo_meta: str,
                fname_sample: str,
                fname_fitres_lowz: str,
                fname_output: str,
                dir_output: str,
                sample: str, 
                lowz = True,
                to_file=False):
    """
    Read fitres file for a given sample.
    
    Parameters
    ----------
    fname_zenodo_meta: str
        Path to zenodo metadata file.
    fname_sample: str
        Path to sample file. 
    fname_fitres_lowz: str
        Path to lowz fitres file.
    fname_output: str
        Path to output file containing concatenated fitres.
    dir_output: str
        Output folder.
    sample: str
        Sample identifier. 
    lowz: bool (optional)
        If True add lowz sample. Default is True.
    to_file: bool (optional)
        If True, save concatenated fitres to file.
        Default is False.
        
    Returns
    -------
    pd.DataFrame
        Complete fitres data frame.
    """
    
    # read plasticc test metadata
    test_metadata = pd.read_csv(fname_test_zenodo_meta)
    print(test_metadata.shape)

    # read sample 
    fitres_main = pd.read_csv(fname_sample, index_col=False)

    # read lowz sample
    if lowz:
        fitres_lowz = pd.read_csv(fname_fitres_lowz, index_col=False, comment="#", 
                                  skip_blank_lines=True, delim_whitespace=True)
        fitres_lowz['zHD'] = fitres_lowz['SIM_ZCMB']
        fitres_lowz.fillna(value=-99, inplace=True)
        
        flag1 = np.array([item in fitres_main.keys() for item in fitres_lowz.keys()])
        flag2 = np.array([item in fitres_lowz.keys() for item in fitres_main.keys()])
        fitres = pd.concat([fitres_lowz[list(fitres_lowz.keys()[flag1])], 
                            fitres_main[list(fitres_main.keys()[flag2])]], 
                            ignore_index=True)
        
    else:
        fitres = fitres_main

    if to_file:
        fitres.to_csv(dir_output + 'fitres/master_fitres_' + sample + '_lowz_withbias.fitres', 
                      sep=" ", index=False)
        

def fit_salt2mu(biascorr_dir: str, sample: str, root_dir: str, 
                fname_input_salt2mu: str, fname_output_salt2mu:str, 
                nbins=30, field='DDF', biascorr=False):
    """
    Parameters
    ----------
    biascorr_dir: str
        Path to bias correction directory.
    sample: str
        Sample identification.
    root_dir: str
        Path to directory storing SALT2 fitres subfolder.
    fname_input_salt2mu: str
        Path to example input SALT2mu file.
    fname_output_salt2mu: str
        Path to output SALT2mu input file.
    nbins: int (optional)
        Number of bins for SALT2mu fit. Default is 30.
    field: str (optional)
        Observation strategy. DDF or WFD. Default is DDF.
    biascorr: bool (optional)
        If true add bias correction. Default is False.    
    """
    
    # change parameters for SALT2mu
    op = open(fname_input_salt2mu, 'r')
    lin = op.readlines()
    op.close()

    for i in range(len(lin)):
        if lin[i][:4] == 'nlogzbin':
            lin[i] = 'nlogzbin=' + str(nbins) + '\n'
        elif lin[i][:6] == 'prefix':
            lin[i] = 'prefix=test_salt2mu_lowz_withbias_' + sample + '\n'
        elif lin[i][:4] == 'file':
            lin[i] = 'file=' + root_dir + 'fitres/master_fitres_' + sample + \
                     '_lowz_withbias.fitres' + '\n'
        elif lin[i][:7] == 'simfile' and biascorr:
            lin[i] = 'simfile_biascor=' + biascorr_dir + '/LSST_' + field + \
           '_BIASCOR.FITRES.gz,' + biascorr_dir + '/FOUNDATION_BIASCOR.FITRES.gz\n'
        elif lin[i][:7] == 'simfile' and not biascorr:
            lin[i] = '\n'
        elif lin[i][:3] == 'opt' and not biascorr:
            lin[i] = '\n'

    op2 = open(fname_output_salt2mu, 'w')
    for line in lin:
        op2.write(line)
    op2.close()

    # get distances from SALT2MU
    os.system('SALT2mu.exe ' + fname_output_salt2mu)
  
    # put files in correct directory
    move('test_salt2mu_lowz_withbias_' + sample + '.COV', 
         root_dir + 'COV/test_salt2mu_lowz_withbias_' + sample + '.COV')
    move('test_salt2mu_lowz_withbias_' + sample + '.fitres', 
         root_dir + 'fitres/test_salt2mu_lowz_withbias_' + sample + '.fitres')
    move('test_salt2mu_lowz_withbias_' + sample + '.M0DIF', 
         root_dir + 'M0DIF/test_salt2mu_lowz_withbias_' + sample + '.M0DIF')
    move(fname_output_salt2mu, root_dir + 'SALT2mu_input/SALT2mu_lowz_withbias' + sample + '.input')
    
    
def remove_duplicated_z(fitres_final: pd.DataFrame):
    """
    Add small offset to avoid duplicated redshift values.
    
    Parameters
    ----------
    fitres_final: pd.DataFrame
        Data frame containing output from SALT2 fit.
        
    Returns
    -------
    pd.DataFrame
        Fitres data frame with updated redshifts.
    """
    
    z_all = []
    for j in range(fitres_final.shape[0]):
        z = fitres_final.iloc[j]['SIM_ZCMB']
        z_new = z
        if z in z_all:
            while z_new in z_all:
                z_new = z + np.random.normal(loc=0, scale=0.001)
            
        fitres_final.at[j, 'SIM_ZCMB'] = z_new
        z_all.append(z_new)
        
    return fitres_final


def fit_stan(fname_fitres_comb: str, dir_output: str, sample: str,
             dir_input_cosmo: str, screen=False, lowz=True, 
             bias=True, plot=False, om_pri=[0.3, 0.01],
             w_pri=[-11, 9], warmup=10000, n_iter=12000, n_chains=1):
    """
    Fit Stan model for w.
    
    Parameters
    ----------
    fname_fitres_comb: str
        Complete path to fitres (with lowz, if requested).
    dir_output: str
        Complete path to output root folder.
    sample: str
        Sample to be fitted.
    screen: bool (optional)
        If True, print Stan results to screen. Default is False.     
    lowz: bool (optional)
        If True, add low-z sample. Default is True.
    bias: bool (optional)
        If True, add bias correction. Default is True. 
    plot: bool (optional)
        If True, generate chains plot. Default is False.
    om_pri: list (optional)
        Mean and std of om Gaussian prior. Default is [0.3, 0.01].
    w_pri: list (optional)
        Minimum and maximum of flat prior on w. Default is [-11, 9].
    warmup: int (optional)
        Number of interations in warmup from pystan. Default is 10000.
    n_iter: int (optional)
        Total number of interations from pystan. Default is 12000.
    n_chains: int (optional)
        Number of chains. Default is 1.
    """

    # read data for Bayesian model
    fitres_final = pd.read_csv(fname_fitres_comb, index_col=False, comment="#", 
                          skip_blank_lines=True, delim_whitespace=True)

    # set initial conditions
    z0 = 0
    E0 = 0
    c = 3e5
    H0 = 70

    # add small offset to duplicate redshifts
    fitres_final = remove_duplicated_z(fitres_final)

    # order data according to redshift 
    indx = np.argsort(fitres_final['SIM_ZCMB'].values)

    # create input for stan model
    stan_input = {}
    stan_input['nobs'] = fitres_final.shape[0]
    stan_input['z'] = fitres_final['SIM_ZCMB'].values[indx]
    stan_input['mu'] = fitres_final['MU'].values[indx]
    stan_input['muerr'] = fitres_final['MUERR'].values[indx]
    stan_input['z0'] = z0
    stan_input['H0'] = H0
    stan_input['c'] = c
    stan_input['E0'] = np.array([E0])
    stan_input['ompri'] = om_pri[0]
    stan_input['dompri'] = om_pri[1]
    stan_input['wmin'] = w_pri[0]
    stan_input['wmax'] = w_pri[1]

    # save only stan input to file
    fname_stan_input = dir_output + 'stan_input/stan_input_salt2mu_lowz_withbias_' + sample + '.csv'

    stan_input2 = {}
    stan_input2['z'] = stan_input['z']
    stan_input2['mu'] = stan_input['mu']
    stan_input2['muerr'] = stan_input['muerr']
    stan_input2['SIM_TYPE_INDEX'] = fitres_final['SIM_TYPE_INDEX'].values[indx]
    stan_input_tofile = pd.DataFrame(stan_input2)
    stan_input_tofile.to_csv(fname_stan_input, index=False)

    # fit Bayesian model           
    model = CmdStanModel(stan_file=dir_input_cosmo + 'cosmo.stan', 
                         cpp_options={'STAN_THREADS':'true'})
    fit = model.sample(data=stan_input, iter_sampling=n_iter, chains=n_chains,
                      iter_warmup=warmup)
    
    # get posterior
    chains = pd.DataFrame()
    for par in ['om', 'w', 'M']:
        chains[par] = fit.stan_variables()[par]
        
    chains_final = chains.sample(n=n_chains * (n_iter - warmup), replace=False)

    if lowz and bias:
        chains_fname = dir_output + 'posteriors/pkl/chains_' + sample + '_lowz_withbias.pkl'
        trace_fname = dir_output + 'posteriors/trace/trace_plot_' + sample + '_lowz_withbias.png'
    elif lowz and not bias:
        chains_fname = dir_output + 'posteriors/pkl/chains_' + sample + '_lowz_nobias.pkl'
        trace_fname = dir_output + 'posteriors/trace/trace_plot_' + sample + '_lowz_nobias.png'
    else:
        chains_fname = dir_output + 'posteriors/pkl/chains_' + sample + '.pkl'
        trace_fname = dir_output + 'posteriors/trace/trace_plot_' + sample + '.png'

    if plot:
        chains_dict = dict(chains_final)
        arviz.plot_trace(chains_dict, ['om', 'w'])
        plt.savefig(trace_fname)
    
    if lowz and bias:
        chains_final.to_csv(dir_output + 'posteriors/csv/' + \
                            'chains_'  + sample + '_lowz_withbias.csv', index=False)
  
     
##################    user choices     ##########################################
fname_sample = 'RandomSampling_20_batchNone_1year'         # choose sample
nbins = 30                                    # number of bins for SALT2mu
om_pri = [0.3, 0.01]                          # gaussian prior on om => [mean, std]
w_pri = [-11, 9]                              # flat prior on w
lowz = True                                   # choose to add lowz sample
field = 'DDF'                                 # choose field
biascorr = True
screen = True
n_iter = 6000
warmup = 5000
n_chains = 5

save_full_fitres = True
plot_chains = True

# path to output directories
output_root = '/media/RESSPECT/data/PLAsTiCC/for_pipeline/' + field + '/cosmo_results/'

# path to auxiliar input files from Malz et al., 2023
dir_input = '/media/emille/git/COIN/COINtoolbox/RESSPECT_metric/utils/'
dir_input_cosmo = '/media/emille/git/COIN/RESSPECT_repo/RESSPECT/auxiliary_files/'
fname_input_salt2mu = dir_input + 'template_SALT2mu.input'
fname_fitres_lowz = dir_input + 'lowz_only_fittres.fitres'

# path to zenodo input files
fname_test_zenodo_meta = '/media/RESSPECT/data/PLAsTiCC/PLAsTiCC_zenodo/plasticc_test_metadata.csv'

# folder with resources for bias correction calculations
biascorr_dir = '/media/RESSPECT/data/PLAsTiCC/biascorsims/'

samples_dir = '/media/RESSPECT/data/PLAsTiCC/for_pipeline/' + field + '/learn_loop_results/cosmo_samples/' 
fname_fitres_comb = output_root + 'fitres/test_salt2mu_lowz_withbias_' + fname_sample + '.fitres'


###################################################################################

        
##############################
### Create directories #######
##############################

create_directories(dir_output=output_root)
        
#######################
### Generate fitres ###
#######################
fname_output_salt2mu = dir_input + '/SALT2mu_' + fname_sample  + '.input'

read_fitres(fname_zenodo_meta=fname_test_zenodo_meta,
            fname_sample=samples_dir + fname_sample + '.csv',
            fname_fitres_lowz=fname_fitres_lowz,
            fname_output=fname_output_salt2mu,
            dir_output=output_root,
            sample=fname_sample, 
            lowz = lowz,
            to_file=save_full_fitres)



#######################
####### SALT2mu #######
#######################
fit_salt2mu(biascorr_dir=biascorr_dir, 
            sample=fname_sample, root_dir=output_root, 
            fname_input_salt2mu=fname_input_salt2mu, fname_output_salt2mu=fname_output_salt2mu, 
            nbins=nbins, field=field, biascorr=biascorr)

#######################
##### Stan model ######
#######################
        
fit = fit_stan(fname_fitres_comb=fname_fitres_comb, dir_output=output_root, 
               dir_input_cosmo=dir_input_cosmo, sample=fname_sample + '.csv',
               screen=screen, lowz=lowz, bias=biascorr, 
               plot=plot_chains, om_pri=om_pri, w_pri=w_pri,
               n_iter=n_iter, warmup=warmup, n_chains=n_chains)