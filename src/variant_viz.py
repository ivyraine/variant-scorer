import pandas as pd
import numpy as np
import os
from collections import namedtuple
import operator
import logging
import deepdish as dd
from matplotlib import pyplot as plt

from utils.argmanager import *
from utils.helpers import *

def main(args = None):
    if args is None:
        logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')
        args = fetch_viz_args()

    is_using_scoring_file_overrides = False
    filter_output_filenames = [ get_score_output_file_prefix(args.scoring_output_dir, args.sample_name, index) for index in range(len(args.models)) ]
    if args.score_filenames is not None:
        if len(args.score_filenames) != len(args.models):
            raise ValueError("Number of models and score filenames do not match, which they are requird to do if the --score-filenames flag is given. Exiting.")
        else:
            filter_output_filenames = [ os.path.join(args.scoring_output_dir, args.score_filenames[i]) for i in range(len(args.models)) ]
            is_using_scoring_file_overrides = True
    scoring_model_and_output_prefixes = list(zip(args.models, filter_output_filenames))

    is_using_shap_file_overrides = False
    shap_output_names = [ get_shap_output_file_prefix(args.shap_output_dir, args.sample_name, index) for index in range(len(args.models)) ]
    if args.score_filenames is not None:
        if len(args.score_filenames) != len(args.models):
            raise ValueError("Number of models and score filenames do not match, which they are requird to do if the --score-filenames flag is given. Exiting.")
        else:
            shap_output_names = [ os.path.join(args.shap_output_dir, args.score_filenames[i]) for i in range(len(args.models)) ]
            is_using_shap_file_overrides = True
    shap_model_and_output_prefixes = list(zip(args.models, shap_output_names))

    # Get means of allele1 and allele2 predictions
    allele1_preds = {}
    allele2_preds = {}
    total_models = len(scoring_model_and_output_prefixes)
    for model_index, (model_name, scoring_output_prefix) in enumerate(scoring_model_and_output_prefixes):
        predictions_hdf5_filepath = f"{scoring_output_prefix}.variant_predictions.h5"
        if args.predictions_override is not None:
            predictions_hdf5_filepath = args.predictions_override
        if not os.path.isfile(predictions_hdf5_filepath):
            raise ValueError(f"Error: The file {predictions_hdf5_filepath} doesn't exist. Exiting.")
        logging.debug(f"Reading predictions file {predictions_hdf5_filepath}")
        file = h5py.File(predictions_hdf5_filepath, 'r')
        logging.debug(f"Opened file {predictions_hdf5_filepath} successfully.")

        allele1_pred_counts = np.array(file['observed']['allele1_pred_counts'])
        allele1_pred_profile = np.array(file['observed']['allele1_pred_profiles'])
        allele1_preds[model_index] = allele1_pred_counts * allele1_pred_profile

        allele2_pred_counts = np.array(file['observed']['allele2_pred_counts'])
        allele2_pred_profile = np.array(file['observed']['allele2_pred_profiles'])
        allele2_preds[model_index] = allele2_pred_counts * allele2_pred_profile

    allele1_preds['mean'] = np.mean(np.array([allele1_preds[model_index] for model_index in range(total_models)]), axis=0)
    allele2_preds['mean'] = np.mean(np.array([allele2_preds[model_index] for model_index in range(total_models)]), axis=0)

    # Get means of shap predictions
    allele1_shap = {'counts': {}, 'profile': {}}
    allele2_shap = {'counts': {}, 'profile': {}}
    # TODO check if shap_type is in the hdf5
    shap_types = args.shap_type
    for shap_type in shap_types:
        allele1_shap[shap_type] = {}
        allele2_shap[shap_type] = {}
        for model_index, (model_name, shap_output_prefix) in enumerate(shap_model_and_output_prefixes):
            # TODO: generalize this to other shap_types
            shap_filepath = f"{shap_output_prefix}.variant_shap.counts.h5"
            if args.shap_override is not None:
                shap_filepath = f"{args.shap_override}.h5"
            if not os.path.isfile(shap_filepath):
                raise ValueError(f"Error: The file {shap_filepath} doesn't exist. Exiting.")
            logging.debug(f"Reading shap file {shap_filepath}")
            # file = h5py.File(shap_filepath, 'r')
            file = dd.io.load(shap_filepath)
            logging.debug(f"Opened file {shap_filepath} successfully.")
            alleles = np.array(file['alleles'])
            allele1_shap[shap_type][model_index] = np.array(file['projected_shap']['seq'])[alleles==0]
            allele2_shap[shap_type][model_index] = np.array(file['projected_shap']['seq'])[alleles==1]

    allele1_shap[shap_type]['mean'] = np.mean(np.array([allele1_shap[shap_type][model_index] for model_index in range(total_models)]), axis=0)
    allele2_shap[shap_type]['mean'] = np.mean(np.array([allele2_shap[shap_type][model_index] for model_index in range(total_models)]), axis=0)

    # def _plotter_profile(allele1_profiles, allele2_profiles, vlines, ax0, ref_label, alt_label):
    #     xmins, xmaxs = [], []
    #     for i, (x, allele1_profile) in enumerate(allele1_profiles):
    #         ax0.plot(x, allele1_profile, label=(f"ref ({ref_label})" if i==0 else ""), color='C0')
    #         xmins.append(min(x)); xmaxs.append(max(x))
    #     for i, (x, allele2_profile) in enumerate(allele2_profiles):
    #         ax0.plot(x, allele2_profile, label=(f"alt ({alt_label})" if i==0 else ""), color='C1')
    #         xmins.append(min(x)); xmaxs.append(max(x))
    #     for v in vlines:
    #         ax0.axvline(v, color='black', ls='--', linewidth=1)
    #     xmin, xmax = min(xmins), max(xmaxs)
    #     ax0.set_xlim(xmin, xmax)
    #     ax0.set_xticks(np.arange(xmin + (50 - xmin%50)%50, xmax+1, 50))
    #     ax0.legend(prop={'size': 18}, loc='upper right')

    # def _plot_profile(allele1_pred, allele2_pred, allele1_length, allele2_length, ref_label, alt_label, window_size, ax0):
    #     total_length = 1000
    #     C = total_length//2
    #     F = window_size//2
    #     if allele1_length < allele2_length:
    #         # INSERTION
    #         allele1_pred_plots = []
    #         allele1_pred_plots.append((list(range(-F, allele1_length)), allele1_pred[C-F:C+allele1_length]))
    #         allele1_pred_plots.append((list(range(allele2_length, F+allele2_length)),
    #                                 allele1_pred[C+allele1_length:C+F+allele1_length]))
    #         allele2_pred_plots = [(list(range(-F, F+allele2_length)), allele2_pred[C-F:C+F+allele2_length])]
    #         vlines = [-0.5, allele2_length-0.5]
    #     elif allele1_length > allele2_length:
    #         # DELETION
    #         allele1_pred_plots = [(list(range(-F, F+allele1_length)), allele1_pred[C-F:C+F+allele1_length])]
    #         allele2_pred_plots = []
    #         allele2_pred_plots.append((list(range(-F, allele2_length)), allele2_pred[C-F:C+allele2_length]))
    #         allele2_pred_plots.append((list(range(allele1_length, F+allele1_length)),
    #                                 allele2_pred[C+allele2_length:C+F+allele2_length]))
    #         vlines = [-0.5, allele1_length-0.5]
    #     else:
    #         # SUBSTITUTION
    #         allele1_pred_plots = [(list(range(-F, F+allele1_length)), allele1_pred[C-F:C+F+allele1_length])]
    #         allele2_pred_plots = [(list(range(-F, F+allele2_length)), allele2_pred[C-F:C+F+allele2_length])]
    #         vlines = [-0.5, +allele1_length-0.5]
    #     _plotter_profile(allele1_pred_plots, allele2_pred_plots, vlines, ax0, ref_label, alt_label)
    
    # def plot_variant(allele1_pred, allele2_pred, allele1_shap, allele2_shap, ref_length, alt_length, ref_label, alt_label, window_size, title, save_loc):
    #     fig, axs = plt.subplots(3, 1, figsize=(20, 8), dpi=400)
    #     # PLOT PROFILE
    #     _plot_profile(allele1_pred, allele2_pred, ref_length, alt_length, ref_label, alt_label, window_size, axs[0])
    #     # PLOT SHAP
    #     # _plot_shap(allele1_shap, allele2_shap, ref_length, alt_length, ref_label, alt_label, window_size, axs[1], axs[2])
    #     # PLOT
    #     plt.suptitle(title, fontsize=24)
    #     plt.subplots_adjust(hspace=0.3)
    #     plt.subplots_adjust(top=0.915)
    #     fig.set_facecolor('white')
    #     plt.savefig(save_loc, bbox_inches='tight')
    #     plt.close()

    # plot_dir = "/users/airanman/projects/variant-scorer/test_sample/ENCSR999NKW/viz"

    # def plot_variants(df, score_type='counts', fold='mean'):
    #     for index,row in df.iterrows():
    #         # sofar += 1
    #         # if sofar % 100 == 0:
    #         #     print(row['rsid'])
            
    #         # for cluster in [row['adult.logfc.best_cluster'], row['fetal.logfc.best_cluster']]:
    #         if not os.path.isdir(f"{plot_dir}/{score_type}"):
    #             os.mkdir(f"{plot_dir}/{score_type}")
            
    #         outfile = f"{plot_dir}/{score_type}/{row['variant_id']}.pdf"

    #         allele1_p = allele1_preds[fold][row['shap_index']]
    #         allele2_p = allele2_preds[fold][row['shap_index']]
    #         allele1_s = allele1_shap[score_type][fold][row['shap_index']]
    #         allele2_s = allele2_shap[score_type][fold][row['shap_index']]
    #         ref_length = len(row['ref'])
    #         alt_length = len(row['alt'])
    #         ref_label = str(row['ref'])
    #         alt_label = str(row['alt'])
    #         window_size = 300
    #         # title = f"{row['rsid']} ({row['chr']}:{row['pos']}:{row['ref']}:{row['alt']}) in {cluster}"
    #         title = f"{row['rsid']} ({row['chr']}:{row['pos']}:{row['ref']}:{row['alt']}) in {row}"
    #         plot_variant(allele1_p, allele2_p, allele1_s, allele2_s,
    #                     ref_length, alt_length, ref_label, alt_label,
    #                     window_size, title, outfile)
                
    #     return df
    
    # plot_variants(args.df, score_type='counts', fold='mean')

if __name__ == "__main__":
    main()
