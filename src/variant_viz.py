import pandas as pd
import numpy as np
import os
from collections import namedtuple
import operator
import logging
import deepdish as dd
from matplotlib import pyplot as plt
import h5py
import hdf5plugin

from utils.helpers import *

def print_h5_structure(file_name):
    def print_group(name, obj):
        indent = "  " * (name.count('/') - 1)
        print(f"{indent}{name}")
        if name == "observed/allele2_pred_profiles":
            print(obj[:])
        # print(obj[:])
        if isinstance(obj, h5py.Dataset):
            print(f"{indent}  Dataset: shape={obj.shape}, dtype={obj.dtype}")

    with h5py.File(file_name, 'r') as f:
        f.visititems(print_group)

def _plotter_profile(allele1_profiles, allele2_profiles, vlines, ax0, allele1_label, allele2_label):
    xmins, xmaxs = [], []
    for i, (x, allele1_profile) in enumerate(allele1_profiles):
        ax0.plot(x, allele1_profile, label=(f"ref ({allele1_label})" if i==0 else ""), color='C0')
        xmins.append(min(x)); xmaxs.append(max(x))
    for i, (x, allele2_profile) in enumerate(allele2_profiles):
        ax0.plot(x, allele2_profile, label=(f"alt ({allele2_label})" if i==0 else ""), color='C1')
        xmins.append(min(x)); xmaxs.append(max(x))
    for v in vlines:
        ax0.axvline(v, color='black', ls='--', linewidth=1)
    xmin, xmax = min(xmins), max(xmaxs)
    ax0.set_xlim(xmin, xmax)
    ax0.set_xticks(np.arange(xmin + (50 - xmin%50)%50, xmax+1, 50))
    ax0.legend(prop={'size': 18}, loc='upper right')


def _plotter_shap(allele1_shap, allele2_shap, vlines, xmin, ax1, ax2, allele1_label, allele2_label):
    active_allele = "ref" if np.sum(allele1_shap) > np.sum(allele2_shap) else "alt"
    df1 = pd.DataFrame(allele1_shap, columns=["A", "C", "G", "T"])
    df1.index +=  xmin
    df2 = pd.DataFrame(allele2_shap, columns=["A", "C", "G", "T"])
    df2.index += xmin
    logomaker.Logo(df1, ax=ax1)
    logomaker.Logo(df2, ax=ax2)
    for v in vlines:
        ax1.axvline(v, color='k', linestyle='--', linewidth=1)
        ax2.axvline(v, color='k', linestyle='--', linewidth=1)
    ymax = 1.1*max(np.max(np.maximum(allele1_shap, 0)), np.max(np.maximum(allele2_shap, 0)))
    ymin = 1.1*min(np.min(np.minimum(allele1_shap, 0)), np.min(np.minimum(allele2_shap, 0)))
    ax1.set_ylim(bottom=ymin, top=ymax)
    ax2.set_ylim(bottom=ymin, top=ymax)
    plt.text(0.988, 0.903, f"ref ({allele1_label})", 
                             verticalalignment='top', horizontalalignment='right',
                             transform=ax1.transAxes, size=18, color='black',
                             bbox=dict(boxstyle='round', facecolor='white', edgecolor='lightgrey'))
    plt.text(0.988, 0.903, f"alt ({allele2_label})", 
                             verticalalignment='top', horizontalalignment='right',
                             transform=ax2.transAxes, size=18, color='black',
                             bbox=dict(boxstyle='round', facecolor='white', edgecolor='lightgrey'))


def _plot_profile(allele1_pred, allele2_pred, allele1_length, allele2_length, allele1_label, allele2_label, window_size, ax0):
    total_length = 1000
    C = total_length//2
    F = window_size//2
    if allele1_length < allele2_length:
        # INSERTION
        allele1_pred_plots = []
        allele1_pred_plots.append((list(range(-F, allele1_length)), allele1_pred[C-F:C+allele1_length]))
        allele1_pred_plots.append((list(range(allele2_length, F+allele2_length)),
                                  allele1_pred[C+allele1_length:C+F+allele1_length]))
        allele2_pred_plots = [(list(range(-F, F+allele2_length)), allele2_pred[C-F:C+F+allele2_length])]
        vlines = [-0.5, allele2_length-0.5]
    elif allele1_length > allele2_length:
        # DELETION
        allele1_pred_plots = [(list(range(-F, F+allele1_length)), allele1_pred[C-F:C+F+allele1_length])]
        allele2_pred_plots = []
        allele2_pred_plots.append((list(range(-F, allele2_length)), allele2_pred[C-F:C+allele2_length]))
        allele2_pred_plots.append((list(range(allele1_length, F+allele1_length)),
                                  allele2_pred[C+allele2_length:C+F+allele2_length]))
        vlines = [-0.5, allele1_length-0.5]
    else:
        # SUBSTITUTION
        allele1_pred_plots = [(list(range(-F, F+allele1_length)), allele1_pred[C-F:C+F+allele1_length])]
        allele2_pred_plots = [(list(range(-F, F+allele2_length)), allele2_pred[C-F:C+F+allele2_length])]
        vlines = [-0.5, +allele1_length-0.5]
    _plotter_profile(allele1_pred_plots, allele2_pred_plots, vlines, ax0, allele1_label, allele2_label)


def _plot_shap(allele1_shap, allele2_shap, allele1_length, allele2_length, allele1_label, allele2_label, window_size, ax1, ax2):
    total_length = 2114
    C = total_length//2
    F = window_size//2
    if allele1_length < allele2_length:
        # INSERTION
        allele1_shap_plot = np.concatenate([allele1_shap[C-F:C+allele1_length],
                                           np.zeros((allele2_length-allele1_length, 4)),
                                           allele1_shap[C+allele1_length:C+F+allele1_length]])
        allele2_shap_plot = allele2_shap[C-F:C+F+allele2_length]
        vlines = [-0.5, allele2_length-0.5]
    elif allele1_length > allele2_length:
        # DELETION
        allele1_shap_plot = allele1_shap[C-F:C+F+allele1_length]
        allele2_shap_plot = np.concatenate([allele2_shap[C-F:C+allele2_length],
                                           np.zeros((allele1_length-allele2_length, 4)),
                                           allele2_shap[C+allele2_length:C+F+allele2_length]])
        vlines = [-0.5, allele1_length-0.5]
    else:
        # SUBSTITUTION
        allele1_shap_plot = allele1_shap[C-F:C+F+allele1_length]
        allele2_shap_plot = allele2_shap[C-F:C+F+allele2_length]
        vlines = [-0.5, allele1_length-0.5]
    assert(allele1_shap_plot.shape == allele2_shap_plot.shape)
    _plotter_shap(allele1_shap_plot, allele2_shap_plot, vlines, -F, ax1, ax2, allele1_label, allele2_label)

def plot_variant(allele1_pred, allele2_pred, allele1_shap, allele2_shap, allele1_length, allele2_length, allele1_label, allele2_label, window_size, title, save_loc):
	fig, axs = plt.subplots(3, 1, figsize=(20, 8), dpi=400)
	# PLOT PROFILE
	_plot_profile(allele1_pred, allele2_pred, allele1_length, allele2_length, allele1_label, allele2_label, window_size, axs[0])
	# PLOT SHAP
	_plot_shap(allele1_shap, allele2_shap, allele1_length, allele2_length, allele1_label, allele2_label, window_size, axs[1], axs[2])
	# PLOT
	plt.suptitle(title, fontsize=24)
	plt.subplots_adjust(hspace=0.3, top=0.785)
	fig.set_facecolor('white')
	plt.savefig(save_loc, bbox_inches='tight')
	plt.close()

def main(args = None):
    # print_h5_structure("/users/airanman/proj/variant-scorer/test_sample/ENCSR999NKW/score/test.0.variant_predictions.h5")

    if args is None:
        logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')
        args = fetch_viz_args()

    is_using_scoring_file_overrides = False
    filter_output_filenames = [ get_score_file_path(args.filter_dir, args.model_name, index, is_filter_step=True) for index in range(len(args.models)) ]
    print(filter_output_filenames)
    if args.filter_score_filenames is not None:
        if len(args.filter_score_filenames) != len(args.models):
            raise ValueError("Number of models and score filenames do not match, which they are requird to do if the --score-filenames flag is given. Exiting.")
        else:
            filter_output_filenames = [ os.path.join(args.filter_dir, args.filter_score_filenames[i]) for i in range(len(args.models)) ]
            is_using_scoring_file_overrides = True
    scoring_model_and_output_prefixes = list(zip(args.models, filter_output_filenames))

    is_using_shap_file_overrides = False
    shap_output_names = [ get_shap_output_file_prefix(args.shap_dir, args.model_name, index) for index in range(len(args.models)) ]
    if args.filter_score_filenames is not None:
        if len(args.filter_score_filenames) != len(args.models):
            raise ValueError("Number of models and score filenames do not match, which they are requird to do if the --score-filenames flag is given. Exiting.")
        else:
            shap_output_names = [ os.path.join(args.shap_dir, args.filter_score_filenames[i]) for i in range(len(args.models)) ]
            is_using_shap_file_overrides = True
    shap_model_and_output_prefixes = list(zip(args.models, shap_output_names))

    shap_types = args.shap_type
    scoring_and_shap_output_prefixes = zip(filter_output_filenames, shap_output_names)

    # Verify that the scoring file, the shap file, and the (filtered) annotations file have the same variants_ids
    for model_index, (scoring_output_prefix, shap_output_prefix) in enumerate(scoring_and_shap_output_prefixes):

        predictions_hdf5_filepath = f"{scoring_output_prefix}.variant_predictions.h5"
        score_file = h5py.File(predictions_hdf5_filepath, 'r')

        for shap_type in shap_types:
            shap_filepath = f"{shap_output_prefix}.variant_shap.{shap_type}.h5"
            shap_file = h5py.File(shap_filepath, 'r')

            score_variants_ids = score_file['observed']['variant_ids']
            shap_variant_ids = shap_file['variant_ids']

            for index in range(len(scoring_model_and_output_prefixes)):
                if not np.array_equal(score_variants_ids, shap_allele1_variant_ids) or not np.array_equal(score_variants_ids, shap_allele2_variant_ids):
                    print(f"Error: The variant_ids in the scoring file {predictions_hdf5_filepath} and the shap file {shap_filepath} do not match. Exiting.")
                    exit()

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
    for shap_type in shap_types:
        allele1_shap[shap_type] = {}
        allele2_shap[shap_type] = {}
        for model_index, (model_name, shap_output_prefix) in enumerate(shap_model_and_output_prefixes):
            # TODO: generalize this to other shap_types
            shap_filepath = f"{shap_output_prefix}.variant_shap.{shap_type}.h5"
            if args.shap_override is not None:
                shap_filepath = f"{args.shap_override}.h5"
            if not os.path.isfile(shap_filepath):
                raise ValueError(f"Error: The file {shap_filepath} doesn't exist. Exiting.")
            logging.debug(f"Reading shap file {shap_filepath}")
            # file = h5py.File(shap_filepath, 'r')
            file = dd.io.load(shap_filepath)
            logging.debug(f"Opened file {shap_filepath} successfully.")
            allele1_shap[shap_type][model_index] = np.array(file['allele1']['projected_shap']['seq'])
            allele2_shap[shap_type][model_index] = np.array(file['allele2']['projected_shap']['seq'])

    allele1_shap[shap_type]['mean'] = np.mean(np.array([allele1_shap[shap_type][model_index] for model_index in range(total_models)]), axis=0)
    allele2_shap[shap_type]['mean'] = np.mean(np.array([allele2_shap[shap_type][model_index] for model_index in range(total_models)]), axis=0)

    # Load the filtered annotations file
    annotations_filename = get_filter_output_file(args.filter_dir, args.model_name)
    annotations_df = pd.read_csv(annotations_filename, sep="\t")

    for index, annotation in annotations_df.iterrows():
        score_type = 'counts'
        cur_allele1_preds = allele1_preds[index]
        cur_allele2_preds = allele2_preds[index]
        cur_allele1_shap = allele1_shap[score_type][index]
        cur_allele2_shap = allele2_shap[score_type][index]
        ref_length = len(annotation['allele1'])
        alt_length = len(annotation['allele2'])
        ref_label = str(annotation['allele1'])
        alt_label = str(annotation['allele2'])
        variant_id = annotation['variant_id']

        plot_variant(cur_allele1_preds, cur_allele2_preds, cur_allele1_shap, cur_allele2_shap, ref_length, alt_length, ref_label, alt_label, 300, variant_id, variant_id)

        # annotations_df[annotation] = annotations_df[annotation].astype(str)
    # plot_result = plot_variants(annotations_df, score_type='counts', fold='mean')

    # def _plotter_profile(allele1_profiles, allele2_profiles, vlines, ax0, ref_label, alt_label):
    #     xmins, xmaxs = [], []
    #     for i, (x, allele1_profile) in enumerate(allele1_profiles):
    #         # ax0.plot(x, allele1_profile, label=(f"ref ({ref_label})" if i==0 else ""), color='C0')
    #         ax0.plot(x, allele1_profile, label=(f"allele1 ({ref_label})" if i==0 else ""), color='C0')
    #         xmins.append(min(x)); xmaxs.append(max(x))
    #     for i, (x, allele2_profile) in enumerate(allele2_profiles):
    #         # ax0.plot(x, allele2_profile, label=(f"alt ({alt_label})" if i==0 else ""), color='C1')
    #         ax0.plot(x, allele2_profile, label=(f"allele2 ({alt_label})" if i==0 else ""), color='C1')
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
    #         target_dir = f"{plot_dir}/{score_type}"
    #         print(target_dir)
    #         if not os.path.isdir(target_dir):
    #             os.makedirs(target_dir)
            
    #         outfile = f"{plot_dir}/{score_type}/{row['variant_id']}.pdf"

    #         # allele1_p = allele1_preds[fold][row['shap_index']]
    #         # allele2_p = allele2_preds[fold][row['shap_index']]
    #         # allele1_s = allele1_shap[score_type][fold][row['shap_index']]
    #         # allele2_s = allele2_shap[score_type][fold][row['shap_index']]
    #         allele1_p = allele1_preds[fold][index]
    #         allele2_p = allele2_preds[fold][index]
    #         allele1_s = allele1_shap[score_type][fold][index]
    #         allele2_s = allele2_shap[score_type][fold][index]
    #         print(row)
    #         # ref_length = len(row['ref'])
    #         # alt_length = len(row['alt'])
    #         # ref_label = str(row['ref'])
    #         # alt_label = str(row['alt'])
    #         ref_length = len(row['allele1'])
    #         alt_length = len(row['allele2'])
    #         ref_label = str(row['allele1'])
    #         alt_label = str(row['allele2'])
    #         window_size = 300
    #         # title = f"{row['rsid']} ({row['chr']}:{row['pos']}:{row['ref']}:{row['alt']}) in {cluster}"
    #         title = f"{row['variant_id']} ({row['chr']}:{row['pos']}:{row['allele1']}:{row['allele2']}) in {row}"
    #         plot_variant(allele1_p, allele2_p, allele1_s, allele2_s,
    #                     ref_length, alt_length, ref_label, alt_label,
    #                     window_size, title, outfile)
                
    #     return df

if __name__ == "__main__":
    main()
