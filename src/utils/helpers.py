from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.models import load_model
import tensorflow as tf
from scipy.spatial.distance import jensenshannon
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
sys.path.append('..')
from generators.variant_generator import VariantGenerator
from generators.peak_generator import PeakGenerator
from utils import losses
import logging
import os
from typing import List, Tuple
import pybedtools
import os
import tempfile
from multiprocessing import Pool

DEFAULT_ADASTRA_THREADS = 4
MODEL_ID_COL = 'model_id'
PEAKS_PATH_COL = 'peaks_path'
SUMMARIZE_OUT_PATH_COL = 'summarize_output_path'
ANNOTATE_SUMM_OUT_PATH_COL = 'annotate_summ_output_path'
AGGREGATE_OUT_PATH_COL = 'aggregate_output_path'
ANNOTATE_AGGR_OUT_PATH_COL = 'annotate_aggr_output_path'

def get_variant_schema(schema):
    var_SCHEMA = {'original': ['chr', 'pos', 'variant_id', 'allele1', 'allele2'],
                  'plink': ['chr', 'variant_id', 'ignore1', 'pos', 'allele1', 'allele2'],
                  'plink2': ['chr', 'variant_id', 'pos', 'allele1', 'allele2'],
                  'bed': ['chr', 'pos', 'end', 'allele1', 'allele2', 'variant_id'],
                  'chrombpnet': ['chr', 'pos', 'allele1', 'allele2', 'variant_id']}
    return var_SCHEMA[schema]

def get_peak_schema(schema):
    PEAK_SCHEMA = {'narrowpeak': ['chr', 'start', 'end', 'peak_id', 'peak_score',
                                  5, 6, 7, 'rank', 'summit']}
    return PEAK_SCHEMA[schema]

def get_valid_peaks(chrom, pos, summit, input_len, chrom_sizes_dict):
    valid_chrom = chrom in chrom_sizes_dict
    if valid_chrom:
        flank = input_len // 2
        lower_check = ((pos + summit) - flank > 0)
        upper_check = ((pos + summit) + flank <= chrom_sizes_dict[chrom])
        in_bounds = lower_check and upper_check
        valid_peak = valid_chrom and in_bounds
        return valid_peak
    else:
        return False

def get_valid_variants(chrom, pos, allele1, allele2, input_len, chrom_sizes_dict):
    valid_chrom = chrom in chrom_sizes_dict
    # logging.debug(f"chrom: {chrom}, pos: {pos}, valid_chrom: {valid_chrom}")
    if valid_chrom:
        flank = input_len // 2
        lower_check = (pos - flank > 0)
        upper_check = (pos + flank <= chrom_sizes_dict[chrom])
        in_bounds = lower_check and upper_check
        # no_allele1_indel = (len(allele1) == 1)
        # no_allele2_indel = (len(allele2) == 1)
        # no_indel = no_allele1_indel and no_allele2_indel
        # valid_variant = valid_chrom and in_bounds and no_indel
        # logging.debug(f"chr: {chrom}, pos: {pos}, valid_chrom: {valid_chrom}, in_bounds: {in_bounds}")
        valid_variant = valid_chrom and in_bounds
        return valid_variant
    else:
        return False

def softmax(x, temp=1):
    norm_x = x - np.mean(x, axis=1, keepdims=True)
    return np.exp(temp*norm_x)/np.sum(np.exp(temp*norm_x), axis=1, keepdims=True)

def load_model_wrapper(model_file):
    # read .h5 model
    custom_objects = {"multinomial_nll": losses.multinomial_nll, "tf": tf}
    get_custom_objects().update(custom_objects)
    model = load_model(model_file, compile=False)
    logging.debug(f"Model {model_file} loaded successfully.")
    return model

def fetch_peak_predictions(model, peaks, input_len, genome_fasta, batch_size, model_architecture, debug_mode=False, forward_only=False):
    peak_ids = []
    pred_counts = []
    pred_profiles = []
    if not forward_only:
        revcomp_counts = []
        revcomp_profiles = []

    # peak sequence generator
    peak_gen = PeakGenerator(peaks=peaks,
                             input_len=input_len,
                             genome_fasta=genome_fasta,
                             batch_size=batch_size,
                             debug_mode=debug_mode)

    for i in tqdm(range(len(peak_gen))):
        batch_peak_ids, seqs = peak_gen[i]
        revcomp_seq = seqs[:, ::-1, ::-1]

        if model_architecture == 'chrombpnet':
            batch_preds = model.predict(seqs, verbose=False)
            if not forward_only:
                revcomp_batch_preds = model.predict(revcomp_seq, verbose=False)
        elif model_architecture == 'chrombpnet-lite':
            batch_preds = model.predict([seqs,
                                         np.zeros((len(seqs), model.output_shape[0][1])),
                                         np.zeros((len(seqs), ))],
                                        verbose=False)
            if not forward_only:
                revcomp_batch_preds = model.predict([revcomp_seq,
                                             np.zeros((len(revcomp_seq), model.output_shape[0][1])),
                                             np.zeros((len(revcomp_seq), ))],
                                            verbose=False)
        elif model_architecture == 'bpnet':
            batch_preds = model.predict([seqs,
                                         np.zeros((len(seqs), model.output_shape[0][1], 2)),
                                         np.zeros((len(seqs),2))],
                                        verbose=False)
            if not forward_only:
                revcomp_batch_preds = model.predict([revcomp_seq,
                                             np.zeros((len(revcomp_seq), model.output_shape[0][1],2)),
                                             np.zeros((len(revcomp_seq),2))],
                                            verbose=False)
        else:
            raise ValueError(f"Model architecture {model_architecture} not supported")

        batch_preds[1] = np.array([batch_preds[1][i] for i in range(len(batch_preds[1]))])
        pred_counts.extend(np.exp(batch_preds[1]))
        if model_architecture in ['chrombpnet', 'chrombpnet-lite']:
           pred_profiles.extend(np.array(batch_preds[0]))   # np.squeeze(softmax()) to get probability profile
        elif model_architecture == 'bpnet':
            batch_preds_profile = np.array(batch_preds[0])
            pred_profiles.extend(batch_preds_profile.reshape((batch_preds_profile.shape[0],batch_preds_profile.shape[1]*batch_preds_profile.shape[2],1),order="F"))

        if not forward_only:
            revcomp_batch_preds[1] = np.array([revcomp_batch_preds[1][i] for i in range(len(revcomp_batch_preds[1]))])
            revcomp_counts.extend(np.exp(revcomp_batch_preds[1]))
            if model_architecture in ['chrombpnet', 'chrombpnet-lite']:
                revcomp_profiles.extend(np.array(revcomp_batch_preds[0]))
            elif model_architecture == 'bpnet':
                revcomp_batch_preds_profile = np.array(revcomp_batch_preds[0])
                revcomp_profiles.extend(revcomp_batch_preds_profile.reshape((revcomp_batch_preds_profile.shape[0],revcomp_batch_preds_profile.shape[1]*revcomp_batch_preds_profile.shape[2],1),order="F"))

        peak_ids.extend(batch_peak_ids)

    peak_ids = np.array(peak_ids)
    pred_counts = np.array(pred_counts)
    pred_profiles = np.array(pred_profiles)

    if not forward_only:
        revcomp_counts = np.array(revcomp_counts)
        revcomp_profiles = np.array(revcomp_profiles)
        average_counts = np.average([pred_counts,revcomp_counts],axis=0)
        average_profiles = np.average([pred_profiles,revcomp_profiles[:,::-1]],axis=0)
        return peak_ids,average_counts,average_profiles
    else:
        return peak_ids,pred_counts,pred_profiles

def fetch_variant_predictions(model, variants_table, input_len, genome_fasta, batch_size, model_architecture, debug_mode=False, shuf=False, forward_only=False):
    variant_ids = []
    allele1_pred_counts = []
    allele2_pred_counts = []
    allele1_pred_profiles = []
    allele2_pred_profiles = []
    if not forward_only:
        revcomp_allele1_pred_counts = []
        revcomp_allele2_pred_counts = []
        revcomp_allele1_pred_profiles = []
        revcomp_allele2_pred_profiles = []

    # variant sequence generator
    var_gen = VariantGenerator(variants_table=variants_table,
                           input_len=input_len,
                           genome_fasta=genome_fasta,
                           batch_size=batch_size,
                           debug_mode=False,
                           shuf=shuf)

    for i in tqdm(range(len(var_gen))):

        batch_variant_ids, allele1_seqs, allele2_seqs = var_gen[i]
        revcomp_allele1_seqs = allele1_seqs[:, ::-1, ::-1]
        revcomp_allele2_seqs = allele2_seqs[:, ::-1, ::-1]

        if model_architecture == 'chrombpnet':
            allele1_batch_preds = model.predict(allele1_seqs, verbose=False)
            allele2_batch_preds = model.predict(allele2_seqs, verbose=False)
            if not forward_only:
                revcomp_allele1_batch_preds = model.predict(revcomp_allele1_seqs, verbose=False)
                revcomp_allele2_batch_preds = model.predict(revcomp_allele2_seqs, verbose=False)
        elif model_architecture == 'chrombpnet-lite':
            allele1_batch_preds = model.predict([allele1_seqs,
                                                 np.zeros((len(allele1_seqs), model.output_shape[0][1])),
                                                 np.zeros((len(allele1_seqs), ))],
                                                verbose=False)
            allele2_batch_preds = model.predict([allele2_seqs,
                                                 np.zeros((len(allele2_seqs), model.output_shape[0][1])),
                                                 np.zeros((len(allele2_seqs), ))],
                                                verbose=False)
            if not forward_only:
                revcomp_allele1_batch_preds = model.predict([revcomp_allele1_seqs,
                                                     np.zeros((len(revcomp_allele1_seqs), model.output_shape[0][1])),
                                                     np.zeros((len(revcomp_allele1_seqs), ))],
                                                    verbose=False)
                revcomp_allele2_batch_preds = model.predict([revcomp_allele2_seqs,
                                         np.zeros((len(revcomp_allele2_seqs), model.output_shape[0][1])),
                                         np.zeros((len(revcomp_allele2_seqs), ))],
                                        verbose=False)
        elif model_architecture == 'bpnet':
            print("fetch_variant_prediction, bpnet")
            allele1_batch_preds = model.predict([allele1_seqs,
                                                 np.zeros((len(allele1_seqs), model.output_shape[0][1], 2)),
                                                 np.zeros((len(allele1_seqs), 2))],
                                                verbose=False)
            allele2_batch_preds = model.predict([allele2_seqs,
                                                 np.zeros((len(allele2_seqs), model.output_shape[0][1], 2)),
                                                 np.zeros((len(allele2_seqs), 2))],
                                                verbose=False)
            if not forward_only:
                revcomp_allele1_batch_preds = model.predict([revcomp_allele1_seqs,
                                                     np.zeros((len(revcomp_allele1_seqs), model.output_shape[0][1], 2)),
                                                     np.zeros((len(revcomp_allele1_seqs), 2))],
                                                    verbose=False)
                revcomp_allele2_batch_preds = model.predict([revcomp_allele2_seqs,
                                         np.zeros((len(revcomp_allele2_seqs), model.output_shape[0][1], 2)),
                                         np.zeros((len(revcomp_allele2_seqs), 2))],
                                        verbose=False)
        else:
            raise ValueError(f"Model architecture {model_architecture} not supported")

        allele1_batch_preds[1] = np.array([allele1_batch_preds[1][i] for i in range(len(allele1_batch_preds[1]))])
        allele2_batch_preds[1] = np.array([allele2_batch_preds[1][i] for i in range(len(allele2_batch_preds[1]))])
        allele1_pred_counts.extend(np.exp(allele1_batch_preds[1]))
        allele2_pred_counts.extend(np.exp(allele2_batch_preds[1]))
        if model.architecture in ['chrombpnet', 'chrombpnet-lite']:
            allele1_pred_profiles.extend(np.array(allele1_batch_preds[0]))   # np.squeeze(softmax()) to get probability profile
            allele2_pred_profiles.extend(np.array(allele2_batch_preds[0]))
        elif model.architecture == 'bpnet':
            allele1_batch_preds_profile = np.array(allele1_batch_preds[0])
            allele2_batch_preds_profile = np.array(allele2_batch_preds[0])
            allele1_pred_profiles.extend(allele1_batch_preds_profile.reshape((allele1_batch_preds_profile.shape[0],allele1_batch_preds_profile.shape[1]*allele1_batch_preds_profile.shape[2],1),order="F"))
            allele2_pred_profiles.extend(allele2_batch_preds_profile.reshape((allele2_batch_preds_profile.shape[0],allele2_batch_preds_profile.shape[1]*allele2_batch_preds_profile.shape[2],1),order="F"))
        else:
            raise ValueError(f"Model architecture {model_architecture} not supported")

        if not forward_only:
            revcomp_allele1_batch_preds[1] = np.array([revcomp_allele1_batch_preds[1][i] for i in range(len(revcomp_allele1_batch_preds[1]))])
            revcomp_allele2_batch_preds[1] = np.array([revcomp_allele2_batch_preds[1][i] for i in range(len(revcomp_allele2_batch_preds[1]))])
            revcomp_allele1_pred_counts.extend(np.exp(revcomp_allele1_batch_preds[1]))
            revcomp_allele2_pred_counts.extend(np.exp(revcomp_allele2_batch_preds[1]))
            if model.architecture in ['chrombpnet', 'chrombpnet-lite']:
                revcomp_allele1_pred_profiles.extend(np.array(revcomp_allele1_batch_preds[0]))
                revcomp_allele2_pred_profiles.extend(np.array(revcomp_allele2_batch_preds[0]))
            elif model.architecture == 'bpnet':
                revcomp_allele1_batch_preds_profile = np.array(revcomp_allele1_batch_preds[0])
                revcomp_allele2_batch_preds_profile = np.array(revcomp_allele2_batch_preds[0])
                revcomp_allele1_pred_profiles.extend(revcomp_allele1_batch_preds_profile.reshape((revcomp_allele1_batch_preds_profile.shape[0],revcomp_allele1_batch_preds_profile.shape[1]*revcomp_allele1_batch_preds_profile.shape[2],1),order="F"))
                revcomp_allele2_pred_profiles.extend(revcomp_allele2_batch_preds_profile.reshape((revcomp_allele2_batch_preds_profile.shape[0],revcomp_allele2_batch_preds_profile.shape[1]*revcomp_allele2_batch_preds_profile.shape[2],1),order="F"))
            else:
                raise ValueError(f"Model architecture {model_architecture} not supported")

        variant_ids.extend(batch_variant_ids)

    variant_ids = np.array(variant_ids)
    allele1_pred_counts = np.array(allele1_pred_counts)
    allele2_pred_counts = np.array(allele2_pred_counts)
    allele1_pred_profiles = np.array(allele1_pred_profiles)
    allele2_pred_profiles = np.array(allele2_pred_profiles)

    if not forward_only:
        revcomp_allele1_pred_counts = np.array(revcomp_allele1_pred_counts)
        revcomp_allele2_pred_counts = np.array(revcomp_allele2_pred_counts)
        revcomp_allele1_pred_profiles = np.array(revcomp_allele1_pred_profiles)
        revcomp_allele2_pred_profiles = np.array(revcomp_allele2_pred_profiles)
        average_allele1_pred_counts = np.average([allele1_pred_counts,revcomp_allele1_pred_counts],axis=0)
        average_allele1_pred_profiles = np.average([allele1_pred_profiles,revcomp_allele1_pred_profiles[:,::-1]],axis=0)
        average_allele2_pred_counts = np.average([allele2_pred_counts,revcomp_allele2_pred_counts],axis=0)
        average_allele2_pred_profiles = np.average([allele2_pred_profiles,revcomp_allele2_pred_profiles[:,::-1]],axis=0)
        return variant_ids, average_allele1_pred_counts, average_allele2_pred_counts, \
               average_allele1_pred_profiles, average_allele2_pred_profiles
    else:
        return variant_ids, allele1_pred_counts, allele2_pred_counts, \
               allele1_pred_profiles, allele2_pred_profiles

def get_variant_scores_with_peaks(allele1_pred_counts, allele2_pred_counts,
                       allele1_pred_profiles, allele2_pred_profiles, pred_counts):
    # logfc = np.log2(allele2_pred_counts / allele1_pred_counts)
    # jsd = np.array([jensenshannon(x,y,base=2.0) for x,y in zip(allele2_pred_profiles, allele1_pred_profiles)])

    logfc, jsd = get_variant_scores(allele1_pred_counts, allele2_pred_counts,
                                    allele1_pred_profiles, allele2_pred_profiles)
    allele1_quantile = np.array([np.max([np.mean(pred_counts < x), (1/len(pred_counts))]) for x in allele1_pred_counts])
    allele2_quantile = np.array([np.max([np.mean(pred_counts < x), (1/len(pred_counts))]) for x in allele2_pred_counts])

    return logfc, jsd, allele1_quantile, allele2_quantile

def get_variant_scores(allele1_pred_counts, allele2_pred_counts,
                       allele1_pred_profiles, allele2_pred_profiles):

    print('allele1_pred_counts shape:', allele1_pred_counts.shape)
    print('allele2_pred_counts shape:', allele2_pred_counts.shape)
    print('allele1_pred_profiles shape:', allele1_pred_profiles.shape)
    print('allele2_pred_profiles shape:', allele2_pred_profiles.shape)

    logfc = np.squeeze(np.log2(allele2_pred_counts / allele1_pred_counts))
    jsd = np.squeeze([jensenshannon(x, y, base=2.0)
                     for x,y in zip(softmax(allele2_pred_profiles),
                                    softmax(allele1_pred_profiles))])

    print('logfc shape:', logfc.shape)
    print('jsd shape:', jsd.shape)

    return logfc, jsd

def adjust_indel_jsd(variants_table,allele1_pred_profiles,allele2_pred_profiles,original_jsd):
    indel_idx = []
    for i, row in variants_table.iterrows():
        allele1, allele2 = row[['allele1','allele2']]
        if allele1 in ["-", "*"]:
            allele1 = ""
        if allele2 in ["-", "*"]:
            allele2 = ""
        if len(allele1) != len(allele2):
            indel_idx += [i]

    adjusted_jsd = []
    for i in indel_idx:
        row = variants_table.iloc[i]
        allele1, allele2 = row[['allele1','allele2']]
        if allele1 in ["-", "*"]:
            allele1 = ""
        if allele2 in ["-", "*"]:
            allele2 = ""

        allele1_length = len(allele1)
        allele2_length = len(allele2)

        allele1_p = allele1_pred_profiles[i]
        allele2_p = allele2_pred_profiles[i]
        assert len(allele1_p) == len(allele2_p)
        assert allele1_length != allele2_length
        flank_size = len(allele1_p)//2
        allele1_left_flank = allele1_p[:flank_size]
        allele2_left_flank = allele2_p[:flank_size]

        if allele1_length > allele2_length:
            allele1_right_flank = np.concatenate([allele1_p[flank_size:flank_size+allele2_length],allele1_p[flank_size+allele1_length:]])
            allele2_right_flank = allele2_p[flank_size:allele2_length-allele1_length]
        else:
            allele1_right_flank = allele1_p[flank_size:allele1_length-allele2_length]
            allele2_right_flank = np.concatenate([allele2_p[flank_size:flank_size+allele1_length], allele2_p[flank_size+allele2_length:]])


        adjusted_allele1_p = np.concatenate([allele1_left_flank,allele1_right_flank])
        adjusted_allele2_p = np.concatenate([allele2_left_flank,allele2_right_flank])
        adjusted_allele1_p = adjusted_allele1_p/np.sum(adjusted_allele1_p)
        adjusted_allele2_p = adjusted_allele2_p/np.sum(adjusted_allele2_p)
        assert len(adjusted_allele1_p) == len(adjusted_allele2_p)
        adjusted_j = jensenshannon(adjusted_allele1_p,adjusted_allele2_p,base=2.0)
        adjusted_jsd += [adjusted_j]

    adjusted_jsd_list = original_jsd.copy()
    if len(indel_idx) > 0:
        for i in range(len(indel_idx)):
            idx = indel_idx[i]
            adjusted_jsd_list[idx] = adjusted_jsd[i]

    return indel_idx, adjusted_jsd_list


def load_variant_table(table_path, schema=None):
    variants_table = None
    if schema is not None:
        variants_table = pd.read_csv(table_path, header=None, sep='\t', names=get_variant_schema(schema))
    else:
        variants_table = pd.read_table(table_path, header=0, sep='\t')
    variants_table.drop(columns=[str(x) for x in variants_table.columns if str(x).startswith('ignore')], inplace=True)
    variants_table['chr'] = variants_table['chr'].astype(str)
    has_chr_prefix = any('chr' in x.lower() for x in variants_table['chr'].tolist())
    if not has_chr_prefix:
        variants_table['chr'] = 'chr' + variants_table['chr']
    if schema == "bed":
        variants_table['pos'] = variants_table['pos'] + 1
    return variants_table

def create_shuffle_table(variants_table, random_seed=None, total_shuf=None, num_shuf=None):
    if total_shuf != None:
        if len(variants_table) > total_shuf:
            shuf_variants_table = variants_table.sample(total_shuf,
                                                        random_state=random_seed,
                                                        ignore_index=True,
                                                        replace=False)
        else:
            shuf_variants_table = variants_table.sample(total_shuf,
                                                        random_state=random_seed,
                                                        ignore_index=True,
                                                        replace=True)
        shuf_variants_table['random_seed'] = np.random.permutation(len(shuf_variants_table))
    else:
        if num_shuf != None:
            total_shuf = len(variants_table) * num_shuf
            shuf_variants_table = variants_table.sample(total_shuf,
                                                        random_state=random_seed,
                                                        ignore_index=True,
                                                        replace=True)
            shuf_variants_table['random_seed'] = np.random.permutation(len(shuf_variants_table))
        else:
            ## empty dataframe
            shuf_variants_table = pd.DataFrame()
    return shuf_variants_table

def get_pvals(obs, bg, tail):
    sorted_bg = np.sort(bg)
    if tail == 'right' or tail == 'both':
        rank_right = len(sorted_bg) - np.searchsorted(sorted_bg, obs, side='left')
        pval_right = (rank_right + 1) / (len(sorted_bg) + 1)
        if tail == 'right':
            return pval_right
    if tail == 'left' or tail == 'both':
        rank_left = np.searchsorted(sorted_bg, obs, side='right')
        pval_left = (rank_left + 1) / (len(sorted_bg) + 1)
        if tail == 'left':
            return pval_left
    assert tail == 'both'
    min_pval = np.minimum(pval_left, pval_right)
    pval_both = min_pval * 2

    return pval_both

def geo_mean_overflow(iterable,axis=0):
    return np.exp(np.log(iterable).mean(axis=0))

def add_missing_columns_to_peaks_df(peaks, schema):
    if schema != 'narrowpeak':
        raise ValueError("Schema not supported")
    
    required_columns = get_peak_schema(schema)
    num_current_columns = peaks.shape[1]
    
    if num_current_columns == 10:
        peaks.columns = required_columns[:num_current_columns]
        return peaks  # No missing columns, return as is

    elif num_current_columns < 3:
        raise ValueError("Peaks dataframe has fewer than 3 columns, which is invalid")
    
    elif num_current_columns > 10:
        raise ValueError("Peaks dataframe has greater than 10 columns, which is invalid")
    
    # Add missing columns to reach a total of 10 columns
    peaks.columns = required_columns[:num_current_columns]
    columns_to_add = required_columns[num_current_columns:]
    
    for column in columns_to_add:
        peaks[column] = '.'
    
    # Calculate the summit column
    peaks['summit'] = (peaks['end'] - peaks['start']) // 2
    
    return peaks

def get_score_dir(score_out_prefix):
    return os.path.dirname(score_out_prefix)

def get_score_file_path(score_out_prefix, chr=None):
    if chr is None:
        return f"{score_out_prefix}variant_scores.tsv"
    else:
        return f"{score_out_prefix}chr{str(chr)}.variant_scores.tsv"
    
def get_score_peaks_path(score_out_prefix):
    return f"{score_out_prefix}peak_scores.tsv"

def get_score_shuffled_path(score_out_prefix):
    return f"{score_out_prefix}variant_scores.shuffled.tsv"

def get_profiles_file_path(score_out_prefix):
    return f"{score_out_prefix}variant_predictions.tsv"

def get_annotate_output_file(annotate_dir, model_name):
    return f"{os.path.join(annotate_dir, model_name)}.annotations.tsv"

def get_filter_output_file(annotate_dir, model_name):
    return f"{os.path.join(annotate_dir, model_name)}.annotations.filtered.tsv"

def get_shap_output_file_prefix(shap_dir, model_name, model_index):
    return f"{os.path.join(shap_dir, model_name)}.{model_index}"

def add_n_closest_elements(variant_scores: pd.DataFrame, closest_n_elements_args: List[Tuple[str, int, str]], bed_headers: List[str]):
    # Add closest elements to the variant_scores dataframe.
    variant_bed = pybedtools.BedTool.from_dataframe(bed_headers)
    for elements_file, n_elements, element_label in closest_n_elements_args:
        logging.info(f"Annotating with closest {n_elements} elements")
        element_df = pd.read_table(elements_file, header=None)
        element_bed = pybedtools.BedTool.from_dataframe(element_df)
        closest_elements_bed = variant_bed.closest(element_bed, d=True, t='first', k=n_elements)

        closest_element_df = closest_elements_bed.to_dataframe(header=None)
        if not closest_element_df.empty:
            logging.debug(f"Closest elements ({element_label}) table:\n{closest_element_df.shape}\n{closest_element_df.head()}")

            closest_elements = {}
            element_dists = {}

            for index, row in closest_element_df.iterrows():
                if not row[5] in closest_elements:
                    closest_elements[row[5]] = []
                    element_dists[row[5]] = []
                closest_elements[row[5]].append(row.iloc[9])
                element_dists[row[5]].append(row.iloc[-1])

            closest_element_df = closest_element_df.rename({5: 'variant_id'}, axis=1)
            closest_element_df = closest_element_df[['variant_id']]

            for i in range(n_elements):
                closest_element_df[f'closest_{element_label}_{i+1}'] = closest_element_df['variant_id'].apply(lambda x: closest_elements[x][i] if len(closest_elements[x]) > i else '.')
                closest_element_df[f'closest_{element_label}_{i+1}_distance'] = closest_element_df['variant_id'].apply(lambda x: element_dists[x][i] if len(closest_elements[x]) > i else '.')

            closest_element_df.drop_duplicates(inplace=True)
        else:  
            # Make empty columns if no elements are found.
            closest_element_df = pd.DataFrame(columns=['variant_id'])
            for i in range(n_elements):
                closest_element_df[f'closest_{element_label}_{i+1}'] = ''
                closest_element_df[f'closest_{element_label}_{i+1}_distance'] = ''
        variant_scores = variant_scores.merge(closest_element_df, on='variant_id', how='left')
    return variant_scores


def add_closest_elements_in_window(variant_scores: pd.DataFrame, closest_elements_in_window_args: List[Tuple[str, int, str]], bed_headers: List[str]):
    # Add closest elements within a window to the variant_scores dataframe.
    variant_bed = pybedtools.BedTool.from_dataframe(bed_headers)
    for elements_file, window_size, element_label in closest_elements_in_window_args:
        logging.info("Annotating with closest elements within window size")
        element_df = pd.read_table(elements_file, header=None)
        element_bed = pybedtools.BedTool.from_dataframe(element_df)
        closest_elements_bed = variant_bed.window(element_bed, w=window_size)
        closest_element_df = closest_elements_bed.to_dataframe(header=None)
        if not closest_element_df.empty:
            closest_element_df = closest_element_df.rename({5: 'variant_id', 9: 'a_closest_element'}, axis=1)

            closest_elements = {}
            for index, row in closest_element_df.iterrows():
                variant_id = row['variant_id']
                element_name = row['a_closest_element']

                if variant_id not in closest_elements:
                    closest_elements[variant_id] = []
                closest_elements[variant_id].append(element_name)


            logging.debug(f"Closest elements ({element_label}) within window table:\n{closest_element_df.shape}\n{closest_element_df.head()}")
            closest_element_df = closest_element_df[['variant_id']]
            output_label = f"{element_label}_within_{window_size}_bp"
            closest_element_df[output_label] = closest_element_df['variant_id'].apply(
                lambda x: ', '.join(closest_elements[x]) if x in closest_elements else ''
            )

            closest_element_df.drop_duplicates(inplace=True)
        else:
            # Make empty column if no elements are within the window.
            closest_element_df = pd.DataFrame(columns=['variant_id', f"{element_label}_within_{window_size}_bp"])
            closest_element_df['variant_id'] = variant_scores['variant_id']
            closest_element_df[f"{element_label}_within_{window_size}_bp"] = ''

        variant_scores = variant_scores.merge(closest_element_df[['variant_id', output_label]], on='variant_id', how='left')
    return variant_scores


def add_r2(variant_scores: pd.DataFrame, r2_ld_filepath: str):
    logging.info("Annotating with r2")
    # Make temp
    # r2_tsv_filepath = os
    # with open(r2_ld_filepath, 'r') as r2_ld_file, open(r2_tsv_filepath, mode='w') as r2_tsv_file:
    #     # temp=r2_tsv_file.name
    #     for line in r2_ld_file:
    #         # Process the line
    #         line = '\t'.join(line.split())
    #         # Write the processed line to the output file, no need to specify end='' as '\n' is added explicitly
    #         r2_tsv_file.write(line + '\n')
    #     r2_tsv_file.flush()
    # Create a temporary file for the TSV output

    # TODO: test that this works
    r2_tsv_filepath = None
    with tempfile.NamedTemporaryFile(delete=False, suffix=".r2.tsv") as temp_file:
        r2_tsv_filepath = temp_file.name
        with open(r2_ld_filepath, 'r') as r2_ld_file, open(r2_tsv_filepath, mode='w') as r2_tsv_file:
            for line in r2_ld_file:
                # Process the line
                line = '\t'.join(line.split())
                # Write the processed line to the output file, no need to specify end='' as '\n' is added explicitly
                r2_tsv_file.write(line + '\n')
            r2_tsv_file.flush()
        
    with open(r2_tsv_filepath, 'r') as r2_tsv_file:
        plink_variants = pd.read_table(r2_tsv_file)
        logging.debug(f"Plink variants table:\n{plink_variants.shape}\n{plink_variants.head()}")

        # Get just the lead variants, which is provided by the user.
        lead_variants = variant_scores[['chr', 'pos', 'variant_id']].copy()
        lead_variants['r2'] = 1.0
        lead_variants['lead_variant'] = lead_variants['variant_id']
        logging.debug(f"Lead variants table:\n{lead_variants.head()}\n{lead_variants.shape}")

        # Get just the ld variants.
        plink_ld_variants = plink_variants[['SNP_A','CHR_B','BP_B','SNP_B','R2']].copy()
        plink_ld_variants.columns = ['lead_variant', 'chr', 'pos', 'variant_id', 'r2']
        plink_ld_variants = plink_ld_variants[['chr', 'pos', 'variant_id', 'r2', 'lead_variant']]
        plink_ld_variants['chr'] = 'chr' + plink_ld_variants['chr'].astype(str)
        plink_ld_variants = plink_ld_variants.sort_values(by=['variant_id', 'r2'], ascending=False).drop_duplicates(subset='variant_id')
        logging.debug(f"Plink LD variants table:\n{plink_ld_variants.shape}\n{plink_ld_variants.head()}")

        all_plink_variants = pd.concat([lead_variants, plink_ld_variants])
        all_plink_variants = all_plink_variants[['variant_id', 'r2', 'lead_variant']]
        all_plink_variants = all_plink_variants.sort_values( by=['variant_id', 'r2'], ascending=False)
        logging.debug(f"All plink variants table:\n{all_plink_variants.shape}\n{all_plink_variants.head()}")

        variant_scores = variant_scores.merge(all_plink_variants,
            on=['variant_id'],
            how='left')
    return variant_scores


def get_asb_adastra(chunk, sig_adastra_tf, sig_adastra_celltype):
    mean_asb_es_tf_ref = []
    mean_asb_es_tf_alt = []
    asb_tfs = []

    mean_asb_es_celltype_ref = []
    mean_asb_es_celltype_alt = []
    asb_celltypes = []

    for index,row in chunk.iterrows():

        local_tf_df = sig_adastra_tf.loc[sig_adastra_tf['variant_id'] == row['variant_id']].copy()
        if len(local_tf_df) > 0:
            mean_asb_es_tf_ref.append(local_tf_df['es_mean_ref'].mean())
            mean_asb_es_tf_alt.append(local_tf_df['es_mean_alt'].mean())
            asb_tfs.append(', '.join(local_tf_df['tf'].unique().tolist()))
        else:
            mean_asb_es_tf_ref.append(np.nan)
            mean_asb_es_tf_alt.append(np.nan)
            asb_tfs.append(np.nan)

        local_celltype_df = sig_adastra_celltype.loc[sig_adastra_celltype['variant_id'] == row['variant_id']].copy()
        if len(local_celltype_df) > 0:
            mean_asb_es_celltype_ref.append(local_celltype_df['es_mean_ref'].mean())
            mean_asb_es_celltype_alt.append(local_celltype_df['es_mean_alt'].mean())
            asb_celltypes.append(', '.join(local_celltype_df['celltype'].unique().tolist()))
        else:
            mean_asb_es_celltype_ref.append(np.nan)
            mean_asb_es_celltype_alt.append(np.nan)
            asb_celltypes.append(np.nan)
            
    chunk['adastra_asb_tfs'] = asb_tfs
    chunk['adastra_mean_asb_effect_size_tf_ref'] = mean_asb_es_tf_ref
    chunk['adastra_mean_asb_effect_size_tf_alt'] = mean_asb_es_tf_alt
    chunk['adastra_asb_celltypes'] = asb_celltypes
    chunk['adastra_mean_asb_effect_size_celltype_ref'] = mean_asb_es_celltype_ref
    chunk['adastra_mean_asb_effect_size_celltype_alt'] = mean_asb_es_celltype_alt
    
    return chunk


def add_adastra(variant_scores: pd.DataFrame, adastra_tf_file: str, adastra_celltype_file: str, threads=DEFAULT_ADASTRA_THREADS):

    sig_adastra_tf = pd.read_table(adastra_tf_file)
    sig_adastra_celltype = pd.read_table(adastra_celltype_file)

    # Modify both to have a variant_id column, since we don't retrieve their rsids. This takes some extra time, might be worth changing later.
    # variant_id should be <chr>:<pos>:<ref>:<alt>
    sig_adastra_tf['variant_id'] = sig_adastra_tf.apply(lambda x: f"{x['#chr']}:{x['pos']}:{x['ref']}:{x['alt']}", axis=1)
    sig_adastra_celltype['variant_id'] = sig_adastra_celltype.apply(lambda x: f"{x['#chr']}:{x['pos']}:{x['ref']}:{x['alt']}", axis=1)

    logging.debug(f"ADASTRA TF table:\n{sig_adastra_tf.shape}\n{sig_adastra_tf.head()}")
    logging.debug(f"ADASTRA celltype table:\n{sig_adastra_celltype.shape}\n{sig_adastra_celltype.head()}")

    n_threads = threads if threads else DEFAULT_ADASTRA_THREADS
    chunk_size = len(variant_scores) // n_threads
    chunks = np.array_split(variant_scores, len(variant_scores) // chunk_size)

    args_for_starmap = [(chunk, sig_adastra_tf, sig_adastra_celltype) for chunk in chunks]

    with Pool(processes=n_threads) as pool:
        results = pool.starmap(get_asb_adastra, args_for_starmap)

    new_variant_scores = pd.concat(results)

    pool.close()
    pool.join()

    logging.debug(f"ADASTRA annotations added to variant scores:\n{variant_scores.shape}\n{variant_scores.head()}")
    return new_variant_scores


def add_annot_using_pandas(variant_scores: pd.DataFrame, args):
    for label, expression in args:
        # Expose variant_scores as df for the user.
        variant_scores[label] = variant_scores.eval(expression)
    return variant_scores


def add_annot_using_python(variant_scores: pd.DataFrame, args):
    for label, expression in args:
        # Expose variant_scores as df for the user.
        variant_scores[label] = eval(expression, None, {'df': variant_scores})
    return variant_scores


def add_aggregate_annots_using_pandas(agg_annots: pd.DataFrame, cur_annots: pd.DataFrame, args):
    for label, expression, default_value_expression in args:
        # If the label is not in the aggregate DataFrame, add it with the default value.
        if label not in agg_annots.columns:
            # Check if it's a dynamic expression
            if isinstance(default_value_expression, str):
                # Evaluate the default value expression using df as agg_annots
                agg_annots[label] = agg_annots.eval(default_value_expression, local_dict={'df': agg_annots})
            else:
                # Directly assign if it's a static value
                agg_annots[label] = default_value_expression

        # Evaluate the expression using df as agg_annots and cur_df as cur_annots
        agg_annots[label] = agg_annots.eval(expression, local_dict={'df': agg_annots, 'cur_df': cur_annots})
    
    return agg_annots


def add_aggregate_annots_using_python(agg_annots: pd.DataFrame, cur_annots: pd.DataFrame, args):
    for label, expression, default_value_expression in args:
        # If the label is not in the aggregate DataFrame, add it with the default value.
        if label not in agg_annots.columns:
            # agg_annots[label] = default_value_expression
            agg_annots[label] = eval(default_value_expression, None, {'df': agg_annots})
        # Expose the aggregate DataFrame as df, and the DataFrame to be added as cur_df, for the user.
        agg_annots[label] = eval(expression, None, {'df': agg_annots, 'cur_df': cur_annots})
    return agg_annots


def parse_sort_together(group_expressions: List[str]):
    """
    Parse the --sort-together argument which can include column name, delimiter, sort order, and type.
    Format: col:delim:sort_order:type or col:delim:sort_order, or col:delim for default sort order and type.
    """
    parsed_groups = []
    
    for group in group_expressions:
        parsed_group = []
        columns = group.split('|')
        
        for col in columns:
            parts = col.split(':')
            if len(parts) not in [2, 3, 4]:
                raise ValueError(f"Invalid format in {col}. Expected format is col:delimiter:sort_order:type, or col:delimiter:sort_order, or col:delimiter for default sort order.")
            
            column_name = parts[0]
            delimiter = parts[1]
            sort_order = True  # Default is ascending order
            data_type = 'str'  # Default type is string

            if len(parts) >= 3:
                sort_order = parts[2].lower() == 'asc'
            if len(parts) == 4:
                data_type = parts[3].lower()  # Accept data type as the fourth part
            
            parsed_group.append({
                'column': column_name,
                'delimiter': delimiter,
                'sort_order': sort_order,
                'data_type': data_type  # Add data type to the parsed info
            })
        
        parsed_groups.append(parsed_group)
    
    return parsed_groups


def apply_multilevel_sort(df, sort_groups):
    """
    Sort the values within the columns of each row, based on the parsed sort_groups.
    The first column is sorted first, then the second column is used as a tie-breaker, and so on.
    """
    for group in sort_groups:
        # Extract column names, delimiters, sort orders, and data types
        columns = [col_info['column'] for col_info in group]
        delimiters = [col_info['delimiter'] for col_info in group]
        sort_orders = [col_info['sort_order'] for col_info in group]
        data_types = [col_info['data_type'] for col_info in group]

        # Split the columns into lists based on delimiters
        split_columns = [df[col].str.split(delimiter) for col, delimiter in zip(columns, delimiters)]
        
        # Convert each list to the appropriate data type
        for i, data_type in enumerate(data_types):
            if data_type == 'int':
                split_columns[i] = split_columns[i].apply(lambda lst: [int(x) for x in lst])
            elif data_type == 'float':
                split_columns[i] = split_columns[i].apply(lambda lst: [float(x) for x in lst])
            elif data_type == 'str':
                split_columns[i] = split_columns[i].apply(lambda lst: [str(x) for x in lst])

        # Combine the lists in each row into a tuple and sort based on the first column, second column on ties, etc.
        for index in range(df.shape[0]):
            # Zip the values together for this row
            zipped = list(zip(*[split_columns[i][index] for i in range(len(columns))]))

            # Sort based on the first column, then second, etc., respecting individual sort orders
            def custom_sort_key(x):
                key = []
                for i in range(len(columns)):
                    if isinstance(x[i], (int, float)):
                        # If descending, negate the number
                        key.append(-x[i] if not sort_orders[i] else x[i])
                    elif isinstance(x[i], str):
                        if sort_orders[i]:
                            # Ascending order for strings
                            key.append(x[i])
                        else:
                            # Descending order for strings: reverse lexicographical using ord()
                            key.append(tuple(-ord(c) for c in x[i]))
                return tuple(key)

            sorted_zipped = sorted(zipped, key=custom_sort_key)

            # Unzip the sorted result
            sorted_lists = list(zip(*sorted_zipped))

            # Reassign sorted values to the DataFrame columns
            for i, col in enumerate(columns):
                if sorted_lists:
                    df.at[index, col] = delimiters[i].join(map(str, sorted_lists[i]))

    return df
