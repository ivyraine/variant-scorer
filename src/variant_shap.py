from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.models import load_model
import tensorflow as tf
import scipy.stats
from scipy.spatial.distance import jensenshannon
import pandas as pd
import os
import argparse
import numpy as np
import h5py
import math
from generators.variant_generator import VariantGenerator
from generators.peak_generator import PeakGenerator
from utils import losses
from utils.helpers import *
import shap
import logging
from utils.shap_utils import *
import deepdish as dd
tf.compat.v1.disable_v2_behavior()


def main(args = None):
    if args is None:
        args = argmanager.fetch_shap_args()
        logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')

    if not os.path.isdir(args.shap_dir):
        raise OSError(f"Output directory ({args.shap_dir}) does not exist")

    shap_output_prefixes = [ get_shap_output_file_prefix(args.shap_dir, args.model_name, index) for index in range(len(args.models)) ]
    is_using_file_overrides = False
    if args.shap_filenames is not None:
        if len(args.shap_filenames) != len(args.models):
            raise ValueError("Number of models and shap filenames do not match, which they are requird to do if the --shap-filenames flag is given. Exiting.")
        else:
            shap_output_prefixes = [ os.path.join(args.shap_dir, args.shap_filenames[i]) for i in range(len(args.models)) ]
            is_using_file_overrides = True

    model_and_output_prefixes = list(zip(args.models, shap_output_prefixes))

    for model_index, (model_name, shap_output_prefix) in enumerate(model_and_output_prefixes):
        model = load_model_wrapper(model_name)
        filtered_annotations_file = get_filter_output_file(args.filter_dir, args.model_name)
        # variants_table = load_variant_table(filtered_annotations_file, args.schema)
        variants_table = load_variant_table(filtered_annotations_file, None)
        variants_table = variants_table.fillna('-')

        chrom_sizes = pd.read_csv(args.chrom_sizes, header=None, sep='\t', names=['chrom', 'size'])
        chrom_sizes_dict = chrom_sizes.set_index('chrom')['size'].to_dict()
        logging.debug(f"Loaded chromosome sizes from {args.chrom_sizes}:\n{chrom_sizes.head()}\n{chrom_sizes.shape}")

        if args.debug_mode:
            variants_table = variants_table.sample(10)
            print(variants_table.head())
        
        logging.debug(f"Loaded model {model_name} ({str(model_index+1)}/{str(len(model_and_output_prefixes))})")
        logging.debug(f"Loaded variants table from {filtered_annotations_file}:\n{variants_table.head()}\n{variants_table.shape}")

        # infer input length
        if args.lite:
            input_len = model.input_shape[0][1]
        else:
            input_len = model.input_shape[1]
        logging.debug(f"Input length inferred from the model: {input_len}")

        variants_table = variants_table.loc[variants_table.apply(lambda x: get_valid_variants(x.chr, x.pos, x.allele1, x.allele2, input_len, chrom_sizes_dict), axis=1)]
        logging.debug(f"Filtered variants table to only include valid variants:\n{variants_table.head()}\n{variants_table.shape}")
        variants_table.reset_index(drop=True, inplace=True)
        
        for shap_type_index, shap_type in enumerate(args.shap_type):
            # fetch model prediction for variants
            batch_size=args.batch_size
            ### set the batch size to the length of variant table in case variant table is small to avoid error
            batch_size=min(batch_size,len(variants_table))
            # output_file=h5py.File(''.join([args.shap_dir, ".variant_shap.%s.h5"%shap_type]), 'w')
            # observed = output_file.create_group('observed')
            # allele1_write = observed.create_dataset('allele1_shap', (len(variants_table),2114,4), chunks=(batch_size,2114,4), dtype=np.float16, compression='gzip', compression_opts=9)
            # allele2_write = observed.create_dataset('allele2_shap', (len(variants_table),2114,4), chunks=(batch_size,2114,4), dtype=np.float16, compression='gzip', compression_opts=9)
            # variant_ids_write = observed.create_dataset('variant_ids', (len(variants_table),), chunks=(batch_size,), dtype='S100', compression='gzip', compression_opts=9)

            allele1_seqs = []
            allele2_seqs = []
            allele1_scores = []
            allele2_scores = []
            variant_ids = []

            print(f"{len(variants_table)}/{batch_size}")

            num_batches=len(variants_table)//batch_size
            for i in range(num_batches):
                sub_table=variants_table[i*batch_size:(i+1)*batch_size]
                var_ids, allele1_inputs, allele2_inputs, \
                allele1_shap, allele2_shap = fetch_shap(model,
                                                        sub_table,
                                                        input_len,
                                                        args.genome,
                                                        args.batch_size,
                                                        debug_mode=args.debug_mode,
                                                        lite=args.lite,
                                                        bias=None,
                                                        shuf=False,
                                                        shap_type=shap_type)
                # TODO: add fetch_variant_predictions here. You'll need to create new .prioritized files, which the viz step will take if the no_hdf5 flag is provided.
                
                # allele1_write[i*batch_size:(i+1)*batch_size] = allele1_shap
                # allele2_write[i*batch_size:(i+1)*batch_size] = allele2_shap
                # variant_ids_write[i*batch_size:(i+1)*batch_size] = [s.encode("utf-8") for s in var_ids]

                if len(variant_ids) == 0:
                    allele1_seqs = allele1_inputs
                    allele2_seqs = allele2_inputs
                    allele1_scores = allele1_shap
                    allele2_scores = allele2_shap
                    variant_ids = var_ids
                else:
                    allele1_seqs = np.concatenate((allele1_seqs, allele1_inputs))
                    allele2_seqs = np.concatenate((allele2_seqs, allele2_inputs))
                    allele1_scores = np.concatenate((allele1_scores, allele1_shap))
                    allele2_scores = np.concatenate((allele2_scores, allele2_shap))
                    variant_ids = np.concatenate((variant_ids, var_ids))

            if len(variants_table)%batch_size != 0:
                sub_table=variants_table[num_batches*batch_size:len(variants_table)]
                var_ids, allele1_inputs, allele2_inputs, \
                allele1_shap, allele2_shap = fetch_shap(model,
                                                                        sub_table,
                                                                        input_len,
                                                                        args.genome,
                                                                        args.batch_size,
                                                                        debug_mode=args.debug_mode,
                                                                        lite=args.lite,
                                                                        bias=None,
                                                                        shuf=False,
                                                                        shap_type=shap_type)
                
                # allele1_write[num_batches*batch_size:len(variants_table)] = allele1_shap
                # allele2_write[num_batches*batch_size:len(variants_table)] = allele2_shap
                # variant_ids_write[num_batches*batch_size:len(variants_table)] = [s.encode("utf-8") for s in var_ids]

                if len(variant_ids) == 0:
                    allele1_seqs = allele1_inputs
                    allele2_seqs = allele2_inputs
                    allele1_scores = allele1_shap
                    allele2_scores = allele2_shap
                    variant_ids = var_ids
                else:
                    allele1_seqs = np.concatenate((allele1_seqs, allele1_inputs))
                    allele2_seqs = np.concatenate((allele2_seqs, allele2_inputs))
                    allele1_scores = np.concatenate((allele1_scores, allele1_shap))
                    allele2_scores = np.concatenate((allele2_scores, allele2_shap))
                    variant_ids = np.concatenate((variant_ids, var_ids))

            # # store shap at variants
            # with h5py.File(''.join([args.shap_dir, ".variant_shap.%s.h5"%shap_type]), 'w') as f:
            #     observed = f.create_group('observed')
            #     observed.create_dataset('allele1_shap', data=allele1_shap, compression='gzip', compression_opts=9)
            #     observed.create_dataset('allele2_shap', data=allele2_shap, compression='gzip', compression_opts=9)
                
            assert(allele1_seqs.shape==allele1_scores.shape)
            assert(allele2_seqs.shape==allele2_scores.shape)
            assert(allele1_seqs.shape==allele2_seqs.shape)
            assert(allele1_scores.shape==allele2_scores.shape)
            assert(allele1_seqs.shape[2]==4)
            assert(len(allele1_seqs==len(variant_ids)))
            
            output_h5 = f"{shap_output_prefix}.variant_shap.{shap_type}.h5"
            if is_using_file_overrides:
                output_h5 = f"{shap_output_prefix}.h5"

            variant_ids_utf8 = np.array([vid.encode('utf-8') for vid in variant_ids])
            shap_dict = {
                'allele1': {
                    'raw': { 
                        'seq': np.transpose(allele1_seqs, (0, 2, 1)).astype(np.int8)
                    },
                    'shap': {
                        'seq': np.transpose(allele1_scores, (0, 2, 1)).astype(np.float16)
                    },
                    'projected_shap': {
                        'seq': np.transpose(allele1_seqs * allele1_scores, (0, 2, 1)).astype(np.float16)
                    },
                },
                'allele2': {
                    'raw': {
                        'seq': np.transpose(allele2_seqs, (0, 2, 1)).astype(np.int8)
                    },
                    'shap': {
                        'seq': np.transpose(allele2_scores, (0, 2, 1)).astype(np.float16)
                    },
                    'projected_shap': {
                        'seq': np.transpose(allele2_seqs * allele2_scores, (0, 2, 1)).astype(np.float16)
                    },
                },
                'variant_ids': variant_ids_utf8,
            }

            # with h5py.File(output_h5, 'w') as h5f:
            #     for key, value in shap_dict.items():
            #         if isinstance(value, dict):  # For nested dictionaries, create a group first
            #             group = h5f.create_group(key)
            #             for subkey, subvalue in value.items():
            #                 # Check if subvalue is a numpy array of strings and encode as UTF-8 bytes if so
            #                 if isinstance(subvalue, np.ndarray) and subvalue.dtype.kind in {'U', 'S'}:
            #                     subvalue = subvalue.astype(h5py.special_dtype(vlen=str))
            #                 group.create_dataset(subkey, data=subvalue, compression='gzip', compression_opts=9)
            #         else:
            #             # Check if value is a numpy array of strings and encode as UTF-8 bytes if so
            #             if isinstance(value, np.ndarray) and value.dtype.kind in {'U', 'S'}:
            #                 value = value.astype(h5py.special_dtype(vlen=str))
            #             h5f.create_dataset(key, data=value, compression='gzip', compression_opts=9)

            dd.io.save(output_h5, shap_dict, compression='blosc')

            logging.info(f"({(model_index*2)+shap_type_index+1}/{len(model_and_output_prefixes)*len(args.shap_type)}) Finished SHAP for {model_name} for shap type {shap_type} and saved to {output_h5}")
    
    logging.info("Finished SHAP calculations")

if __name__ == "__main__":
    main()
