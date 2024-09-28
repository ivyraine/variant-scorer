import pandas as pd
import numpy as np
import os
import logging
from collections import namedtuple

from utils.helpers import *

def main(args):
    print("in aggr")
    if not os.path.exists(args.input_metadata):
        raise FileNotFoundError(f"Metadata file {args.input_metadata} not found.")

    if args.input_metadata and args.annotate_summ_output_paths and args.aggregate_output_path:
        raise ValueError(f"`aggregate` requires either: --input-metadata to be provided with the {AGGREGATE_OUT_PATH_COL} and {MODEL_ID_COL} columns, or --annotate-output-paths to be provided alongside --aggregate-output-path to be provided. Remove one of either option and try again.")
    elif args.input_metadata:
        metadata = pd.read_csv(args.input_metadata, sep='\t')
        aggregate_out_paths = metadata[AGGREGATE_OUT_PATH_COL].unique()
    elif args.annotate_summ_output_paths and args.aggregate_output_path:
        aggregate_out_paths = [aggregate_out_paths]
    else:
        raise ValueError(f"`aggregate` requires either: --input-metadata to be provided with the {AGGREGATE_OUT_PATH_COL} and {MODEL_ID_COL} columns, or --annotate-output-paths to be provided alongside --aggregate-output-path to be provided.")

    for aggregate_out_path in aggregate_out_paths:
        # Get all rows with the same aggregate_out_path
        if args.input_metadata:
            cur_metadata = metadata[metadata[AGGREGATE_OUT_PATH_COL] == aggregate_out_path]
        elif args.annotate_summ_output_paths and args.aggregate_output_path:
            cur_metadata = pd.DataFrame({ANNOTATE_SUMM_OUT_PATH_COL: args.annotate_summ_output_paths, MODEL_ID_COL: args.annotate_summ_output_paths, AGGREGATE_OUT_PATH_COL: [args.aggregate_output_path] * len(args.annotate_summ_output_paths)})
        
        # Check that all the annotation files exist
        for cur_path in cur_metadata[ANNOTATE_SUMM_OUT_PATH_COL]:
            if not os.path.exists(cur_path):
                raise FileNotFoundError(f"Annotation file {cur_path} not found.")
        
        agg_df = pd.read_csv(cur_metadata.iloc[0][ANNOTATE_SUMM_OUT_PATH_COL], sep='\t')
        if args.add_temp_model_id:
            agg_df[MODEL_ID_COL] = cur_metadata.iloc[0][MODEL_ID_COL]

        # for cur_path in cur_metadata[ANNOTATE_SUMM_OUT_PATH_COL].values[1:]:
        for _, row in cur_metadata.iloc[1:].iterrows():  
            cur_path = row[ANNOTATE_SUMM_OUT_PATH_COL]
            cur_df = pd.read_csv(cur_path, sep='\t')
            if args.add_temp_model_id:
                cur_df[MODEL_ID_COL] = row[MODEL_ID_COL]
            # add_aggregate_annots_using_pandas(agg_df, cur_df, args.add_aggregate_annot_with_pandas)
            add_aggregate_annots_using_python(agg_df, cur_df, args.add_aggregate_annot_with_python)
    
        if args.add_temp_model_id:
            agg_df = agg_df.drop(columns=[MODEL_ID_COL])
        
        # Sort columns together.
        if args.sort_together:
            sort_groups = parse_sort_together(args.sort_together)
            agg_df = apply_multilevel_sort(agg_df, sort_groups)

        # Drop model-specific columns.
        agg_df.drop(columns=['logfc.mean', 'logfc.mean.pval', 'abs_logfc.mean',
                             'abs_logfc.mean.pval', 'jsd.mean', 'jsd.mean.pval',
                             'logfc_x_jsd.mean', 'logfc_x_jsd.mean.pval',
                             'abs_logfc_x_jsd.mean', 'abs_logfc_x_jsd.mean.pval',
                             'active_allele_quantile.mean', 'active_allele_quantile.mean.pval',
                             'logfc_x_active_allele_quantile.mean', 'logfc_x_active_allele_quantile.mean.pval',
                             'abs_logfc_x_active_allele_quantile.mean', 'abs_logfc_x_active_allele_quantile.mean.pval',
                             'jsd_x_active_allele_quantile.mean', 'jsd_x_active_allele_quantile.mean.pval',
                             'logfc_x_jsd_x_active_allele_quantile.mean', 'logfc_x_jsd_x_active_allele_quantile.mean.pval',
                             'abs_logfc_x_jsd_x_active_allele_quantile.mean', 'abs_logfc_x_jsd_x_active_allele_quantile.mean.pval'],
                    inplace=True, errors='ignore')
                            

        os.makedirs(os.path.dirname(aggregate_out_path), exist_ok=True)
        agg_df.to_csv(aggregate_out_path, sep='\t', index=False)
        logging.info(f"Saved aggregated annotations to {aggregate_out_path}")

