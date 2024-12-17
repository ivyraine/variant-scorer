import polars as pl
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
        metadata = pl.read_csv(args.input_metadata, separator='\t')
        aggregate_out_paths = metadata[AGGREGATE_OUT_PATH_COL].unique()
    elif args.annotate_summ_output_paths and args.aggregate_output_path:
        aggregate_out_paths = [aggregate_out_paths]
    else:
        raise ValueError(f"`aggregate` requires either: --input-metadata to be provided with the {AGGREGATE_OUT_PATH_COL} and {MODEL_ID_COL} columns, or --annotate-output-paths to be provided alongside --aggregate-output-path to be provided.")

            # Check for missing input files
    missing_input_files = set()
    for row in metadata.iter_rows(named=True):
        if not os.path.isfile(row[ANNOTATE_SUMM_OUT_PATH_COL]):
            missing_input_files.add(row[ANNOTATE_SUMM_OUT_PATH_COL])
    
    if missing_input_files:
        missing_files_str = "\n".join(missing_input_files)
        if args.invalid_file_log:
            with open(args.invalid_file_log, 'w') as f:
                f.write(f'{missing_files_str}\n')
        if args.skip_invalid_inputs:
            logging.warning(f'The following {len(missing_input_files)} TSV file(s) are missing:\n{missing_files_str}')
        else:
            print(args.skip_invalid_inputs)
            raise FileNotFoundError(f'The following {len(missing_input_files)} TSV file(s) are missing:\n{missing_files_str}. Please fix the missing or incorrect files and try again, or use the --skip-invalid-inputs flag.')

    for aggregate_out_path in aggregate_out_paths:
        # Get all rows with the same aggregate_out_path
        if args.input_metadata:
            cur_metadata = metadata.filter(metadata[AGGREGATE_OUT_PATH_COL] == aggregate_out_path)
        elif args.annotate_summ_output_paths and args.aggregate_output_path:
            cur_metadata = pl.DataFrame({ANNOTATE_SUMM_OUT_PATH_COL: args.annotate_summ_output_paths, MODEL_ID_COL: args.annotate_summ_output_paths, AGGREGATE_OUT_PATH_COL: [args.aggregate_output_path] * len(args.annotate_summ_output_paths)})
        
        # Other rows.
        agg_df = None
        is_first_row = True
        for row in cur_metadata.iter_rows(named=True):  
            print(row)
            cur_path = row[ANNOTATE_SUMM_OUT_PATH_COL]

            if cur_path in missing_input_files:
                continue

            cur_df = pl.read_csv(cur_path, separator='\t')

            if args.expected_row_count and cur_df.height != args.expected_row_count:
                if args.invalid_file_log:
                    with open(args.invalid_file_log, 'a') as f:
                        f.write(f'{cur_path}\n')
                if args.skip_invalid_inputs:
                    logging.warning(f'The following TSV file has an unexpected row count: {cur_path}. Skipping this file.')
                    continue
                else:
                    raise ValueError(f'The following TSV file has an unexpected row count: {cur_path}. Please fix the missing or incorrect files and try again, or use the --skip-invalid-inputs flag.')

            if args.add_temp_model_id:
                # cur_df[MODEL_ID_COL] = row[MODEL_ID_COL]
                cur_df = cur_df.with_columns(
                    pl.lit(row[MODEL_ID_COL]).alias(MODEL_ID_COL)
                )

            # First row.
            if is_first_row:
                if args.add_aggregate_annot_with_python:
                    agg_df = pl.from_pandas(add_aggregate_annots_using_python(args.add_aggregate_annot_with_python, cur_df.to_pandas()))
                is_first_row = False
                continue
            else:
                # add_aggregate_annots_using_pandas(agg_df, cur_df, args.add_aggregate_annot_with_pandas)
                if args.add_aggregate_annot_with_python:
                    agg_df = pl.from_pandas(add_aggregate_annots_using_python(args.add_aggregate_annot_with_python, agg_df.to_pandas(), cur_df.to_pandas()))
            
        if is_first_row:
            raise ValueError(f"No valid input files found for aggregate_out_path: {aggregate_out_path}")
    
        if args.add_temp_model_id:
            agg_df = agg_df.drop([MODEL_ID_COL])
        
        # Sort columns together.
        if args.sort_together:
            sort_groups = parse_sort_together(args.sort_together)
            agg_df = apply_multilevel_sort(agg_df, sort_groups)

        columns_to_drop = [
            'logfc.mean', 'logfc.mean.pval', 'abs_logfc.mean',
            'abs_logfc.mean.pval', 'jsd.mean', 'jsd.mean.pval',
            'logfc_x_jsd.mean', 'logfc_x_jsd.mean.pval',
            'abs_logfc_x_jsd.mean', 'abs_logfc_x_jsd.mean.pval',
            'active_allele_quantile.mean', 'active_allele_quantile.mean.pval',
            'logfc_x_active_allele_quantile.mean', 'logfc_x_active_allele_quantile.mean.pval',
            'abs_logfc_x_active_allele_quantile.mean', 'abs_logfc_x_active_allele_quantile.mean.pval',
            'jsd_x_active_allele_quantile.mean', 'jsd_x_active_allele_quantile.mean.pval',
            'logfc_x_jsd_x_active_allele_quantile.mean', 'logfc_x_jsd_x_active_allele_quantile.mean.pval',
            'abs_logfc_x_jsd_x_active_allele_quantile.mean', 'abs_logfc_x_jsd_x_active_allele_quantile.mean.pval'
        ]
        existing_columns = [col for col in columns_to_drop if col in agg_df.columns]
        agg_df = agg_df.drop(existing_columns)

        os.makedirs(os.path.dirname(aggregate_out_path), exist_ok=True)
        agg_df.write_csv(aggregate_out_path, separator='\t')
        logging.info(f"Saved aggregated annotations to {aggregate_out_path}")

