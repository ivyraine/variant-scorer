import pandas as pd
import numpy as np
import os
import subprocess
import logging

from utils.argmanager import *
from utils.helpers import *
pd.set_option('display.max_columns', 20)

def main(args = None):

    metadata = None
    # Get input and output paths.
    if args.input_metadata:
        metadata = pd.read_csv(args.input_metadata, sep='\t')

    if args.subcommand == 'annotate-summ':
        if args.input_metadata:
            if args.subcommand == 'annotate-summ' and not set([SUMMARIZE_OUT_PATH_COL, ANNOTATE_SUMM_OUT_PATH_COL]).issubset(metadata.columns):
                raise ValueError(f"Metadata file {args.input_metadata} must contain columns: {PEAKS_PATH_COL}, {SUMMARIZE_OUT_PATH_COL}, {ANNOTATE_SUMM_OUT_PATH_COL}")
            if args.peaks and args.peaks != True:
                raise ValueError(f"The --peaks flag must not take an argument when provided with --input-metadata. Remove the argument and try again.")
            elif args.peaks and not set ([PEAKS_PATH_COL]).issubset(metadata.columns):
                raise ValueError(f"Metadata file {args.input_metadata} must contain column: {PEAKS_PATH_COL}")
        else:
            metadata = pd.DataFrame({PEAKS_PATH_COL: [args.peaks], SUMMARIZE_OUT_PATH_COL: [args.summarize_output_path], ANNOTATE_SUMM_OUT_PATH_COL: [args.annotate_summ_output_path]})
        # Check if the peaks path column is provided.
        input_col = SUMMARIZE_OUT_PATH_COL
        output_col = ANNOTATE_SUMM_OUT_PATH_COL
    elif args.subcommand == 'annotate-aggr':
        if args.input_metadata:
            # Check for peaks column if peaks are to be used.
            if args.subcommand == 'annotate-aggr' and not set([AGGREGATE_OUT_PATH_COL, ANNOTATE_AGGR_OUT_PATH_COL]).issubset(metadata.columns):
                raise ValueError(f"Metadata file {args.input_metadata} must contain columns: {PEAKS_PATH_COL}, {AGGREGATE_OUT_PATH_COL}, {ANNOTATE_AGGR_OUT_PATH_COL}")
        else:
            metadata = pd.DataFrame({PEAKS_PATH_COL: [args.peaks], AGGREGATE_OUT_PATH_COL: [args.aggregate_output_path], ANNOTATE_AGGR_OUT_PATH_COL: [args.annotate_aggr_output_path]})
        input_col = AGGREGATE_OUT_PATH_COL
        output_col = ANNOTATE_AGGR_OUT_PATH_COL
    else:
        raise ValueError(f"Invalid subcommand {args.subcommand}")
    # Drop duplicates based on input_col and output_col, keeping other columns intact
    metadata = metadata.drop_duplicates(subset=[input_col, output_col])


    # Process input & output path pairs.
    for index, row in metadata.iterrows():

        variant_scores = pd.read_table(row[input_col])

        variant_scores_bed_format = None
        if args.schema == "bed":
            if variant_scores['pos'].equals(variant_scores['end']):
                variant_scores['pos'] = variant_scores['pos'] - 1
            variant_scores_bed_format = variant_scores[['chr','pos','end','allele1','allele2','variant_id']].copy()
            variant_scores_bed_format.sort_values(by=["chr","pos","end"], inplace=True)
        else:
            variant_scores_bed_format = variant_scores[['chr','pos','allele1','allele2','variant_id']].copy()
            variant_scores_bed_format['pos']  = variant_scores_bed_format.apply(lambda x: int(x.pos)-1, axis = 1)
            variant_scores_bed_format['end']  = variant_scores_bed_format.apply(lambda x: int(x.pos)+len(x.allele1), axis = 1)
            variant_scores_bed_format = variant_scores_bed_format[['chr','pos','end','allele1','allele2','variant_id']]
            variant_scores_bed_format.sort_values(by=["chr","pos","end"], inplace=True)

        if args.join_tsvs:
            for tsv, label, direction in reversed(args.join_args):
                join_df = pd.read_csv(tsv, sep='\t')
                variant_scores = variant_scores.merge(join_df, how=direction, on=label)

        if args.subcommand == 'annotate-summ' and args.peaks:
            logging.info("Annotating with peak overlap")

            # Get the peaks file path from the metadata file if provided.
            peaks_file = row[PEAKS_PATH_COL] if args.peaks is True else args.peaks

            peak_df = pd.read_table(peaks_file, header=None)
            variant_bed = pybedtools.BedTool.from_dataframe(variant_scores_bed_format)
            peak_bed = pybedtools.BedTool.from_dataframe(peak_df)
            peak_intersect_bed = variant_bed.intersect(peak_bed, wa=True, u=True)

            peak_intersect_df = peak_intersect_bed.to_dataframe(names=variant_scores_bed_format.columns.tolist())

            logging.debug(f"Peak overlap table:\n{peak_intersect_df.shape}\n{peak_intersect_df.head()}")

            variant_scores['peak_overlap'] = variant_scores['variant_id'].isin(peak_intersect_df['variant_id'].tolist())

        if args.add_n_closest_elements:
            variant_scores = add_n_closest_elements(variant_scores, args.closest_n_elements_args, variant_scores_bed_format)

        if args.add_closest_elements_in_window:
            variant_scores = add_closest_elements_in_window(variant_scores, args.closest_elements_in_window_args, variant_scores_bed_format)
        
        if args.r2:
            variant_scores = add_r2(variant_scores, args.r2)
                
        if args.add_adastra:
            variant_scores = add_adastra(variant_scores, args.adastra_tf_file, args.adastra_celltype_file, args.threads)
        
        if args.add_annot_using_pandas:
            variant_scores = add_annot_using_pandas(variant_scores, args.add_annot_using_pandas)

        if args.add_annot_using_python:
            variant_scores = add_annot_using_python(variant_scores, args.add_annot_using_python)

        logging.info(f"Final annotation table:\n{variant_scores.shape}\n{variant_scores.head()}")

        output_path = row[output_col]
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        variant_scores.to_csv(output_path,\
                            sep="\t",\
                            index=False)

        logging.info(f"Annotation written to: {output_path}")


if __name__ == "__main__":
    main()
