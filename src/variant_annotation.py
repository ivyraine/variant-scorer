import polars as pl
import numpy as np
import os
import subprocess
import logging
from math import ceil
from multiprocessing import cpu_count, get_context

from utils.helpers import *

def annotate(metadata_split, args, partition_index, process_index, input_col, output_col):

    setup_logging(logging.getLogger(), args.verbose, prefix=f"[Partition {partition_index} Process {process_index}]: ")

    logging.debug(f"Beginning process.")

    if metadata_split.is_empty():
        logging.warning(f"Empty metadata split. Exiting.")
        return

    if args.invalid_file_log and  os.path.isfile(args.invalid_file_log):
        os.remove(args.invalid_file_log)

    # Check for missing input files
    missing_input_files = set()
    for row in metadata_split.iter_rows(named=True):
        if not os.path.isfile(row[input_col]):
            missing_input_files.add(row[input_col])
    
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
    
    logging.info(f"Processing {metadata_split.height} files.")

    for index, row in enumerate(metadata_split.iter_rows(named=True)):

        logging.info(f"Processing {row[input_col]} ({index+1}/{len(metadata_split)})")

        if row[input_col] in missing_input_files:
            logging.warning(f"Skipping missing file: {row[input_col]}")
            continue

        variant_scores = pl.read_csv(row[input_col], separator='\t')

        if args.expected_row_count:
            if args.expected_row_count != variant_scores.height:
                if args.invalid_file_log:
                    with open(args.invalid_file_log, 'a') as f:
                        f.write(f'{row[input_col]}\n')
                if args.skip_invalid_inputs:
                    logging.warning(f"Skipping invalid file: {row[input_col]}. Expected line count: {args.expected_row_count}, Actual line count: {len(variant_scores)}")
                    continue
                else:
                    raise ValueError(f"Invalid file: {row[input_col]}. Expected line count: {args.expected_row_count}, Actual line count: {len(variant_scores)}")

        variant_scores_bed_format = None
        # if args.schema == "bed":
        #     if variant_scores['pos'].equals(variant_scores['end']):
        #         variant_scores['pos'] = variant_scores['pos'] - 1
        #     variant_scores_bed_format = variant_scores[['chr','pos','end','allele1','allele2','variant_id']].copy()
        #     variant_scores_bed_format.sort_values(by=["chr","pos","end"], inplace=True)
        # else:
        #     variant_scores_bed_format = variant_scores[['chr','pos','allele1','allele2','variant_id']].copy()
        #     variant_scores_bed_format['pos']  = variant_scores_bed_format.apply(lambda x: int(x.pos)-1, axis = 1)
        #     variant_scores_bed_format['end']  = variant_scores_bed_format.apply(lambda x: int(x.pos)+len(x.allele1), axis = 1)
        #     variant_scores_bed_format = variant_scores_bed_format[['chr','pos','end','allele1','allele2','variant_id']]
        #     variant_scores_bed_format.sort_values(by=["chr","pos","end"], inplace=True)
        if args.schema == "bed":
            if (variant_scores["pos"] == variant_scores["end"]).all():
                variant_scores = variant_scores.with_columns(
                    (pl.col("pos") - 1).alias("pos")
                )
            variant_scores_bed_format = variant_scores.select(["chr", "pos", "end", "allele1", "allele2", "variant_id"])
            variant_scores_bed_format = variant_scores_bed_format.sort(["chr", "pos", "end"])
        else:
            variant_scores_bed_format = variant_scores.select(["chr", "pos", "allele1", "allele2", "variant_id"])
            variant_scores_bed_format = variant_scores_bed_format.with_columns([
                (pl.col("pos") - 1).alias("pos"),
                (pl.col("pos") + pl.col("allele1").str.len_chars()).alias("end")
            ])
            variant_scores_bed_format = variant_scores_bed_format.select(["chr", "pos", "end", "allele1", "allele2", "variant_id"])
            variant_scores_bed_format = variant_scores_bed_format.sort(["chr", "pos", "end"])


        if args.join_tsvs:
            for tsv, label, direction in reversed(args.join_args):
                join_df = pl.read_csv(tsv, separator='\t')
                variant_scores = variant_scores.join(join_df, on=label, how=direction)

        if args.subcommand == 'annotate-summ' and args.peaks:
            logging.info("Annotating with peak overlap")

            # Get the peaks file path from the metadata_split file if provided.
            peaks_file = row[PEAKS_PATH_COL] if args.peaks is True else args.peaks

            peak_df = pl.read_csv(peaks_file, has_header=False, separator='\t', null_values=['.'])
            variant_bed = pybedtools.BedTool.from_dataframe(variant_scores_bed_format.to_pandas())
            peak_bed = pybedtools.BedTool.from_dataframe(
                    peak_df.to_pandas()
                    )
            peak_intersect_bed = variant_bed.intersect(peak_bed, wa=True, u=True)

            peak_intersect_df = pl.from_pandas(
                peak_intersect_bed.to_dataframe(names=variant_scores_bed_format.to_pandas().columns.tolist())
            )
            # Check for empty peak intersect dataframe.
            if peak_intersect_df.is_empty():
                peak_intersect_df = pl.DataFrame({col: [] for col in variant_scores_bed_format.columns})

            logging.debug(f"Peak overlap table:\n{peak_intersect_df.shape}\n{peak_intersect_df.head()}")

            # variant_scores['peak_overlap'] = variant_scores['variant_id'].isin(peak_intersect_df['variant_id'].tolist())
            variant_scores = variant_scores.with_columns(
                variant_scores["variant_id"].is_in(peak_intersect_df["variant_id"]).alias("peak_overlap")
            )


        variant_scores = variant_scores.to_pandas()
        if args.add_n_closest_elements:
            variant_scores = add_n_closest_elements(variant_scores, args.closest_n_elements_args, variant_scores_bed_format)

        if args.add_closest_elements_in_window:
            variant_scores = add_closest_elements_in_window(variant_scores, args.closest_elements_in_window_args, variant_scores_bed_format)
        
        if args.r2:
            variant_scores = add_r2(variant_scores, args.r2)
                
        if args.add_adastra:
            variant_scores = add_adastra(variant_scores, args.adastra_tf_file, args.adastra_celltype_file, 1)
        
        if args.add_annot_using_pandas:
            variant_scores = add_annot_using_pandas(variant_scores, args.add_annot_using_pandas)

        if args.add_annot_using_python:
            variant_scores = add_annot_using_python(variant_scores, args.add_annot_using_python)
        variant_scores = pl.from_pandas(variant_scores)

        logging.info(f"Final annotation table:\n{variant_scores.shape}\n{variant_scores.head()}")

        output_path = row[output_col]
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # variant_scores.to_csv(output_path,\
        #                     separator="\t",\
        #                     index=False)
        variant_scores.write_csv(output_path, separator="\t")

        logging.info(f"Annotation written to: {output_path}")
    logging.debug(f"Done.")


def main(args = None):
    metadata = None
    # Get input and output paths.
    if args.input_metadata:
        metadata = pl.read_csv(args.input_metadata, separator='\t')

    if args.subcommand == 'annotate-summ':
        input_col = SUMMARIZE_OUT_PATH_COL
        output_col = ANNOTATE_SUMM_OUT_PATH_COL
        if args.input_metadata:
            if args.subcommand == 'annotate-summ' and not set([input_col, output_col]).issubset(metadata.columns):
                raise ValueError(f"Metadata file {args.input_metadata} must contain columns: {PEAKS_PATH_COL}, {input_col}, {output_col}")
            if args.peaks and args.peaks != True:
                raise ValueError(f"The --peaks flag must not take an argument when provided with --input-metadata. Remove the argument and try again.")
            elif args.peaks and not set ([PEAKS_PATH_COL]).issubset(metadata.columns):
                raise ValueError(f"Metadata file {args.input_metadata} must contain column: {PEAKS_PATH_COL}")
        else:
            metadata = pl.DataFrame({
                PEAKS_PATH_COL: [args.peaks],
                input_col: [args.summarize_output_path],
                output_col: [args.annotate_summ_output_path]
            })
        # Check if the peaks path column is provided.
    elif args.subcommand == 'annotate-aggr':
        input_col = AGGREGATE_OUT_PATH_COL
        output_col = ANNOTATE_AGGR_OUT_PATH_COL
        if args.input_metadata:
            # Check for peaks column if peaks are to be used.
            if args.subcommand == 'annotate-aggr' and not set([input_col, output_col]).issubset(metadata.columns):
                raise ValueError(f"Metadata file {args.input_metadata} must contain columns: {PEAKS_PATH_COL}, {input_col}, {output_col}")
        else:
            metadata = pl.DataFrame({
                PEAKS_PATH_COL: [args.peaks],
                input_col: [args.aggregate_output_path],
                output_col: [args.annotate_aggr_output_path]
            })
    else:
        raise ValueError(f"Invalid subcommand {args.subcommand}")

    # Drop duplicates based on input_col and output_col, keeping other columns intact
    metadata = metadata.unique(subset=[input_col, output_col], maintain_order=True)
    
    # Check if there's duplicates in the output_col
    duplicates = metadata.filter(metadata[output_col].is_duplicated())
    if not duplicates.is_empty():
        raise ValueError(f"The following rows have duplicate output cols ({output_col}):\n{duplicates}\nPlease remove them and try again.")

    def get_n_metadata_partitions_evenly(metadata, n):
        partitions = [pl.DataFrame() for _ in range(n)]

        if input_col not in metadata.columns:
            logging.warning(f"Column {input_col} not found in metadata, possibly from having more partitions than assignable tasks.")
            return partitions

        for index, group_pair in enumerate(metadata.group_by(input_col, maintain_order=True)):
            partition_index = index % n
            print(f'{group_pair[1]} {partition_index}')
            partitions[partition_index].vstack(group_pair[1], in_place=True)
        return partitions

    # def get_n_metadata_partitions(metadata, n, input_col):
    #     partitions = [pl.DataFrame() for _ in range(n)]
    #     for index, group_pair in enumerate(metadata.group_by(input_col)):
    #         partition_index = index % n
    #         partitions[partition_index].vstack(group_pair[1], in_place=True)
    #     return partitions

    # For assigning a partition based on user-provided flag
    partition = get_n_metadata_partitions_evenly(metadata, args.partition[0])[args.partition[1]]

    n_processes = min(args.multiprocessing_count, cpu_count())  # Replace with the number of desired splits
    metadata_splits = get_n_metadata_partitions_evenly(partition, n_processes)

    # Create a pool of processes, using the number of available CPUs
    multiprocessing_args = [(split, args, args.partition[1], process_index, input_col, output_col) for process_index, split in enumerate(metadata_splits)]
    # with Pool(processes=n_processes) as pool:
    with get_context("spawn").Pool(processes=n_processes) as pool:
        # Map the function to each DataFrame in parallel
        results = pool.starmap(annotate, multiprocessing_args)
    logging.debug(f"Natural exit.")
    

if __name__ == "__main__":
    main()
