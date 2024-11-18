# import pandas as pd
# import dask as pd
# import pandas as pd
import polars as pl
import numpy as np
import os
from utils.helpers import *
import logging
import time
from subprocess import check_output
from multiprocessing import Pool, cpu_count, get_context

def summarize(metadata_split, args, partition_index, process_index):

    setup_logging(logging.getLogger(), args.verbose, prefix=f"[Partition {partition_index} Process {process_index}]: ")

    logging.debug(f"Beginning process ({process_index+1}/{args.multiprocessing_count}).")

    # Define chromosome suffixes
    chromosome_suffixes = [f'chr{str(i)}.' for i in range(1, 24)] + ['chrX.', 'chrY.']

    logging.debug(f'Checking for missing or invalid TSV files.')
    missing_scores = set()
    skippable_summaries = set()
    for index, group_pair in enumerate(metadata_split.group_by(SUMMARIZE_OUT_PATH_COL)):
        summarize_out_tuple, group = group_pair
        summarize_out = summarize_out_tuple[0]
        score_out_path_prefixes = group[SCORE_OUT_PATH_PREFIX_COL]
        for score_out_path_prefix in score_out_path_prefixes:
            scores_file = get_score_file_path(score_out_path_prefix, args.scores_suffix, chr='all')
            if not os.path.isfile(scores_file):
                missing_scores.add(scores_file)
                logging.debug(f'Missing scores file: {scores_file}')
                skippable_summaries.add(summarize_out)
            # if args.expected_line_count and args.expected_line_count != int(check_output(["wc", "-l", scores_file]).split()[0]):
            #     incorrect_scores.add(scores_file)
            #     skippable_summaries.add(summarize_out)
            # TODO: check chr files
    
    if os.path.isfile(args.invalid_file_log):
        os.remove(args.invalid_file_log)
        
    if missing_scores:
        missing_files_str = "\n".join(missing_scores)
        # incorrect_scores_str = "\n".join(incorrect_scores)
        logging.warning(f'The following {len(missing_scores)} TSV file(s) are missing:\n{missing_files_str}')
        # logging.warning(f'The following {len(incorrect_scores)} TSV file(s) have incorrect line counts:\n{incorrect_scores_str}')
        if args.invalid_file_log:
            with open(args.invalid_file_log, 'w') as f:
                f.write(f'{missing_files_str}\n')
                # f.write(f'{incorrect_scores_str}\n')
        if not args.skip_invalid_scores:
            logging.error('Please fix the missing or incorrect files and try again, or use the --skip-invalid-scores flag.')
            exit(1)

    incorrect_scores = set()

    
    summary_count = metadata_split.group_by(SUMMARIZE_OUT_PATH_COL).all().height
    print(f"Processing {summary_count} summaries.")

    for index, group_pair in enumerate(metadata_split.group_by(SUMMARIZE_OUT_PATH_COL)):
        is_skipping = False
        _summarize_output_path, group = group_pair
        summarize_output_path = _summarize_output_path[0]
        score_dict = []
        score_out_path_prefixes = group[SCORE_OUT_PATH_PREFIX_COL]
        read_time = 0.0
        processing_time = 0.0
        prefix_paths_str = '\n'.join(score_out_path_prefixes)
        logging.debug(f"[Thread {process_index}] ({index+1}/{summary_count}) Processing {summarize_output_path} with scores at the following prefixes:\n{prefix_paths_str}.")
        if args.skip_invalid_scores and summarize_output_path in skippable_summaries:
            logging.debug(f'Skipping {summarize_output_path} as it has missing or invalid scores.')
            continue
        if args.skip_existing_outputs and os.path.isfile(summarize_output_path):
            logging.debug(f'Skipping {summarize_output_path} as it already exists.')
            continue
        for file_index, score_out_path_prefix in enumerate(score_out_path_prefixes):
            is_split_per_chromosome = args.split_per_chromosome or (SPLIT_PER_CHROMOSOME_COL in group.columns and group[SPLIT_PER_CHROMOSOME_COL].iloc[file_index] == True)
            # TODO: move split chr merging elsewhere
            if is_split_per_chromosome:
                # Concatenate TSV files based on the prefix and suffixes for chromosomes
                partial_scores = []
                for suffix in chromosome_suffixes:
                    variant_score_file = get_score_file_path(score_out_path_prefix, args.scores_suffix, chr=suffix)
                    if os.path.isfile(variant_score_file):
                        read_start_time = time.time()
                        # variant_scores = pd.read_csv(variant_score_file, usecols=columns_to_access, sep='\t')
                        cur_scores = pl.read_csv(variant_score_file, separator='\t')
                        if cur_scores.height == 0:
                            incorrect_scores.add(variant_score_file)
                            logging.warning(f"Skipping {variant_score_file} as it has no data.")
                            continue
                        read_time += time.time() - read_start_time
                        partial_scores.append(cur_scores)
                # Concatenate all the chromosome-specific variant scores
                if partial_scores:
                    cur_scores = pl.concat(partial_scores, ignore_index=True)
                else:
                    logging.error(f"No TSV files found for prefix {score_out_path_prefix} with the specified suffixes.")
                    raise FileNotFoundError(f"No TSV files found for prefix {score_out_path_prefix} with the specified suffixes.")
            else:
                # If not split per chromosome, just read the original file
                variant_score_file = f"{score_out_path_prefix}variant_scores.from_nautilus.tsv"
                variant_score_file = get_score_file_path(score_out_path_prefix, args.scores_suffix, chr='all')

                read_start_time = time.time()
                cur_scores = pl.read_csv(variant_score_file, separator='\t')
                if args.expected_line_count and cur_scores.height+1 != args.expected_line_count:
                    logging.warning(f"Skipping {summarize_output_path} due to invalid data.")
                    incorrect_scores.add(variant_score_file)
                    is_skipping = True
                read_time += time.time() - read_start_time
            score_dict.append(cur_scores)
        
        if is_skipping:
            incorrect_scores_str = "\n".join(incorrect_scores)
            with open(args.invalid_file_log, 'a') as f:
                f.write(f'{incorrect_scores_str}\n')
            incorrect_scores = set()
            continue

        os.makedirs(os.path.dirname(summarize_output_path), exist_ok=True)

        result_scores = score_dict[0].select('chr','pos','allele1','allele2','variant_id')

        processing_time_start = time.time()
        for score_type in ["logfc", "jsd", "active_allele_quantile" ]:

            # Calculate mean of scores
            score_mean = pl.DataFrame().with_columns(
                # Get the score column from each DataFrame
                [df[score_type].alias(f"{score_type}_{i}") for i, df in enumerate(score_dict)]
            ).mean_horizontal().alias(f"{score_type}.mean")
            result_scores = result_scores.with_columns(score_mean)

            # Calculate geometric mean of p-values
            pval_name = f'{score_type}_pval' if f'{score_type}_pval' in score_dict[0] else f'{score_type}.pval'
            score_mean_pval = pl.DataFrame().with_columns(
                [df[pval_name].alias(f"{score_type}.pval_{idx}").log() for idx, df in enumerate(score_dict)]
            ).mean_horizontal(
            ).exp(
            ).alias(
                f"{score_type}.mean.pval"
            )
            result_scores = result_scores.with_columns(score_mean_pval)
        processing_time += time.time() - processing_time_start
        
        result_scores.write_csv(summarize_output_path, separator="\t")

        logging.info(f"Summary output written to: {summarize_output_path}")
        logging.debug(f"Time taken to read all the TSV files for {summarize_output_path}: {read_time:.4f} seconds")
        logging.debug(f"Time taken to process the data for {summarize_output_path}: {processing_time:.4f} seconds")
    
    logging.debug(f"Process ({process_index+1}/{args.multiprocessing_count}) complete.")

def main(args = None):

    if args.input_metadata:
        metadata = pl.read_csv(args.input_metadata, separator='\t')
        # Check for MODEL_ID_COL-SCORE duplicates
        duplicates = metadata.filter(metadata.select(SUMMARIZE_OUT_PATH_COL, SCORE_OUT_PATH_PREFIX_COL).is_duplicated())
        if not duplicates.is_empty():
            logging.error(f"Duplicate entries found for {SUMMARIZE_OUT_PATH_COL} and {SCORE_OUT_PATH_PREFIX_COL} combination:\n{duplicates}\nRemove the duplicates and try again.")
            exit(1)
    elif args.score_output_path_prefixes and args.summarize_output_path:
        # Create a DataFrame with the score_output_path_prefixes
        metadata = pl.DataFrame({
            SCORE_OUT_PATH_PREFIX_COL: args.score_output_path_prefixes,
            SUMMARIZE_OUT_PATH_COL: [args.summarize_output_path] * len(args.score_output_path_prefixes),
            SPLIT_PER_CHROMOSOME_COL: [args.split_per_chromosome] * len(args.score_output_path_prefixes)
        })
    
    def get_n_metadata_partitions(metadata, n):
        partitions = []
        rows_per_split = len(metadata) // n
        for i in range(n):
            start = i * rows_per_split
            end = start + rows_per_split
            # Adjust the last split to include any remaining rows
            if i == n - 1:
                partitions.append(metadata.slice(start, len(metadata) - start).clone())
            else:
                partitions.append(metadata.slice(start, end).clone())
        return partitions

    # For assigning a partition based on user-provided flag
    partition = get_n_metadata_partitions(metadata, args.max_partitions)[args.cur_partition]

    n_processes = min(args.multiprocessing_count, cpu_count())  # Replace with the number of desired splits
    metadata_splits = get_n_metadata_partitions(partition, n_processes)


    # Create a pool of processes, using the number of available CPUs
    multiprocessing_args = [(split, args, args.cur_partition, process_index) for process_index, split in enumerate(metadata_splits)]
    # with Pool(processes=n_processes) as pool:
    with get_context("spawn").Pool(processes=n_processes) as pool:
        # Map the function to each DataFrame in parallel
        results = pool.starmap(summarize, multiprocessing_args)
    logging.debug(f"Natural exit.")


if __name__ == "__main__":
    main()
