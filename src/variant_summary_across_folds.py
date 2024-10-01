import pandas as pd
import numpy as np
import os
from utils.helpers import *
import logging

def main(args = None):
    score_dict = {}
    
    if args.input_metadata:
        metadata = pd.read_csv(args.input_metadata, sep='\t')
        # Check for MODEL_ID_COL-SCORE duplicates
        duplicates = metadata[metadata.duplicated(subset=[SUMMARIZE_OUT_PATH_COL, SCORE_OUT_PATH_PREFIX_COL], keep=False)]
        if not duplicates.empty:
            logging.error(f"Duplicate entries found for {SUMMARIZE_OUT_PATH_COL} and {SCORE_OUT_PATH_PREFIX_COL} combination:\n{duplicates}\nRemove the duplicates and try again.")
            exit(1)
    elif args.score_output_path_prefixes and args.summarize_output_path:
        # Create a DataFrame with the score_output_path_prefixes
        metadata = pd.DataFrame({SCORE_OUT_PATH_PREFIX_COL: args.score_output_path_prefixes, SUMMARIZE_OUT_PATH_COL: [args.summarize_output_path] * len(args.score_output_path_prefixes), SPLIT_PER_CHROMOSOME_COL: [args.split_per_chromosome] * len(args.score_output_path_prefixes)})
    
    # Define chromosome suffixes
    chromosome_suffixes = [f'chr{str(i)}.' for i in range(1, 24)] + ['chrX.', 'chrY.']

    # Check for file existence
    missing_score_out_path_prefixes = []
    for summarize_output_path, group in metadata.groupby(SUMMARIZE_OUT_PATH_COL):
        for file_index, score_out_path_prefix in enumerate(group[SCORE_OUT_PATH_PREFIX_COL]):
            is_split_per_chromosome = args.split_per_chromosome or (SPLIT_PER_CHROMOSOME_COL in group.columns and group[SPLIT_PER_CHROMOSOME_COL].iloc[file_index] == True)
            if is_split_per_chromosome:
                has_at_least_one_chr_file = any(os.path.isfile(f"{score_out_path_prefix}{suffix}variant_scores.tsv") for suffix in chromosome_suffixes)
                # Concatenate all the chromosome-specific variant scores
                if not has_at_least_one_chr_file:
                    missing_score_out_path_prefixes.append(score_out_path_prefix)
            else:
                # If not split per chromosome, just read the original file
                variant_score_file = f"{score_out_path_prefix}variant_scores.tsv"
                if not os.path.isfile(variant_score_file):
                    missing_score_out_path_prefixes.append(score_out_path_prefix)
    if missing_score_out_path_prefixes:
        logging.error(f"Missing TSV files for the following prefixes (accounting for split-per-chromosome if provided):\n{missing_score_out_path_prefixes}")
        exit(1)

    # Iterate through each group of rows that share the same summarize_output_path
    for summarize_output_path, group in metadata.groupby(SUMMARIZE_OUT_PATH_COL):
        for file_index, score_out_path_prefix in enumerate(group[SCORE_OUT_PATH_PREFIX_COL]):
            is_split_per_chromosome = args.split_per_chromosome or (SPLIT_PER_CHROMOSOME_COL in group.columns and group[SPLIT_PER_CHROMOSOME_COL].iloc[file_index] == True)
            if is_split_per_chromosome:
                # Concatenate TSV files based on the prefix and suffixes for chromosomes
                all_variant_scores = []
                for suffix in chromosome_suffixes:
                    variant_score_file = f"{score_out_path_prefix}{suffix}variant_scores.tsv"
                    if os.path.isfile(variant_score_file):
                        variant_scores = pd.read_table(variant_score_file)
                        all_variant_scores.append(variant_scores)
                # Concatenate all the chromosome-specific variant scores
                if all_variant_scores:
                    variant_scores = pd.concat(all_variant_scores, ignore_index=True)
                else:
                    raise FileNotFoundError(f"No TSV files found for prefix {score_out_path_prefix} with the specified suffixes.")
            else:
                # If not split per chromosome, just read the original file
                variant_score_file = f"{score_out_path_prefix}variant_scores.tsv"
                variant_scores = pd.read_table(variant_score_file)
            score_dict[file_index] = variant_scores

        os.makedirs(os.path.dirname(summarize_output_path), exist_ok=True)

        variant_scores = score_dict[0][get_variant_schema(args.schema)].copy()

        for file_index in score_dict:
            assert score_dict[file_index]['chr'].tolist() == variant_scores['chr'].tolist()
            assert score_dict[file_index]['pos'].tolist() == variant_scores['pos'].tolist()
            assert score_dict[file_index]['allele1'].tolist() == variant_scores['allele1'].tolist()
            assert score_dict[file_index]['allele2'].tolist() == variant_scores['allele2'].tolist()
            assert score_dict[file_index]['variant_id'].tolist() == variant_scores['variant_id'].tolist()

        for score in ["logfc", "abs_logfc", "jsd", "logfc_x_jsd", "abs_logfc_x_jsd", "active_allele_quantile",
                    "logfc_x_active_allele_quantile", "abs_logfc_x_active_allele_quantile", "jsd_x_active_allele_quantile",
                    "logfc_x_jsd_x_active_allele_quantile", "abs_logfc_x_jsd_x_active_allele_quantile"]:
            if score in score_dict[0]:
                variant_scores.loc[:, (score + '.mean')] = np.mean(np.array([score_dict[fold][score].tolist()
                                                                        for fold in score_dict]), axis=0)
                if score + '.pval' in score_dict[0]:
                    variant_scores.loc[:, (score + '.mean' + '.pval')] = geo_mean_overflow([score_dict[fold][score + '.pval'].values for fold in score_dict])
                elif score + '_pval' in score_dict[0]:
                    variant_scores.loc[:, (score + '.mean' + '.pval')] = geo_mean_overflow([score_dict[fold][score + '_pval'].values for fold in score_dict])

        logging.debug(f"Summary score table:\n{variant_scores.head()}\nSummary score table shape: {variant_scores.shape}")

        variant_scores.to_csv(summarize_output_path,\
                            sep="\t",\
                            index=False)

        logging.info(f"Summary output written to: {summarize_output_path}")


if __name__ == "__main__":
    main()
