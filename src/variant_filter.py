import pandas as pd
import numpy as np
import os
from collections import namedtuple
import operator
import logging
import variant_scoring

from utils.argmanager import *
from utils.helpers import *

Condition= namedtuple("Condition", ["key", "operator", "value"])

def main(args = None):
    if args is None:
        logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')
        args = fetch_filter_args()

    input_file = get_annotation_output_file(args.annotation_output_dir, args.sample_name)

    df = pd.read_csv(input_file, sep="\t")
    existing_headers = df.columns

    conditions = []

    if args.filter_lower is not None:
        for filter_lower_condition in args.filter_lower:
            key, value = filter_lower_condition.split(":")
            conditions.append(Condition(key, operator.le, float(value)))
            # Check if key exists in input_file as a header:
            if key not in existing_headers:
                raise ValueError(f"Error: The key '{key}' does not exist in the input file '{input_file}'")
    
    if args.filter_upper is not None:
        for filter_upper_condition in args.filter_upper:
            key, value = filter_upper_condition.split(":")
            conditions.append(Condition(key, operator.ge, float(value)))
            # Check if key exists in input_file as a header:
            if key not in existing_headers:
                raise ValueError(f"Error: The key '{key}' does not exist in the input file '{input_file}'")
 
    filter_mask = pd.Series([True] * len(df))   

    logic_op = operator.and_ if args.filter_logic == "and" else operator.or_

    # Apply --filter-upper and --filter-lower conditions
    for col, op, value in conditions:
        temp_mask = op(df[col], value)
        filter_mask = logic_op(filter_mask, temp_mask)
    
    # Apply --max-percentile-threshold
    if args.max_percentile_threshold is not None:
        if "max_percentile.mean" not in existing_headers:
            logging.warning(f"Warning: The key 'max_percentile.mean' does not exist in the input file '{input_file}. Ignoring this filter.'")
        else:
            filter_mask &= df["max_percentile.mean"] >= args.max_percentile_threshold

    # Apply --peak-variants-only
    if args.peak_variants_only == True:
        if "peak_overlap" not in existing_headers:
            logging.warning(f"Warning: The key 'peak_overlap' does not exist in the input file '{input_file}. Ignoring this filter.'")
        else:
            filter_mask &= df["peak_overlap"] == True

    filtered_variants = df[filter_mask]

    # Check if any variants remain after filtering
    if len(filtered_variants) == 0:
        logging.warning(f"No variants remain after filtering. Exiting.")
        exit(1)

    out_file = f"{get_filter_output_file(args.filter_output_dir, args.sample_name)}"
    filtered_variants.to_csv(out_file,\
                  sep="\t",\
                  index=False)

    logging.info(f"Running scoring step on the filtered variants to generate predictions...")
    variant_scoring.main(args, filter_output_dir_override=args.filter_output_dir, filtered_variants_df_override=filtered_variants)
    
    logging.info(f"Filter step completed! Output written to: {out_file}")


if __name__ == "__main__":
    main()
