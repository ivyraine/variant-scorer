import sys
import argparse

scoring_args = {
    ("-l", "--variant-list",): { "type": str, "help": "a TSV file containing a list of variants to score.", "required": True},
    ("-g", "--genome",): {"type": str, "help": "Genome fasta", "required": True},
    ("-m", "--model",): {"type": str, "help": "ChromBPNet model to use for variant scoring", "required": True},
    ("-scout", "--scoring-output-prefix",): {"type": str, "help": "Path to storing snp effect score predictions, and for summarizing. Directory should already exist.", "required": True},
    ("-s", "--chrom-sizes",): {"type": str, "help": "Path to TSV file with chromosome sizes", "required": True},
    ("-sc", "--schema",): {"type": str, "choices": ['bed', 'plink', 'plink2', 'chrombpnet', 'original'], "default": 'chrombpnet', "help": "Format for the input variants list"},
    ("-ps", "--peak-chrom-sizes"): {"type": str, "help": "Path to TSV file with chromosome sizes for peak genome"},
    ("-pg", "--peak-genome",): {"type": str, "help": "Genome fasta for peaks"},
    ("-b", "--bias"): {"type": str, "help": "Bias model to use for variant scoring"},
    ("-li", "--lite"): {"action": "store_true", "help": "Models were trained with chrombpnet-lite"},
    ("-dm", "--debug-mode"): {"action": "store_true", "help": "Display allele input sequences"},
    ("-bs", "--batch-size"): {"type": int, "default": 512, "help": "Batch size to use for the model"},
    ("-p", "--peaks"): {"type": str, "help": "Bed file containing peak regions"},
    ("-n", "--num-shuf"): {"type": int, "default": 10, "help": "Number of shuffled scores per SNP"},
    ("-t", "--total-shuf"): {"type": int, "help": "Total number of shuffled scores across all SNPs. Overrides --num_shuf"},
    ("-mp", "--max-peaks"): {"type": int, "help": "Maximum number of peaks to use for peak percentile calculation"},
    ("-c", "--chrom"): {"type": str, "help": "Only score SNPs in selected chromosome"},
    ("-r", "--random-seed"): {"type": int, "default": 1234, "help": "Random seed for reproducibility when sampling"},
    ("--no-hdf5",): {"action": "store_true", "help": "Do not save detailed predictions in hdf5 file"},
    ("-nc", "--num-chunks"): {"type": int, "default": 10, "help": "Number of chunks to divide SNP file into"},
    ("-fo", "--forward-only"): {"action": "store_true", "help": "Run variant scoring only on forward sequence"},
    ("-st", "--shap-type"): {"nargs": '+', "default": ["counts"], "help": "Specify SHAP value type(s) to calculate"},
}

summary_args = {
    ("-scout", "--scoring-output-prefix",): {"type": str, "help": "Path to storing snp effect score predictions, and for summarizing. Directory should already exist.", "required": True},
    ("-suout", "--summary-output-prefix",): { "type": str, "help": "Path prefix for storing the summary file with average scores across folds; directory should already exist" , "required": True},
    ("-sc", "--schema",): { "type": str, "choices": ['bed', 'plink', 'plink2', 'chrombpnet', 'original'], "default": 'chrombpnet', "help": "Format for the input variants list"},
    ("-sl", "--score-list"): { "nargs": '+', "help": "Names of variant score files that will be used to generate summary. Required if --no-scoring is used."},
}

annotation_args = {
    ("-aout", "--annotation-output-prefix",): { "type": str, "help": "Path prefix for storing the annotated file; directory should already exist.", "required": True},
    ("-l", "--variant-list",): { "type": str, "help": "a TSV file containing a list of variants to score.", "required": True},
    ("-sc", "--schema",): {"type": str, "choices": ['bed', 'plink', 'plink2', 'chrombpnet', 'original'], "default": 'chrombpnet', "help": "Format for the input variants list"},
    ("-p", "--peaks"): { "type": str, "help": "Adds overlapping peaks information. Bed file containing peak regions."},
    ("-cg", "--closest-genes"): { "type": str, "help": "Adds closest gene annotations. Bed file containing gene regions. Default amount is 3." },
    ("-cgc", "--closest-gene-count"): { "type": int, "help": "Changes the number of closest genes (using the -g flag) annotated to the specified number." },
    ("-aa", "--add-adastra"): { "action": "store_true", "help": "Annotate with ADASTRA. Requires downloadable ADASTRA data, which you must provide using the -aatf and -aact flags." },
    ("-aatf", "--add-adastra-tf"): { "type": argparse.FileType('r'), "help": "The file containing ADASTRA TF data." },
    ("-aact", "--add-adastra-celltype"): { "type": argparse.FileType('r'), "help": "The file containing ADASTRA cell type data." },
    ("-th", "--threads"): { "type": int, "help": "The maximum amount of threads to use, where possible." },
    ("-r2", "--r2"): { "type": str, "help": "Adds r2 annotations. Requires a PLINK .ld file." },
    ("-v", "--verbose"): { "action": "store_true", "help": "Enable detailed logging." },
}

shap_args = {
    ("-l", "--variant-list",): { "type": str, "help": "a TSV file containing a list of variants to score." , "required": True},
    ("-g", "--genome",): { "type": str, "help": "Genome fasta" , "required": True},
    ("-m", "--model",): {"type": str, "help": "ChromBPNet model to use for variant scoring", "required": True},
    ("-shout", "--shap-output-prefix",): { "type": str, "help": "Path to storing snp effect score predictions from the script, directory should already exist", "required": True},
    ("-s", "--chrom-sizes",): {"type": str, "help": "Path to TSV file with chromosome sizes", "required": True},
    ("-sc", "--schema",): {"type": str, "choices": ['bed', 'plink', 'plink2', 'chrombpnet', 'original'], "default": 'chrombpnet', "help": "Format for the input variants list"},
    ("-li", "--lite"): { "action": "store_true", "help": "Models were trained with chrombpnet-lite" },
    ("-dm", "--debug-mode"): { "action": "store_true", "help": "Display allele input sequences" },
    ("-bs", "--batch-size"): { "type": int, "default": 10000, "help": "Batch size to use for the model" },
    ("-c", "--chrom"): { "type": str, "help": "Only score SNPs in selected chromosome" },
    ("-st", "--shap-type"): { "nargs": '+', "default": ["counts"], "help": "Specify SHAP value type(s) to calculate" },
}

def update_conditional_args(parser):
    parser.add_argument("--no-scoring", action='store_true', help="Exclude the scoring step of the pipeline")
    parser.add_argument("--no-summary", action='store_true', help="Exclude the fold summary step of the pipeline")
    parser.add_argument("--no-annotation", action='store_true', help="Exclude the annotation step of the pipeline")
    parser.add_argument("--no-shap", action='store_true', help="Exclude the shap step of the pipeline")

def fetch_scoring_args():
    parser = argparse.ArgumentParser()
    args_dict = {k: args_dict[k] for k in sorted(scoring_args, key=lambda x: x[0])}
    for arg_names, kwargs in args_dict.items():
        parser.add_argument(*arg_names, **kwargs)
    args = parser.parse_args()
    return args

def fetch_summary_args():
    parser = argparse.ArgumentParser()
    args_dict = {k: args_dict[k] for k in sorted(summary_args, key=lambda x: x[0])}
    for arg_names, kwargs in args_dict.items():
        parser.add_argument(*arg_names, **kwargs)
    args = parser.parse_args()
    return args

def fetch_annotation_args():
    parser = argparse.ArgumentParser()
    args_dict = {k: args_dict[k] for k in sorted(annotation_args, key=lambda x: x[0])}
    for arg_names, kwargs in args_dict.items():
        parser.add_argument(*arg_names, **kwargs)
    args = parser.parse_args()
    return args

def fetch_shap_args():
    parser = argparse.ArgumentParser()
    args_dict = {k: args_dict[k] for k in sorted(shap_args, key=lambda x: x[0])}
    for arg_names, kwargs in args_dict.items():
        parser.add_argument(*arg_names, **kwargs)
    args = parser.parse_args()
    return args

def fetch_main_parser():
    parser = argparse.ArgumentParser(add_help=False)
    update_conditional_args(parser)
    conditional_args, _ = parser.parse_known_args()

    parser = argparse.ArgumentParser()
    # Combine dictionaries of arguments for each subcommand.
    # We do this rather than argparse's argument_groups b.c.
    # there are some arguments that are shared between subcommands.
    args_dict = {}

    if not conditional_args.no_scoring:
        args_dict.update(scoring_args)
    else:
        summary_args[("-sl", "--score_list")]["required"] = True

    if not conditional_args.no_summary:
        args_dict.update(summary_args)

    if not conditional_args.no_annotation:
        args_dict.update(annotation_args)

    if not conditional_args.no_shap:
        args_dict.update(shap_args)

    args_dict = {k: args_dict[k] for k in sorted(args_dict, key=lambda x: x[0])}
    for arg_names, kwargs in args_dict.items():
        parser.add_argument(*arg_names, **kwargs)

    # Add conditional arguments back to the parser
    update_conditional_args(parser)

    # Check if a subcommand has been provided; if not, print help and exit
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    # Parse all args again
    args = parser.parse_args()

    return args
