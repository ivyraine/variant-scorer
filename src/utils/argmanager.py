import sys
import argparse

scoring_args = {
    ("-l", "--variant-list"): { "type": str, "help": "a TSV file containing a list of variants to score.", "required": True},
    ("-g", "--genome"): {"type": str, "help": "Genome fasta.", "required": True},
    ("-m", "--models"): {"type": str, "nargs": '+', "help": "ChromBPNet models to use for variant scoring, whose outputs will be labeled with numerical indexes beginning from 0 in the order they are provided.", "required": True},
    ("-scout", "--scoring-output-dir"): {"type": str, "help": "The directory to store all output files like: <output-dir>/<sample-name>.<index>.variant_scores.tsv; directory should already exist.", "required": True},
    ("-s", "--chrom-sizes"): {"type": str, "help": "Path to TSV file with chromosome sizes", "required": True},
    ("-sa", "--sample-name"): {"type": str, "help": "The prefix that will be prepended to the filename like: <output-dir>/<sample-name>.<index>.variant_scores.tsv", "required": True},
    ("-sc", "--schema"): {"type": str, "choices": ['bed', 'plink', 'plink2', 'chrombpnet', 'original'], "default": 'chrombpnet', "help": "Format for the input variants list."},
    ("-ps", "--peak-chrom-sizes"): {"type": str, "help": "Path to TSV file with chromosome sizes for peak genome."},
    ("-pg", "--peak-genome"): {"type": str, "help": "Genome fasta for peaks."},
    ("-b", "--bias"): {"type": str, "help": "Bias model to use for variant scoring."},
    ("-li", "--lite"): {"action": "store_true", "help": "Models were trained with chrombpnet-lite."},
    ("-dm", "--debug-mode"): {"action": "store_true", "help": "Display allele input sequences."},
    ("-bs", "--batch-size"): {"type": int, "default": 512, "help": "Batch size to use for the model."},
    ("-p", "--peaks"): {"type": str, "help": "Bed file containing peak regions."},
    ("-n", "--num-shuf"): {"type": int, "default": 10, "help": "Number of shuffled scores per SNP."},
    ("-ts", "--total-shuf"): {"type": int, "help": "Total number of shuffled scores across all SNPs. Overrides --num_shuf."},
    ("-mp", "--max-peaks"): {"type": int, "help": "Maximum number of peaks to use for peak percentile calculation."},
    ("-c", "--chrom"): {"type": str, "help": "Only score SNPs in selected chromosome."},
    ("-r", "--random-seed"): {"type": int, "default": 1234, "help": "Random seed for reproducibility when sampling."},
    ("--no-hdf5",): {"action": "store_true", "help": "Do not save detailed predictions in hdf5 file."},
    ("-nc", "--num-chunks"): {"type": int, "default": 10, "help": "Number of chunks to divide SNP file into."},
    ("-fo", "--forward-only"): {"action": "store_true", "help": "Run variant scoring only on forward sequence."},
    ("-st", "--shap-type"): {"nargs": '+', "default": ["counts"], "help": "Specify shap value type(s) to calculate."},
    ("-scf", "--score-filenames"): { "nargs": '+', "help": "A list of file names of variant score files that will be used to overwrite the otherwise generated index filenames, and will be used like so: <scoring-output-dir>/<score-filename>.{tsv,h5} for each file in the list. Generally only needed if --no-scoring is used."},
    ("-v", "--verbose"): { "action": "store_true", "help": "Enable detailed logging." },
}

summary_args = {
    ("-scout", "--scoring-output-dir"): {"type": str, "help": "The directory to store all output files like: <output-dir>/<sample-name>.<index>.variant_scores.tsv; directory should already exist.", "required": True},
    ("-suout", "--summary-output-dir"): { "type": str, "help": "The directory to store the summary file with average scores across folds; directory should already exist." , "required": True},
    ("-sa", "--sample-name"): {"type": str, "help": "The prefix to be prepended to the filename, like: <output-dir>/<sample-name>.<index>.variant_scores.tsv.", "required": True},
    ("-sc", "--schema"): { "type": str, "choices": ['bed', 'plink', 'plink2', 'chrombpnet', 'original'], "default": 'chrombpnet', "help": "Format for the input variants list."},
    ("-scf", "--score-filenames"): { "nargs": '+', "help": "A list of file names of variant score files to be used to overwrite the otherwise generated index filenames, and will be used like so: <scoring-output-dir>/<score-filename>.{tsv,h5} for each file in the list. Generally only needed if --no-scoring is used.", "required": True},
    ("-v", "--verbose"): { "action": "store_true", "help": "Enable detailed logging." },
}

annotation_args = {
    ("-suout", "--summary-output-dir"): { "type": str, "help": "The directory to store the summary file with average scores across folds like so: <annotation-output-dir>/<sample-name>.annotations.tsv; directory should already exist." , "required": True},
    ("-sa", "--sample-name"): {"type": str, "help": "The prefix to be prepended to the filename like: <output-dir>/<sample-name>.<index>.variant_scores.tsv.", "required": True},
    ("-aout", "--annotation-output-dir"): { "type": str, "help": "The directory to store the unfiltered annotations file like so: <annotation-output-dir>/<sample-name>.annotations.tsv. This directory should already exist.", "required": True},
    ("-sc", "--schema"): {"type": str, "choices": ['bed', 'plink', 'plink2', 'chrombpnet', 'original'], "default": 'chrombpnet', "help": "Format for the input variants list."},
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

filter_args = {
    ("-aout", "--annotation-output-dir"): { "type": str, "help": "The directory to store the unfiltered annotations file like so: <annotation-output-dir>/<sample-name>.annotations.tsv. This directory should already exist.", "required": True},
    ("-sa", "--sample-name"): {"type": str, "help": "The prefix to be prepended to the filename like: <output-dir>/<sample-name>.<index>.variant_scores.tsv.", "required": True},
    ("-fout", "--filter-output-dir"): {"type": str, "help": "The directory to store the filtered annotations file like so: <filter-output-dir>/<sample-name>.annotations.filtered.tsv. This directory should already exist.", "required": True},
    ("-lo", "--filter-lower"): {"type": str, "nargs": "+", "help": "Removes all variants from the annotations list containing a field(s) with a value greater than or equal to the specified threshold(s). This flag accepts a list of pairs, where each pair contains the field and value delimited by a colon. By default this flag is set to `-lo abs_logfc.mean.pval:0.01 jsd.mean.pval:0.01`", "default": ["abs_logfc.mean.pval:0.01", "jsd.mean.pval:0.01"]},
    ("-up", "--filter-upper"): {"type": str, "nargs": "+", "help": "Removes all variants from the annotations list containing a field(s) with a value lower than or equal to the specified threshold(s). This flag accepts a list of pairs, where each pair contains the field and value delimited by a colon. An example of this flag would be: `-up field1:0.5 field2:0.1`."},
    ("-fl", "--filter-logic"): {"type": str, "choices": ["and", "or"], "default": "and", "help": "The logic to use when filtering variants based on the filter-lower and filter-upper flags, excluding --max-percentile-threshold and --peak-variants-only, which always use 'and' logic. The default is 'and', which means that a variant will be removed if it fails any of the filter conditions. If set to 'or', a variant will be removed if it fails all of the filter conditions."},
    ("-mpt", "--max-percentile-threshold"): {"type": float, "help": "Removes all variants from the annotations list whose max_percentile.mean (generated from running the scoring step with the -p flag, followed by the summary step) is less than or equal to the specified threshold, and always uses 'and' threshold logic. The default is 0.05.", "default": 0.05},
    ("-p", "--peaks"): { "type": str, "help": "Adds overlapping peaks information. Bed file containing peak regions."},
    ("-po", "--peak-variants-only"): {"action": "store_true", "help": "Only keep variants whose peak_overlap (generated from the annotation step with the -p flag) is equal to True, and always uses 'and' threshold logic. Peaks file must be provided if used."},
    ("-v", "--verbose"): { "action": "store_true", "help": "Enable detailed logging." },
}

shap_args = {
    # ("-aout", "--annotation-output-dir"): { "type": str, "help": "The directory to store the annotations file like so: <annotation-output-dir>/<sample-name>.annotations.tsv. This directory should already exist.", "required": True},
    ("-fout", "--filter-output-dir"): {"type": str, "help": "The directory to store the filtered annotations file like so: <filter-output-dir>/<sample-name>.annotations.filtered.tsv. This directory should already exist.", "required": True},
    ("-sa", "--sample-name"): {"type": str, "help": "The prefix to be prepended to the filename like: <output-dir>/<sample-name>.<index>.variant_scores.tsv.", "required": True},
    # ("-t", "--score-filenames"): { "nargs": '+', "help": "A list of file names of variant score files that will be used to overwrite the otherwise generated index filenames, and will be used like so: <scoring-output-dir>/<file> for each file in the list. Generally only needed if --no-scoring is used."},
    ("-shout", "--shap-output-dir"): { "type": str, "help": "The directory that will store the SNP effect score predictions from the script, directory should already exist.", "required": True},
    ("-g", "--genome"): { "type": str, "help": "Genome fasta." , "required": True},
    ("-m", "--models"): {"type": str, "nargs": '+', "help": "ChromBPNet models to use for variant scoring, whose outputs will be labeled with numerical indexes beginning from 0 in the order they are provided.", "required": True},
    ("-s", "--chrom-sizes"): {"type": str, "help": "Path to TSV file with chromosome sizes.", "required": True},
    # ("-sc", "--schema"): {"type": str, "choices": ['bed', 'plink', 'plink2', 'chrombpnet', 'original'], "default": 'chrombpnet', "help": "Format for the input variants list."},
    ("-li", "--lite"): { "action": "store_true", "help": "Models were trained with chrombpnet-lite."},
    ("-dm", "--debug-mode"): { "action": "store_true", "help": "Display allele input sequences."},
    ("-bs", "--batch-size"): { "type": int, "default": 10000, "help": "Batch size to use for the model."},
    ("-c", "--chrom"): { "type": str, "help": "Only score SNPs in selected chromosome."},
    ("-shf", "--shap-filenames"): { "nargs": '+', "help": "A list of file names of shap files to be used to overwrite the otherwise generated index filenames, and will be used like so: <shap-output-dir>/<shap-filename>.{h5,bw} for each file in the list."},
    ("-st", "--shap-type"): { "nargs": '+', "default": ["counts"], "help": "Specify shap value type(s) to calculate." },
    ("-v", "--verbose"): { "action": "store_true", "help": "Enable detailed logging." },
}

viz_args = {
    ("-shout", "--shap-output-dir"): { "type": str, "help": "The directory that will store the SNP effect score predictions from the script, directory should already exist.", "required": True},
}

def update_conditional_args(parser):
    parser.add_argument("--no-scoring", action='store_true', help="Exclude the scoring step of the pipeline")
    parser.add_argument("--no-summary", action='store_true', help="Exclude the fold summary step of the pipeline")
    parser.add_argument("--no-annotation", action='store_true', help="Exclude the annotation step of the pipeline")
    parser.add_argument("--no-filter", action='store_true', help="Exclude the filter step of the pipeline")
    parser.add_argument("--no-shap", action='store_true', help="Exclude the shap step of the pipeline")
    parser.add_argument("--no-viz", action='store_true', help="Exclude the visualization step of the pipeline")

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

def fetch_filter_args():
    parser = argparse.ArgumentParser()
    args_dict = {k: args_dict[k] for k in sorted(filter_args, key=lambda x: x[0])}
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

def fetch_viz_args():
    parser = argparse.ArgumentParser()
    args_dict = {k: args_dict[k] for k in sorted(viz_args, key=lambda x: x[0])}
    for arg_names, kwargs in args_dict.items():
        parser.add_argument(*arg_names, **kwargs)
    args = parser.parse_args()
    return args

def fetch_main_parser():
    parser = argparse.ArgumentParser(add_help=False)
    update_conditional_args(parser)
    conditional_args, _ = parser.parse_known_args()

    # Combine dictionaries of arguments for each subcommand.
    # We do this rather than argparse's argument_groups b.c.
    # there are some arguments that are shared between subcommands.
    args_dict = {}
    included_modules = []
    parser = argparse.ArgumentParser(add_help=True)

    if not conditional_args.no_scoring:
        summary_args[("-scf", "--score-filenames")]["required"] = False
        args_dict.update(scoring_args)
        included_modules.append("scoring")

    if not conditional_args.no_summary:
        args_dict.update(summary_args)
        included_modules.append("summary")

    if not conditional_args.no_annotation:
        args_dict.update(annotation_args)
        included_modules.append("annotation")

    if not conditional_args.no_filter:
        args_dict.update(filter_args)
        included_modules.append("filter")

    if not conditional_args.no_shap:
        args_dict.update(shap_args)
        included_modules.append("shap")

    if not conditional_args.no_viz:
        args_dict.update(viz_args)
        included_modules.append("viz")

    args_dict = {k: args_dict[k] for k in sorted(args_dict, key=lambda x: x[0])}
    for arg_names, kwargs in args_dict.items():
        parser.add_argument(*arg_names, **kwargs)

    # Add conditional arguments back to the parser
    update_conditional_args(parser)

    included_modules_str = ", ".join(included_modules)
    parser.description = f"This script (varscore) annotates genetic variants using chrombpnet and similar models. It can be run in multiple steps, each of which can be disabled using the --no-* flags. The steps are: scoring, summary, annotation, filter, and shap. The following flags are shared between the following modules: {included_modules_str}."

    # Parse all args again
    args = parser.parse_args()

    return args
