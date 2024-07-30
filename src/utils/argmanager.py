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
    ("--no-hdf5",): {"action": "store_true", "help": "Prevents saving detailed predictions in hdf5 file during storing, and using those files during the shap and viz steps. Recommended when the variants list is large (>1,000,000)."},
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
    ("-scf", "--score-filenames"): { "nargs": '+', "help": "A list of file names of variant score files to be used to overwrite the otherwise generated index filenames, and will be used like so: <scoring-output-dir>/<score-filename>.{tsv,h5} for each file in the list. Generally only needed if --no-scoring is used."},
    ("-m", "--models"): {"type": str, "nargs": '+', "help": "ChromBPNet models to use for variant scoring, whose outputs will be labeled with numerical indexes beginning from 0 in the order they are provided."},
    ("-v", "--verbose"): { "action": "store_true", "help": "Enable detailed logging." },
}

class ClosestGenesAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, 'closest_genes_file', values[0])
        setattr(namespace, 'closest_gene_count', int(values[1]))

class ClosestGenesInWindowAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, 'closest_genes_in_window_file', values[0])
        setattr(namespace, 'closest_genes_window_size', int(values[1]))

class AdastraAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if values:
            setattr(namespace, 'adastra_tf_file', values[0])
            setattr(namespace, 'adastra_celltype_file', values[1])
        setattr(namespace, self.dest, True)

annotation_args = {
    ("-suout", "--summary-output-dir"): { "type": str, "help": "The directory to store the summary file with average scores across folds like so: <annotation-output-dir>/<sample-name>.annotations.tsv; directory should already exist." , "required": True},
    ("-sa", "--sample-name"): {"type": str, "help": "The prefix to be prepended to the filename like: <output-dir>/<sample-name>.<index>.variant_scores.tsv.", "required": True},
    ("-aout", "--annotation-output-dir"): { "type": str, "help": "The directory to store the unfiltered annotations file like so: <annotation-output-dir>/<sample-name>.annotations.tsv. This directory should already exist.", "required": True},
    ("-sc", "--schema"): {"type": str, "choices": ['bed', 'plink', 'plink2', 'chrombpnet', 'original'], "default": 'chrombpnet', "help": "Format for the input variants list."},
    ("-p", "--peaks"): { "type": str, "help": "Adds overlapping peaks information. Bed file containing peak regions."},
    ("-cg", "--add-closest-genes"): { "nargs": 2, "metavar": ('GENE_BED', 'COUNT'), "action": ClosestGenesAction, "help": "Adds closest gene annotations. Requires a bed file containing gene regions and the number of closest genes." },
    ("-cgw", "--add-closest-genes-in-window"): { "nargs": 2, "metavar": ('GENE_BED', 'WINDOWSIZE'), "action": ClosestGenesInWindowAction, "help": "Adds annotations for the closest genes within a window. Requires a bed file containing gene regions and the number of closest genes." },
    ("-aa", "--add-adastra"): { "nargs": 2, "metavar": ('ADASTRA_TF', 'ADASTRA_CELLTYPE'), "action": AdastraAction, "help": "Annotate with ADASTRA. Provide ADASTRA TF and cell type data files." },
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
    ("-shout", "--shap-output-dir"): { "type": str, "help": "The directory that will store the SNP effect score predictions from the script. This directory should already exist.", "required": True},
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
    ("--no-hdf5",): {"action": "store_true", "help": "Prevents saving detailed predictions in hdf5 file during storing, and using those files during the shap and viz steps. Recommended when the variants list is large (>1,000,000)."},
    ("-v", "--verbose"): { "action": "store_true", "help": "Enable detailed logging." },
}

viz_args = {
    ("-l", "--variant-list"): { "type": str, "help": "a TSV file containing a list of variants to score.", "required": True},
    ("-m", "--models"): {"type": str, "nargs": '+', "help": "ChromBPNet models to use for variant scoring, whose outputs will be labeled with numerical indexes beginning from 0 in the order they are provided.", "required": True},
    ("-sa", "--sample-name"): {"type": str, "help": "The prefix to be prepended to the filename like: <output-dir>/<sample-name>.<index>.variant_scores.tsv.", "required": True},
    ("-fout", "--filter-output-dir"): {"type": str, "help": "The directory to store the filtered annotations file like so: <filter-output-dir>/<sample-name>.annotations.filtered.tsv. This directory should already exist.", "required": True},
    ("-shout", "--shap-output-dir"): { "type": str, "help": "The directory that will store the SNP effect score predictions from the script. This directory should already exist.", "required": True},
    ("-vout", "--viz-output-dir"): {"type": str, "help": "The directory that will store the visualization outputs. This directory should already exist.", "required": True},
    ("-fscf", "--filter-score-filenames"): { "nargs": '+', "help": "A list of file names of filtered variant score files that will be used to overwrite the otherwise generated index filenames, and will be used like so: <filter-output-dir>/<filtered-score-filename>.{tsv,h5} for each file in the list. Generally only needed if --no-scoring is used."},
    # ("--no-hdf5",): {"action": "store_true", "help": "Prevents saving detailed predictions in hdf5 file during storing, and using those files during the shap and viz steps. Recommended when the variants list is large (>1,000,000)."},
    ("-st", "--shap-type"): { "nargs": '+', "default": ["counts"], "help": "Specify shap value type(s) to calculate." },
    ("-predo", "--predictions-override"): { "type": str, "help": "The name of the variant effect predictions' file. Use this if you want to use a name other than the default."},
    ("-shapo", "--shap-override"): { "type": str, "help": "The name of the shap output file. Use this if you want to use a name other than the default."},
    # TODO: Allow users to provide hdf5 files directly
    ("-v", "--verbose"): { "action": "store_true", "help": "Enable detailed logging." },
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
        # If args_dict has --models, then make summary_args' --models required:
        if any("--models" in arg_names for arg_names in args_dict):
            summary_args[("-m", "--models")]["required"] = True
        args_dict.update(summary_args)
        included_modules.append("summary")

    if not conditional_args.no_annotation:
        args_dict.update(annotation_args)
        included_modules.append("annotation")

    if not conditional_args.no_filter:
        args_dict.update(filter_args)
        included_modules.append("filter")
        # Also include (a copy of) scoring args here (but without the required scoring output dir), as it'll be re-run for the filter step.
        scoring_args_copy = scoring_args.copy()
        scoring_args_copy.pop(("-scout", "--scoring-output-dir"))
        scoring_args_copy.pop(("--no-hdf5",))
        args_dict.update(scoring_args_copy)

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
    # args = parser.parse_args()
    args, unknown_args = parser.parse_known_args()

    # Provide information about unneeded and incorrect arguments.
    all_possible_args = set([arg for arg_names in args_dict for arg in arg_names])
    unneeded_args = []
    error_args = []
    for unknown_arg in unknown_args:
        if unknown_arg in all_possible_args:
            unneeded_args.append(unknown_arg)
        else:
            error_args.append(unknown_arg)
    for arg in unneeded_args:
        print(f"Warning: Argument '{arg}' is not needed for chosen modules ({included_modules_str}) and will be ignored.")
    for arg in error_args:
        print(f"Error: Argument '{arg}' is not recognized.")
    if len(error_args) > 0:
        exit(1)

    return args
