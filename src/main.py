from utils import argmanager
import variant_scoring
import variant_shap
import variant_summary_across_folds
import variant_annotation
import variant_filter
import variant_viz
import sys
import logging

import argparse
from os.path import isfile

class ClosestElementsAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if len(values) % 3 != 0:
            raise argparse.ArgumentError(self, f"Must provide 3 arguments per element type: <BED1> <COUNT1> <LABEL1> <BED2> <COUNT2> <LABEL2> ...; you provided {len(values)} arguments: {values}")
        
        closest_n_elements_args = []
        for i in range(0, len(values), 3):
            closest_elements_file = values[i]
            closest_elements_count = int(values[i + 1])
            closest_elements_label = values[i + 2]
            
            if not isfile(closest_elements_file):
                parser.error(f"TSV file '{closest_elements_file}' does not exist.")
            
            closest_n_elements_args.append((closest_elements_file, closest_elements_count, closest_elements_label))
            
        setattr(namespace, "closest_n_elements_args", closest_n_elements_args)
        setattr(namespace, self.dest, True)

class ClosestElementsInWindowAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        print(len(values))
        if len(values) % 3 != 0:
            raise argparse.ArgumentError(self, f"Must provide 3 arguments per element type: <BED1> <SIZE1> <LABEL1> <BED2> <SIZE2> <LABEL2> ...; you provided {len(values)} arguments: {values}")
        
        closest_n_elements_args = []
        for i in range(0, len(values), 3):
            closest_elements_window_file = values[i]
            closest_elements_window_size = int(values[i + 1])
            closest_elements_window_label = values[i + 2]
            
            if not isfile(closest_elements_window_file):
                parser.error(f"TSV file '{closest_elements_window_file}' does not exist.")
            
            closest_n_elements_args.append((closest_elements_window_file, closest_elements_window_size, closest_elements_window_label))
            
        setattr(namespace, "closest_elements_in_window_args", closest_n_elements_args)
        setattr(namespace, self.dest, True)

class AdastraAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, 'adastra_tf_file', values[0])
        setattr(namespace, 'adastra_celltype_file', values[1])
        setattr(namespace, self.dest, True)

class JoinTSVsAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        valid_directions = {'left', 'right', 'inner', 'outer'}
        join_args = []

        if len(values) % 3 != 0:
            raise argparse.ArgumentError(self, f"Must provide 3 arguments per TSV: <TS1> <LABEL1> <DIRECTION1> <TSV2> <LABEL2> <DIRECTION2> ...; you provided {len(values)} arguments: {values}")
        for i in range(0, len(values), 3):
            tsv_file = values[i]
            label = values[i + 1]
            direction = values[i + 2]
            
            if direction not in valid_directions:
                parser.error(f"Invalid direction '{direction}'. Valid directions are {valid_directions}.")
            
            if not isfile(tsv_file):
                parser.error(f"TSV file '{tsv_file}' does not exist.")
            
            join_args.append((tsv_file, label, direction))
        setattr(namespace, "join_args", join_args)
        setattr(namespace, self.dest, True)

subcommand_args = {
    "score": {
        "help": "Gather variant effect information (i.e. \"scores\") from running variants through ChromBPNet, a chromatin accessibility model.",
		"function": variant_scoring.main,
        "args": {
            ("-l", "--variant-list"): { "type": str, "help": "a TSV file containing a list of variants to score.", "required": True},
            ("-g", "--genome"): {"type": str, "help": "Genome fasta.", "required": True},
            ("-m", "--models"): {"type": str, "nargs": '+', "help": "ChromBPNet models to use for variant scoring, whose outputs will be labeled with numerical indexes beginning from 0 in the order they are provided.", "required": True},
            ("-scdir", "--scoring-dir"): {"type": str, "help": "The directory to store scoring files. Used in this way: <scoring-dir>/<model-name>/fold_<index>.variant_scores.tsv. Directory should already exist.", "required": True},
            ("-s", "--chrom-sizes"): {"type": str, "help": "Path to TSV file with chromosome sizes", "required": True},
            ("-mn", "--model-name"): {"type": str, "help": "The prefix that will be prepended to the filename like: <subcommand-dir>/<model-name>/fold_<index>.variant_scores.tsv", "required": True},
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
            ("-scf", "--score-filenames"): { "nargs": '+', "help": "A list of file names of variant score files that will be used to overwrite the otherwise generated index filenames, and will be used like so: <scoring-dir>/<score-filename>.{tsv,h5} for each file in the list. Generally only needed if --no-scoring is used."},
            ("-v", "--verbose"): { "action": "store_true", "help": "Enable detailed logging." },
        },
    },
	"summarize": {
        "help": "Summarize variant scores across folds.",
		"function": variant_summary_across_folds.main,
        "args": {
            ("-scdir", "--scoring-dir"): {"type": str, "help": "The directory to store scoring files. Used in this way: <scoring-dir>/<model-name>/fold_<index>.variant_scores.tsv. Directory should already exist.", "required": True},
            ("-fsdir", "--folds-summarization-dir"): { "type": str, "help": "The directory to store the summary file with average scores across folds. Used in this way: <folds-summarization-dir>/<model-name>/variant_scores.tsv. Directory should already exist.", "required": True},
            ("-mn", "--model-name"): {"type": str, "help": "The prefix to be prepended to the filename, like: <subcommand-dir>/<model-name>/fold_<index>.variant_scores.tsv.", "required": True},
            ("-sc", "--schema"): { "type": str, "choices": ['bed', 'plink', 'plink2', 'chrombpnet', 'original'], "default": 'chrombpnet', "help": "Format for the input variants list."},
            ("-scf", "--score-filenames"): { "nargs": '+', "help": "A list of file names of variant score files to be used to overwrite the otherwise generated index filenames, and will be used like so: <scoring-dir>/<score-filename>.{tsv,h5} for each file in the list. Generally only needed if --no-scoring is used."},
            ("-m", "--models"): {"type": str, "nargs": '+', "help": "ChromBPNet models to use for variant scoring, whose outputs will be labeled with numerical indexes beginning from 0 in the order they are provided."},
            ("-v", "--verbose"): { "action": "store_true", "help": "Enable detailed logging." },
        },
    },
    "annotate": {
        "help": "Generate useful annotations for each variant.",
		"function": variant_annotation.main,
        "args": {
            ("-fsdir", "--folds-summarization-dir"): { "type": str, "help": "The directory to store the summary file with average scores across folds. Used in this way: <folds-summarization-dir>/<model-name>/variant_scores.tsv. Directory should already exist.", "required": True},
            ("-mn", "--model-name"): {"type": str, "help": "The prefix to be prepended to the filename like: <subcommand-dir>/<model-name>/fold_<index>.variant_scores.tsv.", "required": True},
            ("-adir", "--annotation-dir"): { "type": str, "help": "The directory to store the unfiltered annotations file like so: <annotation-dir>/<model-name>/annotations.tsv. This directory should already exist.", "required": True},
            ("-sc", "--schema"): {"type": str, "choices": ['bed', 'plink', 'plink2', 'chrombpnet', 'original'], "default": 'chrombpnet', "help": "Format for the input variants list."},
            ("-p", "--peaks"): { "type": str, "help": "Adds overlapping peaks information. Bed file containing peak regions."},
            ("-ce", "--add-n-closest-elements"): { "nargs": '+', "metavar": ('BED1 COUNT1 LABEL1', 'BED2 COUNT2 LABEL2'), "action": ClosestElementsAction, "help": "Adds variant annotations for the N closest elements (such as genes) and their distances to the variant. Takes in (1) a bed file containing gene regions, (2) number of closest genes, and (3) output an optional label. For example, `--add-closest-elements hg38.genes.bed 5 gene hg38.lnRNA.bed 3 lnRNA` will output 5`closest_gene_i` and `closest_gene_i_distance` columns, as well as 3 `closest_lnRNA_i` and `closest_lnRNA_i_distance` columns.", "default": False },
            ("-cew", "--add-closest-elements-in-window"): { "nargs": '+', "metavar": ('BED1 SIZE1 LABEL1', 'BED2 SIZE2 LABEL2'), "action": ClosestElementsInWindowAction, "help": "Add variant annotations for the elements (such as genes) existing up to <WINDOWSIZE> bp away from the variant. Takes in (1) a bed file containing gene regions, (2) the window size, and (3) output an optional label. For example, `--add-closest-elements-in-window hg38.genes.bed 100000 genes` will output genes existing within 100000 bp of each variant to their `genes_within_100000_bp` column.", "default": False },
            ("-aa", "--add-adastra"): { "nargs": 2, "metavar": ('ADASTRA_TF_FILE', 'ADASTRA_CELLTYPE_FILE'), "action": AdastraAction, "help": "Annotate with ADASTRA. Provide ADASTRA TF and cell type data files.", "default": False },
            ("-j", "--join-tsvs"): { "nargs": '+', "metavar": ('TSV1 LABEL1 DIRECTION1', 'TSV2 LABEL2 DIRECTION2'), "action": JoinTSVsAction, "help": "Add external annotations by joining variant annotations with external TSVs on the specified labels and direction. Valid directions are 'left', 'right', 'inner', 'outer'. For example, `--join-tsvs 1.tsv variant_id left 2.tsv variant_id outer 3.tsv hpo left`.", "default": False},
            ("-th", "--threads"): { "type": int, "help": "The maximum amount of threads to use, where possible." },
            ("-r2", "--r2"): { "type": str, "help": "Adds r2 annotations. Requires a PLINK .ld file." },
            ("-v", "--verbose"): { "action": "store_true", "help": "Enable detailed logging." },
        },
    },
    "filter": {
        "help": "",
		"function": variant_filter.main,
        "args": {
            ("-adir", "--annotation-dir"): { "type": str, "help": "The directory to store the unfiltered annotations file like so: <annotation-dir>/<model-name>/annotations.tsv. This directory should already exist.", "required": True},
            ("-mn", "--model-name"): {"type": str, "help": "The prefix to be prepended to the filename like: <subcommand-dir>/<model-name>/fold_<index>.variant_scores.tsv.", "required": True},
            ("-fdir", "--filter-dir"): {"type": str, "help": "The directory to store the filtered annotations file like so: <filter-dir>/<model-name>/annotations.filtered.tsv. This directory should already exist.", "required": True},
            ("-lo", "--filter-lower"): {"type": str, "nargs": "+", "help": "Removes all variants from the annotations list containing a field(s) with a value greater than or equal to the specified threshold(s). This flag accepts a list of pairs, where each pair contains the field and value delimited by a colon. By default this flag is set to `-lo abs_logfc.mean.pval:0.01 jsd.mean.pval:0.01`", "default": ["abs_logfc.mean.pval:0.01", "jsd.mean.pval:0.01"]},
            ("-up", "--filter-upper"): {"type": str, "nargs": "+", "help": "Removes all variants from the annotations list containing a field(s) with a value lower than or equal to the specified threshold(s). This flag accepts a list of pairs, where each pair contains the field and value delimited by a colon. An example of this flag would be: `-up field1:0.5 field2:0.1`."},
            ("-fl", "--filter-logic"): {"type": str, "choices": ["and", "or"], "default": "and", "help": "The logic to use when filtering variants based on the filter-lower and filter-upper flags, excluding --max-percentile-threshold and --peak-variants-only, which always use 'and' logic. The default is 'and', which means that a variant will be removed if it fails any of the filter conditions. If set to 'or', a variant will be removed if it fails all of the filter conditions."},
            ("-mpt", "--max-percentile-threshold"): {"type": float, "help": "Removes all variants from the annotations list whose max_percentile.mean (generated from running the scoring step with the -p flag, followed by the summary step) is less than or equal to the specified threshold, and always uses 'and' threshold logic. The default is 0.05.", "default": 0.05},
            ("-p", "--peaks"): { "type": str, "help": "Adds overlapping peaks information. Bed file containing peak regions."},
            ("-po", "--peak-variants-only"): {"action": "store_true", "help": "Only keep variants whose peak_overlap (generated from the annotation step with the -p flag) is equal to True, and always uses 'and' threshold logic. Peaks file must be provided if used."},
            ("-v", "--verbose"): { "action": "store_true", "help": "Enable detailed logging." },
        }
    },
    "shap":
    { 
        "help": "",
		"function": variant_shap.main,
        "args": {
            ("-fdir", "--filter-dir"): {"type": str, "help": "The directory to store the filtered annotations file like so: <filter-dir>/<model-name>/annotations.filtered.tsv. This directory should already exist.", "required": True},
            ("-mn", "--model-name"): {"type": str, "help": "The prefix to be prepended to the filename like: <subcommand-dir>/<model-name>/fold_<index>.variant_scores.tsv.", "required": True},
            ("-shdir", "--shap-dir"): { "type": str, "help": "The directory that will store the SNP effect score predictions from the script. Used like: <shap-dir>/<model-name>/annotations.filtered.tsv. This directory should already exist.", "required": True},
            ("-g", "--genome"): { "type": str, "help": "Genome fasta." , "required": True},
            ("-m", "--models"): {"type": str, "nargs": '+', "help": "ChromBPNet models to use for variant scoring, whose outputs will be labeled with numerical indexes beginning from 0 in the order they are provided.", "required": True},
            ("-s", "--chrom-sizes"): {"type": str, "help": "Path to TSV file with chromosome sizes.", "required": True},
            ("-li", "--lite"): { "action": "store_true", "help": "Models were trained with chrombpnet-lite."},
            ("-dm", "--debug-mode"): { "action": "store_true", "help": "Display allele input sequences."},
            ("-bs", "--batch-size"): { "type": int, "default": 10000, "help": "Batch size to use for the model."},
            ("-c", "--chrom"): { "type": str, "help": "Only score SNPs in selected chromosome."},
            ("-shf", "--shap-filenames"): { "nargs": '+', "help": "A list of file names of shap files to be used to overwrite the otherwise generated index filenames, and will be used like so: <shap-dir>/<shap-filename>.{h5,bw} for each file in the list."},
            ("-st", "--shap-type"): { "nargs": '+', "default": ["counts"], "help": "Specify shap value type(s) to calculate." },
            ("--no-hdf5",): {"action": "store_true", "help": "Prevents saving detailed predictions in hdf5 file during storing, and using those files during the shap and viz steps. Recommended when the variants list is large (>1,000,000)."},
            ("-v", "--verbose"): { "action": "store_true", "help": "Enable detailed logging." },
        },
    },
    "viz": {
        "help": "",
		"function": variant_viz.main,
        "args": {
            ("-l", "--variant-list"): { "type": str, "help": "a TSV file containing a list of variants to score.", "required": True},
            ("-m", "--models"): {"type": str, "nargs": '+', "help": "ChromBPNet models to use for variant scoring, whose outputs will be labeled with numerical indexes beginning from 0 in the order they are provided.", "required": True},
            ("-mn", "--model-name"): {"type": str, "help": "The prefix to be prepended to the filename like: <subcommand-dir>/<model-name>/fold_<index>.variant_scores.tsv.", "required": True},
            ("-fdir", "--filter-dir"): {"type": str, "help": "The directory to store the filtered annotations file like so: <filter-dir>/<model-name>/annotations.filtered.tsv. This directory should already exist.", "required": True},
            ("-shdir", "--shap-dir"): { "type": str, "help": "The directory that will store the SNP effect score predictions from the script. Used like: <shap-dir>/<model-name>/annotations.filtered.tsv. This directory should already exist.", "required": True},
            ("-vdir", "--viz-dir"): {"type": str, "help": "The directory that will store the visualization outputs. This directory should already exist.", "required": True},
            ("-fscf", "--filter-score-filenames"): { "nargs": '+', "help": "A list of file names of filtered variant score files that will be used to overwrite the otherwise generated index filenames, and will be used like so: <filter-dir>/<filtered-score-filename>.{tsv,h5} for each file in the list. Generally only needed if --no-scoring is used."},
            ("-st", "--shap-type"): { "nargs": '+', "default": ["counts"], "help": "Specify shap value type(s) to calculate." },
            ("-predo", "--predictions-override"): { "type": str, "help": "The name of the variant effect predictions' file. Use this if you want to use a name other than the default."},
            ("-shapo", "--shap-override"): { "type": str, "help": "The name of the shap output file. Use this if you want to use a name other than the default."},
            ("-v", "--verbose"): { "action": "store_true", "help": "Enable detailed logging." },
        }
    }
}


def get_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='subcommand', help='Subcommands for different steps:')

    for subcommand, subcommand_dict in subcommand_args.items():
        args_dict = subcommand_dict['args']
        subparser = subparsers.add_parser(subcommand, help=subcommand_dict['help'])
        for arg_names, kwargs in args_dict.items():
            subparser.add_argument(*arg_names, **kwargs)

    # Parse and handle errors
    try:
        args, _ = parser.parse_known_args()
        if not vars(args) or args.subcommand is None:
            parser.print_help()
            parser.exit()
    except argparse.ArgumentError:
        parser.print_help()
        parser.exit()

    # Parse the subcommand arguments
    try:
        subparser = subparsers.choices[args.subcommand]
        args = subparser.parse_args()
    except argparse.ArgumentError:
        subparser.print_help()
        subparser.exit()

    return args


def cli():
	args = get_args()
	logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
						format='%(asctime)s - %(levelname)s - %(message)s')
	subcommand_args[args.subcommand]['function'](args)
