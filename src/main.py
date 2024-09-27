from utils import argmanager
import variant_scoring
import variant_shap
import variant_summary_across_folds
import variant_annotation
import variant_viz
import aggregate
import sys
import logging

import argparse
from os.path import isfile
from utils.helpers import MODEL_ID_COL, ANNOTATE_OUT_PATH_COL, AGGREGATE_OUT_PATH_COL

class ClosestElementsAction(argparse.Action):
	def __call__(self, parser, namespace, values, option_string=None):
		# Ensure the attribute exists and is a list
		if not hasattr(namespace, 'closest_n_elements_args'):
			setattr(namespace, 'closest_n_elements_args', [])
		
		closest_elements_file, closest_elements_count, closest_elements_label = values
		
		if not isfile(closest_elements_file):
			parser.error(f"BED file '{closest_elements_file}' does not exist.")
		
		# Append the new values as a tuple
		getattr(namespace, 'closest_n_elements_args').append(
			(closest_elements_file, int(closest_elements_count), closest_elements_label)
		)
		setattr(namespace, self.dest, True)

class ClosestElementsInWindowAction(argparse.Action):
	def __call__(self, parser, namespace, values, option_string=None):
		# Ensure the attribute exists and is a list
		if not hasattr(namespace, 'closest_elements_in_window_args'):
			setattr(namespace, 'closest_elements_in_window_args', [])
		
		closest_elements_window_file, closest_elements_window_size, closest_elements_window_label = values
		
		if not isfile(closest_elements_window_file):
			parser.error(f"BED file '{closest_elements_window_file}' does not exist.")
		
		# Append the new values as a tuple
		getattr(namespace, 'closest_elements_in_window_args').append(
			(closest_elements_window_file, int(closest_elements_window_size), closest_elements_window_label)
		)
		setattr(namespace, self.dest, True)

class JoinTSVsAction(argparse.Action):
	def __call__(self, parser, namespace, values, option_string=None):
		valid_directions = {'left', 'right', 'inner', 'outer'}
		
		# Ensure the attribute exists and is a list
		if not hasattr(namespace, 'join_args'):
			setattr(namespace, 'join_args', [])
		
		tsv_file, label, direction = values
		
		if direction not in valid_directions:
			parser.error(f"Invalid direction '{direction}'. Valid directions are {valid_directions}.")
		
		if not isfile(tsv_file):
			parser.error(f"TSV file '{tsv_file}' does not exist.")
		
		# Append the new values as a tuple
		getattr(namespace, 'join_args').append((tsv_file, label, direction))
		setattr(namespace, self.dest, True)

class AdastraAction(argparse.Action):
	def __call__(self, parser, namespace, values, option_string=None):
		setattr(namespace, 'adastra_tf_file', values[0])
		setattr(namespace, 'adastra_celltype_file', values[1])
		setattr(namespace, self.dest, True)

def parse_default_value(val):
	# Check if the value can be converted to an int
	if val.isdigit():
		return int(val)
	# Check if the value can be converted to a float
	try:
		float_val = float(val)
		return float_val
	except ValueError:
		pass
	# Check if the value can be converted to a boolean
	if val in ['True', 'False']:
		return val.lower() == 'true'
	# Return the value as a string if no other conversion applies
	return val

shared_annotate_args = {
	# TODO rename this since this is not just for bool. also rename to like: add-label-using-pandas
	("-p", "--peaks"): { "type": str, "help": "Adds overlapping peaks information. Bed file containing peak regions."},
	("-po", "--peak-variants-only"): {"action": "store_true", "help": "Only keep variants whose peak_overlap is equal to True, and always uses 'and' threshold logic. Peaks file must be provided if used."},
	("-pd", "--add-annot-using-pandas"): { "nargs": 2, "metavar": ('LABEL', 'EXPRESSION'), "action": "append", "help": '''Generate a new annotation with values equal to the evaluation of a Pandas expression using `pandas.DataFrame.eval(EXPRESSION)`. For example, `--add-annot-using-pandas is_significant "logfc.mean.pval < 0.01) | jsd.mean.pval < 0.01 & active_allele_quantile.mean > 0.05"`. Can be called multiple times to generate multiple annotations.''' },
	("-py", "--add-annot-using-python"): { "nargs": 2, "metavar": ('LABEL', 'EXPRESSION'), "action": "append", "help": '''Generate a new annotation with values equal to the evaluation of a Python expression using `eval(EXPRESSION)`. For example, `--add-annot-using-python is_significant "((df['logfc.mean.pval'] < 0.01 | (df['jsd.mean.pval'] < 0.01)) & (df['active_allele_quantile.mean'] > 0.05)"`. Can be called multiple times to generate multiple annotations.''' },
	("-sc", "--schema"): {"type": str, "choices": ['bed', 'plink', 'plink2', 'chrombpnet', 'original'], "default": 'chrombpnet', "help": "Format for the input variants list."},
	("-p", "--peaks"): { "type": str, "help": "Adds overlapping peaks information. Bed file containing peak regions."},
	("-ce", "--add-n-closest-elements"): { "nargs": 3, "metavar": ('BED', 'COUNT', 'LABEL'), "action": ClosestElementsAction, "help": "Adds variant annotations for the N closest elements (such as genes) and their distances to the variant. Takes in (1) a bed file containing gene regions, (2) number of closest genes, and (3) output an optional label. Can be used multiple times to generate multiple annotations. For example, `--add-n-closest-elements hg38.genes.bed 5 gene --add-n-closest-elements hg38.lnRNA.bed 3 lnRNA` will output 5`closest_gene_i` and `closest_gene_i_distance` columns, as well as 3 `closest_lnRNA_i` and `closest_lnRNA_i_distance` columns.", "default": False },
	("-cew", "--add-closest-elements-in-window"): { "nargs": 3, "metavar": ('BED', 'WINDOW_SIZE', 'LABEL'), "action": ClosestElementsInWindowAction, "help": "Add variant annotations for the elements (such as genes) existing up to <WINDOWSIZE> bp away from the variant. Takes in (1) a bed file containing gene regions, (2) the window size, and (3) output an optional label. Can be used multiple times to generate multiple annotations. For example, `--add-closest-elements-in-window hg38.genes.bed 100000 genes` will output genes existing within 100000 bp of each variant to their `genes_within_100000_bp` column.", "default": False },
	("-aa", "--add-adastra"): { "nargs": 2, "metavar": ('ADASTRA_TF_FILE', 'ADASTRA_CELLTYPE_FILE'), "action": AdastraAction, "help": "Annotate with ADASTRA. Provide ADASTRA TF and cell type data files.", "default": False },
	("-j", "--join-tsvs"): { "nargs": 3, "metavar": ('TSV', 'LABEL', 'DIRECTION'), "action": JoinTSVsAction, "help": "Add external annotations by joining variant annotations with external TSVs on the specified labels and direction. Valid directions are 'left', 'right', 'inner', 'outer'. Can be used multiple times to generate multiple annotations. For example, `--join-tsvs 1.tsv variant_id left --join-tsvs 2.tsv variant_id outer --join-tsvs 3.tsv hpo left`.", "default": False},
	("-th", "--threads"): { "type": int, "help": "The maximum amount of threads to use, where possible." },
	("-r2", "--r2"): { "type": str, "help": "Adds r2 annotations. Requires a PLINK .ld file." },
	("-v", "--verbose"): { "action": "store_true", "help": "Enable detailed logging." },
}
subcommand_args = {
	"score": {
		"help": "Gather variant effect information (i.e. \"scores\") from running variants through ChromBPNet, a chromatin accessibility model.",
		"function": variant_scoring.main,
		"args": {
			("-l", "--variant-list"): { "type": str, "help": "a TSV file containing a list of variants to score.", "required": True},
			("-g", "--genome"): {"type": str, "help": "Genome fasta.", "required": True},
			("-m", "--model-path"): {"type": str, "help": "ChromBPNet model to use for variant scoring, whose outputs will be labeled with numerical indexes beginning from 0 in the order they are provided.", "required": True},
			("-o", "--score-output-path-prefix"): {"type": str, "help": 'A string that will be prefixed to the outputs of the `score` subcommand, to form a valid path to be written to. Should contain information relevant to the project, subcommand, model, and fold, if relevant. Used in this way: "<output-path-prefix><output-file-suffix>". Example usage: `--score-output-path-prefix /projects/score/adipocytes/fold_0/` will output to paths like /projects/score/adipocytes/fold_0/variant_scores.tsv.', "required": True},
			("-s", "--chrom-sizes"): {"type": str, "help": "Path to TSV file with chromosome sizes", "required": True},
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
			("-v", "--verbose"): { "action": "store_true", "help": "Enable detailed logging." },
		},
	},
	"summarize": {
		"help": "Summarize variant scores across folds.",
		"function": variant_summary_across_folds.main,
		"args": {
			("-i", "--score-output-paths"): { "nargs": '+', "help": "A (space-separated) list of variant score file paths (generally, each from a different fold) to be summarized together, generated from the `score` subcommand. Used like so: `--score-output-paths /projects/score/adipocytes/fold_0/variant_scores.tsv /projects/score/adipocytes/fold_1/variant_scores.tsv /projects/score/adipocytes/fold_2/variant_scores.tsv ...`."},
			("-o", "--summarize-output-path"): { "type": str, "help": 'A string representing the output file path of the `summarize` subcommand. Should contain information relevant to the project, subcommand, and model. Example usage: `--summarize-output-path /projects/summarize/adipocytes/mean.variant_scores.tsv` will output to /projects/summarize/adipocytes/mean.variant_scores.tsv.'},
			("-sc", "--schema"): { "type": str, "choices": ['bed', 'plink', 'plink2', 'chrombpnet', 'original'], "default": 'chrombpnet', "help": "Format for the input variants list."},
			("-v", "--verbose"): { "action": "store_true", "help": "Enable detailed logging." },
		},
	},
	"annotate": {
		"help": "Generate useful annotations for each variant. Same capabilities as the `annotate-prio` subcommand, but is named differently for clarity of usage; in particular, you should aim to only generate annotations that you will use in deciding variant prioritization, and generate other annotations in the `annotate-prio` step. It's recommended to keep these annotations minimal for large-scale projects to save on storage space and processing time.",
		"function": variant_annotation.main,
		"args": {**{
			("-i", "--summarize-output-path"): { "type": str, "help": 'A string representing the output file path of the `summarize` subcommand. Should contain information relevant to the project, subcommand, and model. Example usage: `--summarize-output-path /projects/summarize/adipocytes/mean.variant_scores.tsv` will output to /projects/summarize/adipocytes/mean.variant_scores.tsv.'},
			("-o", "--annotate-output-path"): { "type": str, "help": 'A string representing the output file path of the `annotate` subcommand. Should contain information relevant to the project, subcommand, and model. Example usage: `--annotate-output-path /projects/annotate/adipocytes/annotated.mean.variant_scores.tsv` will output to /projects/summarize/adipocytes/annotated.mean.variant_scores.tsv.'},
			# TODO add a retain-only flag, to keep specified columns only
			# ("-lo", "--filter-lower"): {"type": str, "nargs": "+", "help": "Removes all variants from the annotations list containing a field(s) with a value greater than or equal to the specified threshold(s). This flag accepts a list of pairs, where each pair contains the field and value delimited by a colon. By default this flag is set to `-lo abs_logfc.mean.pval:0.01 jsd.mean.pval:0.01`", "default": ["abs_logfc.mean.pval:0.01", "jsd.mean.pval:0.01"]},
			# ("-up", "--filter-upper"): {"type": str, "nargs": "+", "help": "Removes all variants from the annotations list containing a field(s) with a value lower than or equal to the specified threshold(s). This flag accepts a list of pairs, where each pair contains the field and value delimited by a colon. An example of this flag would be: `-up field1:0.5 field2:0.1`."},
			# ("-fl", "--filter-logic"): {"type": str, "choices": ["and", "or"], "default": "and", "help": "The logic to use when filtering variants based on the filter-lower and filter-upper flags, excluding --max-percentile-threshold and --peak-variants-only, which always use 'and' logic. The default is 'and', which means that a variant will be removed if it fails any of the filter conditions. If set to 'or', a variant will be removed if it fails all of the filter conditions."},
			# ("-mpt", "--max-percentile-threshold"): {"type": float, "help": "Removes all variants from the annotations list whose max_percentile.mean (generated from running the scoring step with the -p flag, followed by the summary step) is less than or equal to the specified threshold, and always uses 'and' threshold logic. The default is 0.05.", "default": 0.05},
		}, **shared_annotate_args},
	},
	"aggregate": {
		"help": "Generate aggregate variant annotations from the per-model annotations generated with `score`, `summarize`, and `annotate`.",
		"function": aggregate.main,
		"args": {
			("-i", "--annotate-output-paths"): { "nargs": '+', "type": str, "help": 'A (space-separated) list of strings representing the output file path of the `annotate` subcommand. Example usage: `--annotate-output-paths /projects/annotate/adipocytes/annotated.mean.variant_scores.tsv /projects/annotate/cardiomyocytes/annotated.mean.variant_scores.tsv ...`. It is recommended to use --input-metadata instead of this flag.'},
			("-im", "--input-metadata"): {"type": str, "help": f'A TSV file containing three required columns for this subcommand: "{MODEL_ID_COL}", "{ANNOTATE_OUT_PATH_COL}", and "{AGGREGATE_OUT_PATH_COL}".'},
			("-o", "--aggregate-output-path"): { "type": str, "help": 'A string representing the output file path of the `aggregate` subcommand. Example usage: `--aggregate-output-path /projects/aggregate/adipocytes/aggregate.variant_scores.tsv`. It is recommended to use --input-metadata instead of this flag.'},
			# ("-pa", "--add-aggregate-annot-with-pandas"): { "nargs": 3, "metavar": ("LABEL", "EXPRESSION", "DEFAULT_VALUE_EXPRESSION"), "action": "append", "help": '''Generate a new annotation equal to the evaluation of a Pandas expression using `pandas.DataFrame.eval(EXPRESSION)`, applied on the resulting aggregate variant annotations. The per-model DataFrame is exposed as `cur`. For example, `-pa sum "sum + @cur_df['sum']" 0` generates a new column named "is_significant" with values True or False.`'''},
			("-py", "--add-aggregate-annot-with-python"): { "nargs": 3, "metavar": ("LABEL", "EXPRESSION", "DEFAULT_VALUE_EXPRESSION"), "action": "append", "help": '''Generate a new annotation equal to the evaluation of a Python expression using `eval(EXPRESSION)`, applied on the resulting aggregate variant annotations. The aggregate DataFrame is exposed as `df` and the per-model DataFrame as `cur_df`. For example, `-py sum "df['sum'] + cur_df['sum']" 0` generates a new column named "is_significant" with values True or False.''' },
			("-sort", "--sort-together"): { "nargs": '+', "type": str, "help": '''Sorts multiple columns together, where each column represents a list of elements. These are inputed as strings describing the columns in the order they're given, delimiter, sort_order (optional, defaults to "asc"), and data type (optional, defaults as "str"). Groups of columns to sort together are provided as separate expressions, such as `--sort-together "col1:, :asc:str|col2:; :desc:int" "col3:,:asc:float|col4:;::str"`. Information describing each column and their delimiter and sort order are separated by ':'. Each of of these column specifications are separated by '|'.''' },
			("-id", "--add-temp-model-id"): {'action': 'store_true', "help": f'Temporarily adds a single-value "{MODEL_ID_COL} column to each TSV iterated through, so that the model ID may be used in annotations.".'},
			# ("-id", "--add-temp-model-id"): {"type": bool, "help": f'Temporarily adds a single-value "{MODEL_ID_COL} column to each TSV iterated through, so that the model ID may be used in annotations.".'},
			# LABEL CONDITION (applied to model-specific to get annotations) AGGREGATION_OPERATION (i.e. sum, concat strings. unsure how to do this. prolly with pandas expression again, like df['col1'] + df['col2'])
			# except sometimes you don't even want condition. Instead, you can have one expression, like aggr_df['new_label'] = aggr_df.get('new_label'] + model_df['new_label']
			("-v", "--verbose"): { "action": "store_true", "help": "Enable detailed logging." },
		}
	},
	"annotate-prio": {
		"help": "Generate useful annotations for each variant.",
		"function": variant_annotation.main,
		"args": {**{
			# ("-i", "--prioritize-output-path"): { "type": str, "help": 'A string representing the output file path of the `prioritize` subcommand. Should contain information relevant to the project and subcommand. Example usage: `--summarize-output-path /projects/prioritize/adipocytes/mean.variant_scores.tsv` will output to /projects/summarize/adipocytes/mean.variant_scores.tsv.'},
			# ("-o", "--annotate-prio-output-path"): { "type": str, "help": 'A string representing the output file path of the `annotate` subcommand. Should contain information relevant to the project, subcommand, and model. Example usage: `--annotate-output-path /projects/annotate/adipocytes/mean.variant_scores.tsv` will output to /projects/summarize/adipocytes/mean.variant_scores.tsv.'},
		}, **shared_annotate_args},
	},
	"shap":
	{ 
		"help": "",
		"function": variant_shap.main,
		"args": {
			("-fdir", "--filter-dir"): {"type": str, "help": "The directory to store the filtered annotations file like so: <filter-dir>/<model-name>/annotations.filtered.tsv. This directory should already exist.", "required": True},
			("-mn", "--model-name"): {"type": str, "help": "The prefix to be prepended to the filename like: <subcommand-dir>/<model-name>/fold_<fold>/variant_scores.tsv.", "required": True},
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
			("-mn", "--model-name"): {"type": str, "help": "The prefix to be prepended to the filename like: <subcommand-dir>/<model-name>/fold_<fold>/variant_scores.tsv.", "required": True},
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


	# subparser = subparsers.choices[args.subcommand]
	# try:
	# 	args = subparser.parse_args()
	# except argparse.ArgumentError:
	# 	subparser.print_help()
	# 	subparser.exit()

	return args


def cli():
	args = get_args()
	logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
						format='%(asctime)s - %(levelname)s - %(message)s')
	print(args)
	print(args.subcommand)
	subcommand_args[args.subcommand]['function'](args)
