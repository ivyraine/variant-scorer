from utils import argmanager
import variant_scoring
# import variant_scoring_per_chrom
# import variant_scoring_per_chunk
import variant_shap
import variant_summary_across_folds
import variant_annotation
import variant_filter
import os
import sys
import logging


def cli():
	args = argmanager.fetch_main_parser()
	logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
						format='%(asctime)s - %(levelname)s - %(message)s')

	if not args.no_filter and not args.no_scoring and not args.no_summary and not args.no_annotation and args.peak_variants_only is True and args.peaks is None:
		# Require peaks for peak_variants_only
		print("Error: The --peaks argument is required when --peak-variants-only is True.")
		sys.exit(1)

	# Check output dirs ahead of time.
	if not args.no_scoring:
		if not os.path.isdir(args.scoring_output_dir):
			print(f"Error: The --scoring-output-dir directory '{args.scoring_output_dir}' does not exist.")
			exit(1)
	if not args.no_summary:
		if not os.path.isdir(args.summary_output_dir):
			print(f"Error: The --summary-output-dir directory '{args.summary_output_dir}' does not exist.")
			exit(1)
	if not args.no_annotation:
		if not os.path.isdir(args.annotation_output_dir):
			print(f"Error: The --annotation-output-dir directory '{args.annotation_output_dir}' does not exist.")
			exit(1)
	if not args.no_filter:
		if not os.path.isdir(args.filter_output_dir):
			print(f"Error: The --filter-output-dir directory '{args.filter_output_dir}' does not exist.")
			exit(1)
	if not args.no_shap:
		if not os.path.isdir(args.shap_output_dir):
			print(f"Error: The --shap-output-dir directory '{args.shap_output_dir}' does not exist.")
			exit(1)

	if not args.no_scoring:
		variant_scoring.main(args)
	if not args.no_summary:
		variant_summary_across_folds.main(args)
	if not args.no_annotation:
		variant_annotation.main(args)
	if not args.no_filter:
		variant_filter.main(args)
	if not args.no_shap:
		variant_shap.main(args)


if __name__ == "__main__":
	cli()
