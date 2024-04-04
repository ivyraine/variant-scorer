from utils import argmanager
import variant_scoring
# import variant_scoring_per_chrom
# import variant_scoring_per_chunk
import variant_shap
import variant_summary_across_folds
import variant_annotation


def cli():
	args = argmanager.fetch_main_parser()
	if not args.no_scoring:
		variant_scoring.main(args)
	if not args.no_summary:
		variant_summary_across_folds.main(args)
	if not args.no_annotation:
		variant_annotation.main(args)
	if not args.no_shap:
		variant_shap.main(args)


if __name__ == "__main__":
	cli()
