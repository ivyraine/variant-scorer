import pandas as pd
import numpy as np
import os
from utils.argmanager import *
from utils.helpers import *

def main(args = None):
    if args is None:
        args = fetch_summary_args()
        logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')
    print(args)

    variant_scores_files = []
    if args.score_filenames is not None and 0 < len(args.score_filenames):
        variant_scores_files = [ os.path.join(args.scoring_output_dir, args.score_filenames[i]) for i in range(len(args.score_filenames)) ]
    elif hasattr(args, "models"):
        variant_scores_files = [ f"{get_score_output_file_prefix(args.scoring_output_dir, args.sample_name, index)}.variant_scores.tsv" for index in range(len(args.models)) ]
    else:
        print("Error: No models or score list provided. Cannot perform summary. Exiting.")
        exit(1)

    score_dict = {}
    for file_index, variant_score_file in enumerate(variant_scores_files):
        assert os.path.isfile(variant_score_file), f"Error: The file '{variant_score_file}' does not exist or is not a file."
        var_score = pd.read_table(variant_score_file)
        score_dict[file_index] = var_score

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

    print()
    print(variant_scores.head())
    print("Summary score table shape:", variant_scores.shape)
    print()

    out_file = f"{get_summary_output_file(args.summary_output_dir, args.sample_name)}"
    variant_scores.to_csv(out_file,\
                          sep="\t",\
                          index=False)

    logging.info(f"Summary step completed! Output written to: {out_file}")


if __name__ == "__main__":
    main()
