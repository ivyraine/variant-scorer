import pandas as pd
import numpy as np
import os
from utils.argmanager import *
from utils.helpers import *
import logging

def main(args = None):
    score_dict = {}

    for file_index, variant_score_file in enumerate(args.scores_paths):
        assert os.path.isfile(variant_score_file), f"Error: The file '{variant_score_file}' does not exist or is not a file."
        var_score = pd.read_table(variant_score_file)
        score_dict[file_index] = var_score

    os.makedirs(os.path.dirname(args.summarize_output_path), exist_ok=True)

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

    variant_scores.to_csv(args.summarize_output_path,\
                          sep="\t",\
                          index=False)

    logging.info(f"Summary step completed! Output written to: {args.summarize_output_path}")


if __name__ == "__main__":
    main()
