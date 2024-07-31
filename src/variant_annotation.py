import pandas as pd
import numpy as np
import os
import subprocess
import logging

from utils.argmanager import *
from utils.helpers import *
pd.set_option('display.max_columns', 20)

def main(args = None):

    if args is None:
        logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')

        args = fetch_annotation_args()

    variant_scores_file = get_summary_output_file(args.summary_dir, args.model_name)
    peak_path = args.peaks

    variant_scores = pd.read_table(variant_scores_file)
    # TODO use os temp instead.
    # tmp_bed_file_path = f"/tmp/{args.model_name}.variant_table.tmp.bed"

    variant_scores_bed_format = None
    if args.schema == "bed":
        if variant_scores['pos'].equals(variant_scores['end']):
            variant_scores['pos'] = variant_scores['pos'] - 1
        variant_scores_bed_format = variant_scores[['chr','pos','end','allele1','allele2','variant_id']].copy()
        variant_scores_bed_format.sort_values(by=["chr","pos","end"], inplace=True)
    else:
        variant_scores_bed_format = variant_scores[['chr','pos','allele1','allele2','variant_id']].copy()
        variant_scores_bed_format['pos']  = variant_scores_bed_format.apply(lambda x: int(x.pos)-1, axis = 1)
        variant_scores_bed_format['end']  = variant_scores_bed_format.apply(lambda x: int(x.pos)+len(x.allele1), axis = 1)
        variant_scores_bed_format = variant_scores_bed_format[['chr','pos','end','allele1','allele2','variant_id']]
        variant_scores_bed_format.sort_values(by=["chr","pos","end"], inplace=True)

    print(args)
    if args.join_tsvs:
        for tsv, label, direction in reversed(args.join_args):
            join_df = pd.read_csv(tsv, sep='\t')
            variant_scores = variant_scores.merge(join_df, how=direction, on=label)

    if args.peaks:
        logging.info("Annotating with peak overlap")
        peak_df = pd.read_table(peak_path, header=None)
        variant_bed = pybedtools.BedTool.from_dataframe(variant_scores_bed_format)
        peak_bed = pybedtools.BedTool.from_dataframe(peak_df)
        peak_intersect_bed = variant_bed.intersect(peak_bed, wa=True, u=True)

        peak_intersect_df = peak_intersect_bed.to_dataframe(names=variant_scores_bed_format.columns.tolist())

        logging.debug(f"Peak overlap table:\n{peak_intersect_df.shape}\n{peak_intersect_df.head()}")

        variant_scores['peak_overlap'] = variant_scores['variant_id'].isin(peak_intersect_df['variant_id'].tolist())

    if args.add_n_closest_elements:
        add_n_closest_elements_inplace(variant_scores, args.add_n_closest_elements, variant_scores_bed_format)

    if args.add_closest_elements_in_window:
        add_closest_elements_in_window_inplace(variant_scores, args.add_n_closest_elements, variant_scores_bed_format)

    if args.r2:
        add_r2_inplace(variant_scores, args.r2)
            
    if args.add_adastra:
        add_adastra_inplace(variant_scores, args.adastra_tf_file, args.adastra_celltype_file, args.threads)

    logging.info(f"Final annotation table:\n{variant_scores.shape}\n{variant_scores.head()}")

    out_file = get_annotation_output_file(args.annotate_dir, args.model_name)
    variant_scores.to_csv(out_file,\
                          sep="\t",\
                          index=False)

    logging.info(f"Annotation step completed! Output written to: {out_file}")


if __name__ == "__main__":
    main()
