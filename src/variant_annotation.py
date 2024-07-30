import pandas as pd
import numpy as np
import os
import subprocess
import logging
from multiprocessing import Pool
import pybedtools

from utils.argmanager import *
from utils.helpers import *
pd.set_option('display.max_columns', 20)

DEFAULT_THREADS = 4

def get_asb_adastra(chunk, sig_adastra_tf, sig_adastra_celltype):
    mean_asb_es_tf_ref = []
    mean_asb_es_tf_alt = []
    asb_tfs = []

    mean_asb_es_celltype_ref = []
    mean_asb_es_celltype_alt = []
    asb_celltypes = []

    for index,row in chunk.iterrows():

        local_tf_df = sig_adastra_tf.loc[sig_adastra_tf['variant_id'] == row['variant_id']].copy()
        if len(local_tf_df) > 0:
            mean_asb_es_tf_ref.append(local_tf_df['es_mean_ref'].mean())
            mean_asb_es_tf_alt.append(local_tf_df['es_mean_alt'].mean())
            asb_tfs.append(', '.join(local_tf_df['tf'].unique().tolist()))
        else:
            mean_asb_es_tf_ref.append(np.nan)
            mean_asb_es_tf_alt.append(np.nan)
            asb_tfs.append(np.nan)

        local_celltype_df = sig_adastra_celltype.loc[sig_adastra_celltype['variant_id'] == row['variant_id']].copy()
        if len(local_celltype_df) > 0:
            mean_asb_es_celltype_ref.append(local_celltype_df['es_mean_ref'].mean())
            mean_asb_es_celltype_alt.append(local_celltype_df['es_mean_alt'].mean())
            asb_celltypes.append(', '.join(local_celltype_df['celltype'].unique().tolist()))
        else:
            mean_asb_es_celltype_ref.append(np.nan)
            mean_asb_es_celltype_alt.append(np.nan)
            asb_celltypes.append(np.nan)
            
    chunk['adastra_asb_tfs'] = asb_tfs
    chunk['adastra_mean_asb_effect_size_tf_ref'] = mean_asb_es_tf_ref
    chunk['adastra_mean_asb_effect_size_tf_alt'] = mean_asb_es_tf_alt
    chunk['adastra_asb_celltypes'] = asb_celltypes
    chunk['adastra_mean_asb_effect_size_celltype_ref'] = mean_asb_es_celltype_ref
    chunk['adastra_mean_asb_effect_size_celltype_alt'] = mean_asb_es_celltype_alt
    
    return chunk

def main(args = None):

    if args is None:
        logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')

        args = fetch_annotation_args()

    variant_scores_file = get_summary_output_file(args.summary_output_dir, args.sample_name)
    peak_path = args.peaks

    variant_scores = pd.read_table(variant_scores_file)
    # TODO use os temp instead.
    # tmp_bed_file_path = f"/tmp/{args.sample_name}.variant_table.tmp.bed"

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

    variant_bed = pybedtools.BedTool.from_dataframe(variant_scores_bed_format)

    print(args)
    if args.join_tsvs:
        for tsv, label, direction in reversed(args.join_args):
            join_df = pd.read_csv(tsv, sep='\t')
            variant_scores = variant_scores.merge(join_df, how=direction, on=label)

    if args.add_n_closest_elements:
        for elements_file, n_elements, element_label in args.closest_n_elements_args:
            logging.info(f"Annotating with closest {n_elements} elements")
            element_df = pd.read_table(elements_file, header=None)
            print(element_df.head())
            element_bed = pybedtools.BedTool.from_dataframe(element_df)
            closest_elements_bed = variant_bed.closest(element_bed, d=True, t='first', k=n_elements)

            closest_element_df = closest_elements_bed.to_dataframe(header=None)
            if not closest_element_df.empty:
                logging.debug(f"Closest elements ({element_label}) table:\n{closest_element_df.shape}\n{closest_element_df.head()}")

                closest_elements = {}
                element_dists = {}

                for index, row in closest_element_df.iterrows():
                    if not row[5] in closest_elements:
                        closest_elements[row[5]] = []
                        element_dists[row[5]] = []
                    closest_elements[row[5]].append(row.iloc[9])
                    element_dists[row[5]].append(row.iloc[-1])

                closest_element_df = closest_element_df.rename({5: 'variant_id'}, axis=1)
                closest_element_df = closest_element_df[['variant_id']]

                for i in range(n_elements):
                    closest_element_df[f'closest_{element_label}_{i+1}'] = closest_element_df['variant_id'].apply(lambda x: closest_elements[x][i] if len(closest_elements[x]) > i else '.')
                    closest_element_df[f'closest_{element_label}_{i+1}_distance'] = closest_element_df['variant_id'].apply(lambda x: element_dists[x][i] if len(closest_elements[x]) > i else '.')

                closest_element_df.drop_duplicates(inplace=True)
            else:  
                # Make empty columns if no elements are found.
                closest_element_df = pd.DataFrame(columns=['variant_id'])
                for i in range(n_elements):
                    closest_element_df[f'closest_{element_label}_{i+1}'] = ''
                    closest_element_df[f'closest_{element_label}_{i+1}_distance'] = ''
            variant_scores = variant_scores.merge(closest_element_df, on='variant_id', how='left')

    if args.add_closest_elements_in_window:

        for elements_file, window_size, element_label in args.closest_elements_in_window_args:
            print(elements_file, window_size, element_label)
            logging.info("Annotating with closest elements within window size")
            element_df = pd.read_table(elements_file, header=None)
            element_bed = pybedtools.BedTool.from_dataframe(element_df)
            closest_elements_bed = variant_bed.window(element_bed, w=window_size)
            print("Closest elements bed", closest_elements_bed)
            closest_element_df = closest_elements_bed.to_dataframe(header=None)
            if not closest_element_df.empty:
                closest_element_df = closest_element_df.rename({5: 'variant_id', 9: 'a_closest_element'}, axis=1)

                closest_elements = {}
                for index, row in closest_element_df.iterrows():
                    variant_id = row['variant_id']
                    element_name = row['a_closest_element']
                    print(element_name)

                    if variant_id not in closest_elements:
                        closest_elements[variant_id] = []
                    closest_elements[variant_id].append(element_name)


                logging.debug(f"Closest elements ({element_label}) within window table:\n{closest_element_df.shape}\n{closest_element_df.head()}")
                closest_element_df = closest_element_df[['variant_id']]
                output_label = f"{element_label}_within_{window_size}_bp"
                closest_element_df[output_label] = closest_element_df['variant_id'].apply(
                    lambda x: '; '.join(closest_elements[x]) if x in closest_elements else ''
                )

                closest_element_df.drop_duplicates(inplace=True)
            else:
                # Make empty column if no elements are within the window.
                closest_element_df = pd.DataFrame(columns=['variant_id', f"{element_label}_within_{window_size}_bp"])
                closest_element_df['variant_id'] = variant_scores['variant_id']
                closest_element_df[f"{element_label}_within_{window_size}_bp"] = ''

            variant_scores = variant_scores.merge(closest_element_df[['variant_id', output_label]], on='variant_id', how='left')

    if args.peaks:

        logging.info("Annotating with peak overlap")
        peak_df = pd.read_table(peak_path, header=None)
        peak_bed = pybedtools.BedTool.from_dataframe(peak_df)
        peak_intersect_bed = variant_bed.intersect(peak_bed, wa=True, u=True)

        peak_intersect_df = peak_intersect_bed.to_dataframe(names=variant_scores_bed_format.columns.tolist())

        logging.debug(f"Peak overlap table:\n{peak_intersect_df.shape}\n{peak_intersect_df.head()}")

        variant_scores['peak_overlap'] = variant_scores['variant_id'].isin(peak_intersect_df['variant_id'].tolist())

    if args.r2:
        logging.info("Annotating with r2")
        r2_ld_filepath = args.r2

        r2_tsv_filepath = f"/tmp/{args.sample_name}.r2.tsv"
        with open(r2_ld_filepath, 'r') as r2_ld_file, open(r2_tsv_filepath, mode='w') as r2_tsv_file:
            # temp=r2_tsv_file.name
            for line in r2_ld_file:
                # Process the line
                line = '\t'.join(line.split())
                # Write the processed line to the output file, no need to specify end='' as '\n' is added explicitly
                r2_tsv_file.write(line + '\n')
            r2_tsv_file.flush()
            
        with open(r2_tsv_filepath, 'r') as r2_tsv_file:
            plink_variants = pd.read_table(r2_tsv_file)
            logging.debug(f"Plink variants table:\n{plink_variants.shape}\n{plink_variants.head()}")

            # Get just the lead variants, which is provided by the user.
            lead_variants = variant_scores[['chr', 'pos', 'variant_id']].copy()
            lead_variants['r2'] = 1.0
            lead_variants['lead_variant'] = lead_variants['variant_id']
            logging.debug(f"Lead variants table:\n{lead_variants.head()}\n{lead_variants.shape}")

            # Get just the ld variants.
            plink_ld_variants = plink_variants[['SNP_A','CHR_B','BP_B','SNP_B','R2']].copy()
            plink_ld_variants.columns = ['lead_variant', 'chr', 'pos', 'variant_id', 'r2']
            plink_ld_variants = plink_ld_variants[['chr', 'pos', 'variant_id', 'r2', 'lead_variant']]
            plink_ld_variants['chr'] = 'chr' + plink_ld_variants['chr'].astype(str)
            plink_ld_variants = plink_ld_variants.sort_values(by=['variant_id', 'r2'], ascending=False).drop_duplicates(subset='variant_id')
            logging.debug(f"Plink LD variants table:\n{plink_ld_variants.shape}\n{plink_ld_variants.head()}")

            all_plink_variants = pd.concat([lead_variants, plink_ld_variants])
            all_plink_variants = all_plink_variants[['variant_id', 'r2', 'lead_variant']]
            all_plink_variants = all_plink_variants.sort_values( by=['variant_id', 'r2'], ascending=False)
            logging.debug(f"All plink variants table:\n{all_plink_variants.shape}\n{all_plink_variants.head()}")

            variant_scores = variant_scores.merge(all_plink_variants,
                on=['variant_id'],
                how='left')
            
    if args.add_adastra:
        adastra_tf_file = args.adastra_tf_file
        adastra_celltype_file = args.adastra_celltype_file
        sig_adastra_tf = pd.read_table(adastra_tf_file)
        sig_adastra_celltype = pd.read_table(adastra_celltype_file)

        # Modify both to have a variant_id column, since we don't retrieve their rsids. This takes some extra time, might be worth changing later.
        # variant_id should be <chr>:<pos>:<ref>:<alt>
        sig_adastra_tf['variant_id'] = sig_adastra_tf.apply(lambda x: f"{x['#chr']}:{x['pos']}:{x['ref']}:{x['alt']}", axis=1)
        sig_adastra_celltype['variant_id'] = sig_adastra_celltype.apply(lambda x: f"{x['#chr']}:{x['pos']}:{x['ref']}:{x['alt']}", axis=1)

        logging.debug(f"ADASTRA TF table:\n{sig_adastra_tf.shape}\n{sig_adastra_tf.head()}")
        logging.debug(f"ADASTRA celltype table:\n{sig_adastra_celltype.shape}\n{sig_adastra_celltype.head()}")

        n_threads = args.threads if args.threads else DEFAULT_THREADS
        chunk_size = len(variant_scores) // n_threads
        chunks = np.array_split(variant_scores, len(variant_scores) // chunk_size)

        args_for_starmap = [(chunk, sig_adastra_tf, sig_adastra_celltype) for chunk in chunks]

        with Pool(processes=n_threads) as pool:
            results = pool.starmap(get_asb_adastra, args_for_starmap)

        variant_scores = pd.concat(results)

        pool.close()
        pool.join()

        logging.debug(f"ADASTRA annotations added to variant scores:\n{variant_scores.shape}\n{variant_scores.head()}")


    logging.info(f"Final annotation table:\n{variant_scores.shape}\n{variant_scores.head()}")

    out_file = get_annotation_output_file(args.annotation_output_dir, args.sample_name)
    variant_scores.to_csv(out_file,\
                          sep="\t",\
                          index=False)

    logging.info(f"Annotation step completed! Output written to: {out_file}")


if __name__ == "__main__":
    main()
