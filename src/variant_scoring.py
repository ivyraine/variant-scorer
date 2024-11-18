import pandas as pd
import os
import numpy as np
import h5py
from utils.helpers import *
import logging

def main(args = None, filter_dir_override = None):

    variants_table = None

    np.random.seed(args.random_seed)
    if args.forward_only:
        print("running variant scoring only for forward sequences")

    # load the model and variants
    model = load_model_wrapper(args.model_path, is_compiling=args.model_architecture == 'bpnet')
    variants_table = load_variant_table(args.variant_list, args.schema)
    variants_table = variants_table.fillna('-')
    
    chrom_sizes = pd.read_csv(args.chrom_sizes, header=None, sep='\t', names=['chrom', 'size'])
    chrom_sizes_dict = chrom_sizes.set_index('chrom')['size'].to_dict()

    print("Original variants table shape:", variants_table.shape)

    # infer input length
    if args.model_architecture == "chrombpnet":
        input_len = model.input_shape[1]
    elif args.model_architecture == 'chrombpnet-lite':
        input_len = model.input_shape[0][1]
    elif args.model_architecture == 'bpnet':
        input_len = model.input_shape[0][1]
    else:
        raise ValueError(f"Model architecture {args.model_architecture} not supported")

    print("Input length inferred from the model:", input_len)

    variants_table = variants_table.loc[variants_table.apply(lambda x: get_valid_variants(x.chr, x.pos, x.allele1, x.allele2, input_len, chrom_sizes_dict), axis=1)]
    variants_table.reset_index(drop=True, inplace=True)

    print("Final variants table shape:", variants_table.shape)

    if args.shuffled_scores:
        shuf_variants_table = pd.read_table(args.shuffled_scores)
        print("Shuffled variants table shape:", shuf_variants_table.shape)
        shuf_scores_file = args.shuffled_scores

    else:
        shuf_variants_table = create_shuffle_table(variants_table, args.random_seed, args.total_shuf, args.num_shuf)
        print("Shuffled variants table shape:", shuf_variants_table.shape)
        shuf_scores_file = get_score_shuffled_path(args.score_output_path_prefix)

    peak_scores_file = get_score_peaks_path(args.score_output_path_prefix)

    if len(shuf_variants_table) > 0:
        if args.debug_mode:
            shuf_variants_table = shuf_variants_table.sample(10000, random_state=args.random_seed, ignore_index=True)
            print()
            print(shuf_variants_table.head())
            print("Debug shuffled variants table shape:", shuf_variants_table.shape)
            print()

    shuf_variants_done = False
    if os.path.isfile(shuf_scores_file):
        shuf_variants_table_loaded = pd.read_table(shuf_scores_file)
        if shuf_variants_table_loaded['variant_id'].tolist() == shuf_variants_table['variant_id'].tolist():
            shuf_variants_table = shuf_variants_table_loaded.copy()
            shuf_variants_done = True
        
    if not shuf_variants_done:
        shuf_variant_ids, shuf_allele1_pred_counts, shuf_allele2_pred_counts, \
        shuf_allele1_pred_profiles, shuf_allele2_pred_profiles = fetch_variant_predictions(model,
                                                                            shuf_variants_table,
                                                                            input_len,
                                                                            args.genome,
                                                                            args.batch_size,
                                                                            model_architecture=args.model_architecture,
                                                                            debug_mode=args.debug_mode,
                                                                            shuf=True,
                                                                            forward_only=args.forward_only)
        assert np.array_equal(shuf_variants_table["variant_id"].tolist(), shuf_variant_ids)
        shuf_variants_table["allele1_pred_counts"] = shuf_allele1_pred_counts
        shuf_variants_table["allele2_pred_counts"] = shuf_allele2_pred_counts

    if args.peaks:
        if args.peak_chrom_sizes == None:
            args.peak_chrom_sizes = args.chrom_sizes
        if args.peak_genome == None:
            args.peak_genome = args.genome

        peak_chrom_sizes = pd.read_csv(args.peak_chrom_sizes, header=None, sep='\t', names=['chrom', 'size'])
        peak_chrom_sizes_dict = peak_chrom_sizes.set_index('chrom')['size'].to_dict()

        peaks = pd.read_csv(args.peaks, header=None, sep='\t')
        peaks = add_missing_columns_to_peaks_df(peaks, schema='narrowpeak')
        peaks['peak_id'] = peaks['chr'] + ':' + peaks['start'].astype(str) + '-' + peaks['end'].astype(str)

        print("Original peak table shape:", peaks.shape)

        peaks.sort_values(by=['chr', 'start', 'end', 'summit', 'rank'], ascending=[True, True, True, True, False], inplace=True)
        peaks.drop_duplicates(subset=['chr', 'start', 'end', 'summit'], inplace=True)
        peaks = peaks.loc[peaks.apply(lambda x: get_valid_peaks(x.chr, x.start, x.summit, input_len, peak_chrom_sizes_dict), axis=1)]
        peaks.reset_index(drop=True, inplace=True)

        print("De-duplicated peak table shape:", peaks.shape)

        if args.debug_mode:
            peaks = peaks.sample(10000, random_state=args.random_seed, ignore_index=True)
            logging.debug(f"Debug peak table:\n{peaks.head()}\n{peaks.shape}")

        if args.max_peaks:
            if len(peaks) > args.max_peaks:
                peaks = peaks.sample(args.max_peaks, random_state=args.random_seed, ignore_index=True)
                print("Subsampled peak table shape:", peaks.shape)

    peak_scores_done = False
    if os.path.isfile(peak_scores_file):
        peaks_loaded = pd.read_table(peak_scores_file)
        if peaks_loaded['peak_id'].tolist() == peaks['peak_id'].tolist():
            peaks = peaks_loaded.copy()
            peak_scores_done = True
        
    if not peak_scores_done:
        peak_ids, peak_pred_counts, peak_pred_profiles = fetch_peak_predictions(model,
                                                            peaks,
                                                            input_len,
                                                            args.peak_genome,
                                                            args.batch_size,
                                                            model_architecture=args.model_architecture,
                                                            debug_mode=args.debug_mode,
                                                            forward_only=args.forward_only)
        assert np.array_equal(peaks["peak_id"].tolist(), peak_ids)
        peaks["peak_score"] = peak_pred_counts
        logging.debug(f"Peak table with scores:\n{peaks.head()}\n{peaks.shape}")
        os.makedirs(get_score_dir(args.score_output_path_prefix), exist_ok=True)
        peaks.to_csv(peak_scores_file, sep="\t", index=False)

    if len(shuf_variants_table) > 0 and not shuf_variants_done:
        shuf_logfc, shuf_jsd, \
        shuf_allele1_quantile, shuf_allele2_quantile = get_variant_scores_with_peaks(shuf_allele1_pred_counts,
                                                                                        shuf_allele2_pred_counts,
                                                                                        shuf_allele1_pred_profiles,
                                                                                        shuf_allele2_pred_profiles,
                                                                                        np.array(peaks["peak_score"].tolist()))
        shuf_indel_idx, shuf_adjusted_jsd_list = adjust_indel_jsd(shuf_variants_table,
                                                                    shuf_allele1_pred_profiles,
                                                                    shuf_allele2_pred_profiles,
                                                                    shuf_jsd)
        shuf_has_indel_variants = (len(shuf_indel_idx) > 0)
        
        shuf_variants_table["logfc"] = shuf_logfc
        shuf_variants_table["abs_logfc"] = np.abs(shuf_logfc)
        if shuf_has_indel_variants:
            shuf_variants_table["jsd"] = shuf_adjusted_jsd_list
        else:
            shuf_variants_table["jsd"] = shuf_jsd
            assert np.array_equal(shuf_adjusted_jsd_list, shuf_jsd)
        shuf_variants_table['original_jsd'] = shuf_jsd
        shuf_variants_table["logfc_x_jsd"] =  shuf_variants_table["logfc"] * shuf_variants_table["jsd"]
        shuf_variants_table["abs_logfc_x_jsd"] = shuf_variants_table["abs_logfc"] * shuf_variants_table["jsd"]

        shuf_variants_table["allele1_quantile"] = shuf_allele1_quantile
        shuf_variants_table["allele2_quantile"] = shuf_allele2_quantile
        shuf_variants_table["active_allele_quantile"] = shuf_variants_table[["allele1_quantile", "allele2_quantile"]].max(axis=1)
        shuf_variants_table["quantile_change"] = shuf_variants_table["allele2_quantile"] - shuf_variants_table["allele1_quantile"]
        shuf_variants_table["abs_quantile_change"] = np.abs(shuf_variants_table["quantile_change"])
        shuf_variants_table["logfc_x_active_allele_quantile"] = shuf_variants_table["logfc"] * shuf_variants_table["active_allele_quantile"]
        shuf_variants_table["abs_logfc_x_active_allele_quantile"] = shuf_variants_table["abs_logfc"] * shuf_variants_table["active_allele_quantile"]
        shuf_variants_table["jsd_x_active_allele_quantile"] = shuf_variants_table["jsd"] * shuf_variants_table["active_allele_quantile"]
        shuf_variants_table["logfc_x_jsd_x_active_allele_quantile"] = shuf_variants_table["logfc_x_jsd"] * shuf_variants_table["active_allele_quantile"]
        shuf_variants_table["abs_logfc_x_jsd_x_active_allele_quantile"] = shuf_variants_table["abs_logfc_x_jsd"] * shuf_variants_table["active_allele_quantile"]

        assert shuf_variants_table["abs_logfc"].shape == shuf_logfc.shape
        assert shuf_variants_table["abs_logfc"].shape == shuf_jsd.shape
        assert shuf_variants_table["abs_logfc"].shape == shuf_variants_table["abs_logfc_x_jsd"].shape

        logging.debug("Shuffled score table shape:", shuf_variants_table.shape, shuf_variants_table.head())
        shuf_variants_table.to_csv(shuf_scores_file, sep="\t", index=False)

    else:
        if len(shuf_variants_table) > 0 and not shuf_variants_done:
            shuf_logfc, shuf_jsd = get_variant_scores(shuf_allele1_pred_counts,
                                                    shuf_allele2_pred_counts,
                                                    shuf_allele1_pred_profiles,
                                                    shuf_allele2_pred_profiles)
            
            shuf_indel_idx, shuf_adjusted_jsd_list = adjust_indel_jsd(shuf_variants_table,
                                                                    shuf_allele1_pred_profiles,
                                                                    shuf_allele2_pred_profiles,
                                                                    shuf_jsd)
            shuf_has_indel_variants = (len(shuf_indel_idx) > 0)
            
            shuf_variants_table["logfc"] = shuf_logfc
            shuf_variants_table["abs_logfc"] = np.abs(shuf_logfc)
            if shuf_has_indel_variants:
                shuf_variants_table["jsd"] = shuf_adjusted_jsd_list
            else:
                shuf_variants_table["jsd"] = shuf_jsd
                assert np.array_equal(shuf_adjusted_jsd_list, shuf_jsd)
            shuf_variants_table['original_jsd'] = shuf_jsd
            shuf_variants_table["logfc_x_jsd"] =  shuf_variants_table["logfc"] * shuf_variants_table["jsd"]
            shuf_variants_table["abs_logfc_x_jsd"] = shuf_variants_table["abs_logfc"] * shuf_variants_table["jsd"]

            assert shuf_variants_table["abs_logfc"].shape == shuf_logfc.shape
            assert shuf_variants_table["abs_logfc"].shape == shuf_jsd.shape
            assert shuf_variants_table["abs_logfc"].shape == shuf_variants_table["abs_logfc_x_jsd"].shape

            print()
            print(shuf_variants_table.head())
            print("Shuffled score table shape:", shuf_variants_table.shape)
            print()
            shuf_variants_table.to_csv(shuf_scores_file, sep="\t", index=False)

    if args.debug_mode:
        variants_table = variants_table.sample(10000, random_state=args.random_seed, ignore_index=True)
        print()
        print(variants_table.head())
        print("Debug variants table shape:", variants_table.shape)
        print()

    todo_chroms = [x for x in variants_table.chr.unique()] if args.split_per_chromosome else ['all']

    for chrom in todo_chroms:

        logging.info(f'Processing {chrom} variants')
        output_tsv = get_score_file_path(args.score_output_path_prefix, args.scores_suffix, chr=chrom)
        if args.split_per_chromosome:
            curr_variants_table = variants_table.loc[variants_table['chr'] == chrom].sort_values(by='pos').copy()
            curr_variants_table.reset_index(drop=True, inplace=True)

            if os.path.isfile(output_tsv):
                curr_variants_table_loaded = pd.read_table(output_tsv)
                if curr_variants_table_loaded['variant_id'].tolist() == curr_variants_table['variant_id'].tolist():
                    logging.info(f"Skipping {chrom} as the {output_tsv} already exists")
                    continue
        else:
            curr_variants_table = variants_table

        # fetch model prediction for variants
        variant_ids, allele1_pred_counts, allele2_pred_counts, \
        allele1_pred_profiles, allele2_pred_profiles = fetch_variant_predictions(model,
                                                                            curr_variants_table,
                                                                            input_len,
                                                                            args.genome,
                                                                            args.batch_size,
                                                                            model_architecture=args.model_architecture,
                                                                            debug_mode=args.debug_mode,
                                                                            shuf=False,
                                                                            forward_only=args.forward_only)

        if args.peaks:
            logfc, jsd, \
            allele1_quantile, allele2_quantile = get_variant_scores_with_peaks(allele1_pred_counts,
                                                                                    allele2_pred_counts,
                                                                                    allele1_pred_profiles,
                                                                                    allele2_pred_profiles,
                                                                                    np.array(peaks["peak_score"].tolist()))
        else:
            logfc, jsd = get_variant_scores(allele1_pred_counts,
                                            allele2_pred_counts,
                                            allele1_pred_profiles,
                                            allele2_pred_profiles)

        indel_idx, adjusted_jsd_list = adjust_indel_jsd(curr_variants_table,allele1_pred_profiles,allele2_pred_profiles,jsd)
        has_indel_variants = (len(indel_idx) > 0)

        assert np.array_equal(curr_variants_table["variant_id"].tolist(), variant_ids)
        curr_variants_table["allele1_pred_counts"] = allele1_pred_counts
        curr_variants_table["allele2_pred_counts"] = allele2_pred_counts
        curr_variants_table["logfc"] = logfc
        curr_variants_table["abs_logfc"] = np.abs(curr_variants_table["logfc"])
        if has_indel_variants:
            curr_variants_table["jsd"] = adjusted_jsd_list
        else:
            curr_variants_table["jsd"] = jsd
            assert np.array_equal(adjusted_jsd_list, jsd)
        curr_variants_table["original_jsd"] = jsd
        curr_variants_table["logfc_x_jsd"] = curr_variants_table["logfc"] * curr_variants_table["jsd"]
        curr_variants_table["abs_logfc_x_jsd"] = curr_variants_table["abs_logfc"] * curr_variants_table["jsd"]

        if len(shuf_variants_table) > 0:
            curr_variants_table["logfc.pval"] = get_pvals(curr_variants_table["logfc"].tolist(), shuf_variants_table["logfc"], tail="both")
            curr_variants_table["abs_logfc.pval"] = get_pvals(curr_variants_table["abs_logfc"].tolist(), shuf_variants_table["abs_logfc"], tail="right")
            curr_variants_table["jsd.pval"] = get_pvals(curr_variants_table["jsd"].tolist(), shuf_variants_table["jsd"], tail="right")
            curr_variants_table["logfc_x_jsd.pval"] = get_pvals(curr_variants_table["logfc_x_jsd"].tolist(), shuf_variants_table["logfc_x_jsd"], tail="both")
            curr_variants_table["abs_logfc_x_jsd.pval"] = get_pvals(curr_variants_table["abs_logfc_x_jsd"].tolist(), shuf_variants_table["abs_logfc_x_jsd"], tail="right")
        if args.peaks:
            curr_variants_table["allele1_quantile"] = allele1_quantile
            curr_variants_table["allele2_quantile"] = allele2_quantile
            curr_variants_table["active_allele_quantile"] = curr_variants_table[["allele1_quantile", "allele2_quantile"]].max(axis=1)
            curr_variants_table["quantile_change"] = curr_variants_table["allele2_quantile"] - curr_variants_table["allele1_quantile"]
            curr_variants_table["abs_quantile_change"] = np.abs(curr_variants_table["quantile_change"])
            curr_variants_table["logfc_x_active_allele_quantile"] = curr_variants_table["logfc"] * curr_variants_table["active_allele_quantile"]
            curr_variants_table["abs_logfc_x_active_allele_quantile"] = curr_variants_table["abs_logfc"] * curr_variants_table["active_allele_quantile"]
            curr_variants_table["jsd_x_active_allele_quantile"] = curr_variants_table["jsd"] * curr_variants_table["active_allele_quantile"]
            curr_variants_table["logfc_x_jsd_x_active_allele_quantile"] = curr_variants_table["logfc_x_jsd"] * curr_variants_table["active_allele_quantile"]
            curr_variants_table["abs_logfc_x_jsd_x_active_allele_quantile"] = curr_variants_table["abs_logfc_x_jsd"] * curr_variants_table["active_allele_quantile"]

            if len(shuf_variants_table) > 0:
                curr_variants_table["active_allele_quantile.pval"] = get_pvals(curr_variants_table["active_allele_quantile"].tolist(),
                                                                    shuf_variants_table["active_allele_quantile"], tail="right")
                curr_variants_table['quantile_change.pval'] = get_pvals(curr_variants_table["quantile_change"].tolist(),
                                                                        shuf_variants_table["quantile_change"], tail="both")
                curr_variants_table["abs_quantile_change.pval"] = get_pvals(curr_variants_table["abs_quantile_change"].tolist(),
                                                                            shuf_variants_table["abs_quantile_change"], tail="right")
                curr_variants_table["logfc_x_active_allele_quantile.pval"] = get_pvals(curr_variants_table["logfc_x_active_allele_quantile"].tolist(),
                                                                            shuf_variants_table["logfc_x_active_allele_quantile"], tail="both")
                curr_variants_table["abs_logfc_x_active_allele_quantile.pval"] = get_pvals(curr_variants_table["abs_logfc_x_active_allele_quantile"].tolist(),
                                                                                shuf_variants_table["abs_logfc_x_active_allele_quantile"], tail="right")
                curr_variants_table["jsd_x_active_allele_quantile.pval"] = get_pvals(curr_variants_table["jsd_x_active_allele_quantile"].tolist(),
                                                                        shuf_variants_table["jsd_x_active_allele_quantile"], tail="right")
                curr_variants_table["logfc_x_jsd_x_active_allele_quantile.pval"] = get_pvals(curr_variants_table["logfc_x_jsd_x_active_allele_quantile"].tolist(),
                                                                                shuf_variants_table["logfc_x_jsd_x_active_allele_quantile"], tail="both")
                curr_variants_table["abs_logfc_x_jsd_x_active_allele_quantile.pval"] = get_pvals(curr_variants_table["abs_logfc_x_jsd_x_active_allele_quantile"].tolist(),
                                                                                    shuf_variants_table["abs_logfc_x_jsd_x_active_allele_quantile"], tail="right")

        if args.schema == "bed":
            curr_variants_table['pos'] = curr_variants_table['pos'] - 1

        logging.info(f"Output score table:\n{curr_variants_table.head()}\n{curr_variants_table.shape}")

        print(f"writing to {output_tsv}")
        curr_variants_table.to_csv(output_tsv, sep="\t", index=False)

        # store predictions at variants
        if hasattr(args, "no_hdf5") and not args.no_hdf5 or filter_dir_override is not None:
            output_h5 = get_profiles_file_path(args.score_output_path_prefix, chrom)
            with h5py.File(output_h5, 'w') as f:
                observed = f.create_group('observed')
                # Print shapes
                observed.create_dataset('allele1_pred_counts', data=allele1_pred_counts, compression='gzip', compression_opts=9)
                observed.create_dataset('allele2_pred_counts', data=allele2_pred_counts, compression='gzip', compression_opts=9)
                observed.create_dataset('allele1_pred_profiles', data=allele1_pred_profiles, compression='gzip', compression_opts=9)
                observed.create_dataset('allele2_pred_profiles', data=allele2_pred_profiles, compression='gzip', compression_opts=9)
                variant_ids_encoded = variant_ids.astype(h5py.string_dtype(encoding='utf-8'))
                # Unencode it
                # variant_ids_decoded = [x.decode('utf-8') for x in variant_ids_encoded]
                # print(variant_ids_decoded)
                observed.create_dataset('variant_ids', data=variant_ids_encoded, compression='gzip', compression_opts=9)
                # if len(shuf_variants_table) > 0:
                #     shuffled = f.create_group('shuffled')
                #     shuffled.create_dataset('shuf_allele1_pred_counts', data=shuf_allele1_pred_counts, compression='gzip', compression_opts=9)
                #     shuffled.create_dataset('shuf_allele2_pred_counts', data=shuf_allele2_pred_counts, compression='gzip', compression_opts=9)
                #     shuffled.create_dataset('shuf_logfc', data=shuf_logfc, compression='gzip', compression_opts=9)
                #     shuffled.create_dataset('shuf_abs_logfc', data=shuf_abs_logfc, compression='gzip', compression_opts=9)
                #     shuffled.create_dataset('shuf_jsd', data=shuf_jsd, compression='gzip', compression_opts=9)
                #     shuffled.create_dataset('shuf_logfc_x_jsd', data=shuf_logfc_jsd, compression='gzip', compression_opts=9)
                #     shuffled.create_dataset('shuf_abs_logfc_x_jsd', data=shuf_abs_logfc_jsd, compression='gzip', compression_opts=9)
                #     if args.peaks:
                #         shuffled.create_dataset('shuf_max_percentile', data=shuf_max_percentile, compression='gzip', compression_opts=9)
                #         shuffled.create_dataset('shuf_percentile_change', data=shuf_percentile_change, compression='gzip', compression_opts=9)
                #         shuffled.create_dataset('shuf_abs_percentile_change', data=shuf_abs_percentile_change, compression='gzip', compression_opts=9)
                #         shuffled.create_dataset('shuf_logfc_x_max_percentile', data=shuf_logfc_max_percentile, compression='gzip', compression_opts=9)
                #         shuffled.create_dataset('shuf_abs_logfc_max_percentile', data=shuf_abs_logfc_max_percentile, compression='gzip', compression_opts=9)
                #         shuffled.create_dataset('shuf_jsd_max_percentile', data=shuf_jsd_max_percentile, compression='gzip', compression_opts=9)
                #         shuffled.create_dataset('shuf_logfc_x_jsd_x_max_percentile', data=shuf_logfc_jsd_max_percentile, compression='gzip', compression_opts=9)
                #         shuffled.create_dataset('shuf_abs_logfc_x_jsd_x_max_percentile', data=shuf_abs_logfc_jsd_max_percentile, compression='gzip', compression_opts=9)
            logging.info(f"Finished scoring {chrom}. Saved to {output_tsv} and {output_h5}")
        else:
            logging.info(f"Finished scoring {chrom}. Saved to {output_tsv}")


if __name__ == "__main__":
    main()
