import pandas as pd
import os
import numpy as np
import h5py
from utils import argmanager
from utils.helpers import *
import logging

def main(args = None, filter_dir_override = None, filtered_variants_df_override = None):
    if args is None:
        args = argmanager.fetch_scoring_args()
        logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')

    output_dir = None
    if filter_dir_override is not None:
        output_dir = filter_dir_override
    elif not os.path.isdir(args.score_dir):
        raise OSError(f"Output directory ({output_dir}) does not exist or is not a directory")
    else:
        output_dir = args.score_dir

    is_using_result_filename_overrides = False
    is_filter_step = filter_dir_override is not None
    scoring_output_prefixes = [ get_score_output_file_prefix(output_dir, args.model_name, index, is_filter_step=is_filter_step) for index in range(len(args.models)) ]
    if args.score_filenames is not None:
        if len(args.score_filenames) != len(args.models):
            raise ValueError("Number of models and score filenames do not match, which they are requird to do if the --score-filenames flag is given. Exiting.")
        else:
            scoring_output_prefixes = [ os.path.join(output_dir, args.score_filenames[i]) for i in range(len(args.models)) ]
            is_using_result_filename_overrides = True

    model_and_output_prefixes = list(zip(args.models, scoring_output_prefixes))

    variants_table = None
    if filtered_variants_df_override is not None:
        variants_table = filtered_variants_df_override[['chr', 'pos', 'allele1', 'allele2', 'variant_id']]
    else:
        variants_table = load_variant_table(args.variant_list, args.schema)
        variants_table = variants_table.fillna('-')

    for model_index, (model_name, scoring_output_prefix) in enumerate(model_and_output_prefixes):
        np.random.seed(args.random_seed)
        if args.forward_only:
            print("running variant scoring only for forward sequences")

        # load the model and variants
        model = load_model_wrapper(model_name)
        
        chrom_sizes = pd.read_csv(args.chrom_sizes, header=None, sep='\t', names=['chrom', 'size'])
        chrom_sizes_dict = chrom_sizes.set_index('chrom')['size'].to_dict()

        print("Original variants table shape:", variants_table.shape)

        if args.chrom:
            variants_table = variants_table.loc[variants_table['chr'] == args.chrom]
            print("Chromosome variants table shape:", variants_table.shape)

        # infer input length
        if args.lite:
            input_len = model.input_shape[0][1]
        else:
            input_len = model.input_shape[1]

        print("Input length inferred from the model:", input_len)

        variants_table = variants_table.loc[variants_table.apply(lambda x: get_valid_variants(x.chr, x.pos, x.allele1, x.allele2, input_len, chrom_sizes_dict), axis=1)]
        variants_table.reset_index(drop=True, inplace=True)

        print("Final variants table shape:", variants_table.shape)

        shuf_variants_table = create_shuffle_table(variants_table,args.random_seed, args.total_shuf, args.num_shuf)
        print("Shuffled variants table shape:", shuf_variants_table.shape)

        shuf_scores_file = '.'.join([scoring_output_prefix, "variant_scores.shuffled.tsv"])
        peak_scores_file = '.'.join([scoring_output_prefix, "peak_scores.tsv"])

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
                                                                                debug_mode=args.debug_mode,
                                                                                lite=args.lite,
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
                print()
                print(peaks.head())
                print("Debug peak table shape:", peaks.shape)
                print()

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
                                                                debug_mode=args.debug_mode,
                                                                lite=args.lite,
                                                                forward_only=args.forward_only)
            assert np.array_equal(peaks["peak_id"].tolist(), peak_ids)
            peaks["peak_score"] = peak_pred_counts
            print()
            print(peaks.head())
            print("Peak score table shape:", peaks.shape)
            print()
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

            print()
            print(shuf_variants_table.head())
            print("Shuffled score table shape:", shuf_variants_table.shape)
            print()
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

        # fetch model prediction for variants
        variant_ids, allele1_pred_counts, allele2_pred_counts, \
        allele1_pred_profiles, allele2_pred_profiles = fetch_variant_predictions(model,
                                                                            variants_table,
                                                                            input_len,
                                                                            args.genome,
                                                                            args.batch_size,
                                                                            debug_mode=args.debug_mode,
                                                                            lite=args.lite,
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

        indel_idx, adjusted_jsd_list = adjust_indel_jsd(variants_table,allele1_pred_profiles,allele2_pred_profiles,jsd)
        has_indel_variants = (len(indel_idx) > 0)

        assert np.array_equal(variants_table["variant_id"].tolist(), variant_ids)
        variants_table["allele1_pred_counts"] = allele1_pred_counts
        variants_table["allele2_pred_counts"] = allele2_pred_counts
        variants_table["logfc"] = logfc
        variants_table["abs_logfc"] = np.abs(variants_table["logfc"])
        if has_indel_variants:
            variants_table["jsd"] = adjusted_jsd_list
        else:
            variants_table["jsd"] = jsd
            assert np.array_equal(adjusted_jsd_list, jsd)
        variants_table["original_jsd"] = jsd
        variants_table["logfc_x_jsd"] = variants_table["logfc"] * variants_table["jsd"]
        variants_table["abs_logfc_x_jsd"] = variants_table["abs_logfc"] * variants_table["jsd"]

    if len(shuf_variants_table) > 0:
        variants_table["logfc.pval"] = get_pvals(variants_table["logfc"].tolist(), shuf_variants_table["logfc"], tail="both")
        variants_table["abs_logfc.pval"] = get_pvals(variants_table["abs_logfc"].tolist(), shuf_variants_table["abs_logfc"], tail="right")
        variants_table["jsd.pval"] = get_pvals(variants_table["jsd"].tolist(), shuf_variants_table["jsd"], tail="right")
        variants_table["logfc_x_jsd.pval"] = get_pvals(variants_table["logfc_x_jsd"].tolist(), shuf_variants_table["logfc_x_jsd"], tail="both")
        variants_table["abs_logfc_x_jsd.pval"] = get_pvals(variants_table["abs_logfc_x_jsd"].tolist(), shuf_variants_table["abs_logfc_x_jsd"], tail="right")
    if args.peaks:
        variants_table["allele1_quantile"] = allele1_quantile
        variants_table["allele2_quantile"] = allele2_quantile
        variants_table["active_allele_quantile"] = variants_table[["allele1_quantile", "allele2_quantile"]].max(axis=1)
        variants_table["quantile_change"] = variants_table["allele2_quantile"] - variants_table["allele1_quantile"]
        variants_table["abs_quantile_change"] = np.abs(variants_table["quantile_change"])
        variants_table["logfc_x_active_allele_quantile"] = variants_table["logfc"] * variants_table["active_allele_quantile"]
        variants_table["abs_logfc_x_active_allele_quantile"] = variants_table["abs_logfc"] * variants_table["active_allele_quantile"]
        variants_table["jsd_x_active_allele_quantile"] = variants_table["jsd"] * variants_table["active_allele_quantile"]
        variants_table["logfc_x_jsd_x_active_allele_quantile"] = variants_table["logfc_x_jsd"] * variants_table["active_allele_quantile"]
        variants_table["abs_logfc_x_jsd_x_active_allele_quantile"] = variants_table["abs_logfc_x_jsd"] * variants_table["active_allele_quantile"]

        if len(shuf_variants_table) > 0:
            variants_table["active_allele_quantile.pval"] = get_pvals(variants_table["active_allele_quantile"].tolist(),
                                                              shuf_variants_table["active_allele_quantile"], tail="right")
            variants_table['quantile_change.pval'] = get_pvals(variants_table["quantile_change"].tolist(),
                                                                 shuf_variants_table["quantile_change"], tail="both")
            variants_table["abs_quantile_change.pval"] = get_pvals(variants_table["abs_quantile_change"].tolist(),
                                                                     shuf_variants_table["abs_quantile_change"], tail="right")
            variants_table["logfc_x_active_allele_quantile.pval"] = get_pvals(variants_table["logfc_x_active_allele_quantile"].tolist(),
                                                                      shuf_variants_table["logfc_x_active_allele_quantile"], tail="both")
            variants_table["abs_logfc_x_active_allele_quantile.pval"] = get_pvals(variants_table["abs_logfc_x_active_allele_quantile"].tolist(),
                                                                          shuf_variants_table["abs_logfc_x_active_allele_quantile"], tail="right")
            variants_table["jsd_x_active_allele_quantile.pval"] = get_pvals(variants_table["jsd_x_active_allele_quantile"].tolist(),
                                                                    shuf_variants_table["jsd_x_active_allele_quantile"], tail="right")
            variants_table["logfc_x_jsd_x_active_allele_quantile.pval"] = get_pvals(variants_table["logfc_x_jsd_x_active_allele_quantile"].tolist(),
                                                                            shuf_variants_table["logfc_x_jsd_x_active_allele_quantile"], tail="both")
            variants_table["abs_logfc_x_jsd_x_active_allele_quantile.pval"] = get_pvals(variants_table["abs_logfc_x_jsd_x_active_allele_quantile"].tolist(),
                                                                                shuf_variants_table["abs_logfc_x_jsd_x_active_allele_quantile"], tail="right")

        if args.schema == "bed":
            variants_table['pos'] = variants_table['pos'] - 1

        logging.info(f"Output score table:\n{variants_table.head()}\n{variants_table.shape}")

        output_tsv = f"{scoring_output_prefix}.variant_scores.tsv"
        if is_using_result_filename_overrides:
            output_tsv = f"{scoring_output_prefix}.tsv"
        variants_table.to_csv(output_tsv, sep="\t", index=False)

        # store predictions at variants
        if hasattr(args, "no_hdf5") and not args.no_hdf5 or filter_dir_override is not None:
            output_h5 = f"{scoring_output_prefix}.variant_predictions.h5"
            if is_using_result_filename_overrides:
                output_h5 = f"{scoring_output_prefix}.h5"
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
            logging.info(f"Finished scoring for {model_name} ({model_index+1}/{len(model_and_output_prefixes)}) and saved to {output_tsv} and {output_h5}")
        else:
            logging.info(f"Finished scoring for {model_name} ({model_index+1}/{len(model_and_output_prefixes)}) and saved to {output_tsv}")


if __name__ == "__main__":
    main()
