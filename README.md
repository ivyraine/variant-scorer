# variant-scorer
The variant scoring repository provides a set of scripts for scoring genetic variants using a ChromBPNet model. 

#### variant_scoring.py - This script takes a list of variants in various input formats and generates scores for the variants using a ChromBPNet model. The output is a TSV file containing the scores for each variant. 

python variant_scoring.py -l [VARIANTS_FILE] -g [GENOME_FASTA] -m [MODEL_PATH] -o [OUT_PREFIX] -s [CHROM_SIZES] [OTHER_ARGS]

````
Input arguments:

-l or --list: (required) a TSV file containing the list of variants to score.

-g or --genome: (required) a genome fasta file.

-pg or --peak_genome: a genome fasta file for peaks.

-m or --model: (required) the ChromBPNet model to use for variant scoring.

-o or --out_prefix: (required) the path to store SNP effect score predictions from the script. The directory should already exist.

-s or --chrom_sizes: (required) the path to a TSV file with chromosome sizes.

-ps or --peak_chrom_sizes: the path to a TSV file with chromosome sizes for the peak genome.

-dm or --debug_mode: display allele input sequences.

-bs or --batch_size: the batch size to use for the model. Default is 512.

-sc or --schema: the format for the input variants list. Choices are: 'bed', 'plink', 'neuro-variants', 'chrombpnet', 'original'. Default is 'chrombpnet'.

-p or --peaks: a bed file containing peak regions.

-n or --num_shuf: the number of shuffled scores per SNP. Default is 10.

-t or --total_shuf: the total number of shuffled scores across all SNPs. Overrides --num_shuf.

-c or --chrom: only score SNPs in the selected chromosome.

-r or --random_seed: the random seed for reproducibility when sampling. Default is 1234.

--no_hdf5: do not save detailed predictions in hdf5 file.

-fo or --forward_only: run variant scoring only on forward sequence.

-st or --shap_type: the type of SHAP values to compute. Default is "counts".

````

#### variant_summary_across_folds.py - This script takes variant scores generated by the variant_scoring.py script and generates a summary file for each score type. The output is a TSV file containing summary statistics for each score type. The input arguments for this script are as follows:


---

**Note:** pos (position) column is for 1-indexed SNP position. In a BED file, this is the third column.
