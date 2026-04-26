# Sex-donor-bias-in-cell-type-classification-scRNAseq

This code accompanies the manuscript "Donor-specific, but not sex-specific, signatures dominate cell-type classification in lung scRNA-seq data". Folders 1 and 2 correspond to the main analyses in the text, folder 3 includes additional experiments that are mentioned in the main text but are mostly included in the Supplementary Materials.

1. Exploration + DA of HLCA core - corresponds to the section "Cell sub-populations in the HLCA show significant sex imbalance in abundance" in the results (Figures 1, 2; Supplementary Figures 1, 2; Supplementary Table 1)

2. Main classification experiment - main classification experiment for section "Cell type classification is influenced by donor variation, not sex-specific effects" in the results; results and categorization of individual cell types without cell count fixing or matching throughout sections "Sex-specific classification differences at the individual cell-type level disappear when controlling for donor overlap and other confounders" and "Abundance differences can confound classification results" in the results (Figure 3c,d; Supplementary Figures 4-12; Supplementary Tables 2-4)

3. Additional experiments - includes several experiments: verification of donor bias by excluding donors from training, fixing cell counts in naive and donor-based setting, matching on confounders and batch correction (Figure 3e; Supplementary Figures 3, 8, 9, 11, 13; Supplementary Tables 5, 6)

The data used could not be included here due to size issues but can be accessed at https://cellxgene.cziscience.com/collections/6f6d381a-7701-4781-935c-db10d30de293 (the core HLCA). The path to the data will therefore need to be changed in order to run the notebooks in this repo.

All required packages and the corresponding package versions used are given in requirements.txt.