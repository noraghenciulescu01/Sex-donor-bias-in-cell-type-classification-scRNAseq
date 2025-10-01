# Sex-donor-bias-in-cell-type-classification-scRNAseq

This code accompanies the manuscript "Donor-specific, but not sex-specific, signatures dominate cell-type classification in lung scRNA-seq data". Folders 1 and 2 correspond to the main analyses in the text, folder 3 includes additional experiments that are mentioned in the main text but are mostly included in the Supplementary Materials.

1. Exploration + DA of HLCA core - corresponds to the section "Cell sub-populations in the HLCA show significant sex-bias in abundance" in the results (Main Figure 1)

2. Main classification experiment - this includes all the main classification code for the sections "Cell type classification is influenced by donor variation, not sex-specific effects" and "Sex-specific differences only impact classification performance of AT0 and Goblet bronchial cells" (Main Figures 2, 3)

3. Additional experiments - this includes the experiment where we verify that there is donor bias in the naive split setting (Main Figure 2e) and the experiments where we fix cell counts for a particular cell type (Monocyte-derived Mphs and Suprabasal cells in Supplementary Figure 6; CD4 T-cells in Supplementary Figure 8, Supplementary Table 3)

The data used could not be included in this repo due to size issues but can be accessed at https://cellxgene.cziscience.com/collections/6f6d381a-7701-4781-935c-db10d30de293 (the core HLCA). The path to the data will therefore need to be changed in order to run the notebooks in this repo.