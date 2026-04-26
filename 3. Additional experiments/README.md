### Description of notebooks and analyses

These are additional experiments that are referenced in the main text but are not part of the main classification experiment. For analyses 1., 2., 3. and 5., results are saved in the Result_plots folder. Since analysis 4. requires more files than the others (it is basically equivalent to the main classification analysis, except with matching), the entire analysis has its own folder.

1. Verify donor-based bias in naive setting

 This is done to check that donor bias indeed affects classifier performance in the naive setting, by excluding some of the donors from training and evaluating performance separately on them. 

 The classification is done as in the main experiment, except the training and prediction function includes a donor exclusion step. Since we only perform this for one random seed, we run this experiment locally and not on the cluster. The resulting plots are Naive_bias_average_acc_female.png and Naive_bias_average_acc_male.png, and these constitute Figure 3e in the manuscript.

2. Abundance in naive setting

 Here we fix the sample counts of one cell type, the CD4 T-cells, in the naive setting. As in the main classification experiment, we first generate classification scripts, then we run them on a cluster, then we analyze the results. The template generation step results in the cd4_fix_classif folder; this has the same organization as the classif folders in the main analysis. Once the scripts in here are run on the cluster, the cd4_fix_classif folder also includes the actual classification results. As for the main analysis, for reproduction of the classification results themselves, manual running of the scripts on the cluster is required.

 Analysis is performed as in the main experiment, except compressed into one single notebook since we are only dealing with one cell type : template generation -> determining proportions -> check general classification trend -> check classification trend for individual cell types (in this case just CD4).

 The resulting plot is saved as CD4 T cells_classification_plot.png and corresponds to Supplementary Figure 11b in the manuscript. Categorization results are manually extracted from the notebook and presented in the second row of Supplementary Table 6 in the manuscript.

3. Abundance in donor-based setting

 Here we fix the sample counts of Suprabasal cells, Monocyte-derived Mphs, AT0 cells and Goblet (bronchial) cells, in the donor-based setting. As in 2., this is anologous to the main classification experiment and is performed on the cluster, using the folder of templates donorbased_fix_classif. Note that we perform this at annotation level 4 for Suprabasal cells and at the finest annotation level for Monocyte-derived Mphs.

 The resulting plots are saved as {cell type label}_classification_plot.png and correspond to subplots in Supplementary Figures 9 and 13 in the manuscript. Categorization results are manually extracted from the notebook and presented in rows 1, 3, 5, 6 in Supplementary Table 5 in the manuscript.

4. Matching analysis

 The set-up of the matching analysis sub-folder is almost identical to the Main classification experiment folder (1. template generation -> 2. determining train & test proportions -> 3. overall result analysis -> 4. analysis per cell type), except we only perform matching in the donor-based setting and for the KNN classifier. When generating classification templates, we add a matching step right after loading the data, where we match each male donor with a female donor with the same smoking status and tissue collecting site, and discard all donors that do not have a match. Then, we run the data splitting and classification experiment exactly as in the main donor-based experiment. The experiment is run on the cluster using the folder of templates classif_matched_donorbased. 
 
 Results are saved in results_matched_donorbased. While we also plot the overall results in notebook 3 for verification purposes, we only save the plots used in the manuscript, namely the individual cell type results: AT0_classification_plot.png and Goblet (bronchial)_classification_plot.png; these correspond to Supplementary Figure 8e,f in the manuscript. Categorization results are manually extracted from the notebook and presented in rows 2 and 4 of Supplementary Table 5 in the manuscript.

5. Batch correction

 Here we perform our own batch correction on the SMG subset of the data, correcting first at the dataset level and then at the donor level. The result is a series of UMAPS colored by variables of interest and the corresponding density plots of the computed LISI scores. This is saved as Batch_corr.png and corresponds to Supplementary Figure 3 in the manuscript.


