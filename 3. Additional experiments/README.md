### Description of notebooks and analyses

These are additional experiments that are referenced in the main text but are not part of the main classification experiment.

1. Verify donor-based bias in naive setting

This is done to check that donor bias indeed affects classifier performance in the naive setting, by excluding some of the donors from training and evaluating performance separately on them. 

The classification is done as in the main experiment, except the training and prediction function includes a donor exclusion step. Since we only perform this for one random seed, we run this experiment locally and not on the cluster. The resulting plots are saved in the verif_plots folder.

2. Abundance in naive setting (CD4 fixing)

Here we fix the sample counts of one cell type, the CD4 T-cells, in the naive setting. As in the main classification experiment, we first generate classification scripts, then we run them on a cluster, then we analyze the results. The template generation step results in the cd4_fix_classif folder; once the scripts in here are run on the cluster, this also includes classification results. Analysis is performed as in the main experiment (template generation - determining proportions - check general classification trend - check classification trend for individual cell types, in this case just CD4). The results are not saved explicitly in a results folder but can be retrieved from the notebook; they constitute Supplementary Figure 8 and Supplementary Table 3.

3. Abundance in donor-based setting

Here we fix the sample counts of Monocyte-derived Mph cells and Suprabasal cells, in the donor-based setting. As above, this is performed on the cluster and then analyzed. Note that we perform this at annotation level 4 for Suprabasal cells and at the finest annotation level for Monocyte-derived Mphs. As above, the results are not saved explicitly but can be retrieved from the notebook and constitute the third column of Supplementary Figure 6.