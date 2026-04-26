### Description of notebooks

#### 1. Preliminary exploration of HLCA core
This includes:
 * basic visualizations of the HLCA core
 * generation of the cell_type_info.pickle file - this is a dictionary that describes, for each cell type, the total sample count, the proportion of female samples, the number of donors, the number of female donors and the number of datasets; this will be of use for later analyses
 * saves barplot of sex distribution across constituent datasets of the HLCA core (Dataset_Sex_Distrib.png; this is Figure 1a in the manuscript)

 Results are stored in the folder Exploration_results.

#### 2. Differential abundance analysis of HLCA core
This contains the entire differential abundance analysis. Relevant results are:
 * generation of diagnostic plots (diagnostic_plots.png, nhood_annotation_fraction.png, nhood_sizes.png; these make up Supplementary Figure 1 in the manuscript)
 * visualization of differential abundance results on the UMAP (DA_plot_5e-2.png; this is Figure 1b in the manuscript)
 * update of the cell_type_info.pickle dictionary into cell_type_info_with_DA.pickle that also includes the number and proportion of differentially abundant neighborhoods for each cell type
 * visualization of results by cell type in violin plots, combined with information of number of cells, number of individuals and number of datasets (Combined_Celltypes_DA.png; Figure 1c in the manuscript)
 * table with DA diagnostics for each cell type: median log fold-change, total number of neighborhoods, total number of significant female and male neighborhoods, enrichment calculated based on log fold-change cutoff (balance_table_body.txt; Supplementary Table 1 in the manuscript)
 * barplots of sex and donor distribution for each cell type, where cell types are ordered the same as in the violin plot (Celltype_Sex_and_TopDonor_Distrib.png; Supplementary Figure 2 in the manuscript)
 
 Results are stored in the folder DA_results.

#### 3. Exploration of SMG population
 This is a more in-depth exploration of just the submucousal secretory gland cell population in the HLCA core. We investigate cell type composition, dataset distribution and donor distribution. The result is SMGs.png, which corresponds to Figure 2 in the manuscript, and is also saved in the folder Exploration_results. 