### Description of notebooks

#### 1. Preliminary exploration of HLCA core
This includes:
 * basic visualizations of the HLCA core
 * generation of the cell_type_info.pickle file - this is a dictionary that describes, for each cell type, the total sample count, the proportion of female samples, the number of donors, the number of female donors and the number of datasets; this will be of use for later analyses
 * creation of Main Figure 1a which describes the sex distribution in all constituent datasets of the HLCA core

 Results are stored in the folder Exploration_results.

 #### 2. Differential abundance analysis of HLCA core
This contains the entire differential abundance analysis. Relevant results are:
 * generation of diagnostic plots for Supplementary Figure 1
 * visualization of differential abundance results on the UMAP (Main Figure 1b)
 * update of the cell_type_info.pickle dictionary into cell_type_info_with_DA.pickle that also includes the number and proportion of differentially abundant neighborhoods for each cell type
 * visualization of results by cell type in violin plots, combined with information of number of cells, number of individuals and number of datasets using the cell_type_info.pickle dictionary (Main Figure 1c)

  Results are stored in the folder DA_results.

 #### 3. Exploration of SMG population
This is a more in-depth exploration of just the submucousal secretory gland cell population in the HLCA core. We investigate cell type composition, dataset distribution and donor distribution. The result is Main Figure 1d and is also saved in the folder Exploration_results. 