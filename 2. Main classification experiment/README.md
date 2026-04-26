### Description of analyses and notebooks

We use two classification settings: **naive** and **donor-based**. For each setting, we have 4 notebooks. These are almost identical across the two settings, except that in the donor-based setting the train + prediction function includes donor filtering in the data splitting step.  

#### 1. Template generation
 Here we create the classification scripts. We classify across 4 random seeds, 2 classifiers, 4 annotation levels and 11 proportions of female cells in the training set. Some of these scripts take a long time to run, thus we generate one separate script for each combination of seed x classifier x annotation level x proportion. We run these on an AI cluster using SLURM manager, thus we generate both .py scripts and SLURM .sh job submission files which call the .py scripts. 

 For each classification setting (naive or donor-based), this notebook generates a folder including all of these files: classif_naive for the naive setting and classif_donorbased for the donor-based setting. They both have the following structure:
 * one sub-folder for each random seed used (17, 42, 60, 83)
 * within each random seed folder, one sub-folder for each annotation level (2, 3, 4, finest)
 * within each annotation folder, one sub-folder for each classifier used (KNN or RF)
 
 Thus:
 ```text
    classif_naive/ or classif_donorbased/
    ├── seed17/          # random seed
    │   ├── ann_2/      # annotation level
    │   │   ├── knn_classif/      # classifier
    │   │   └── rf_classif/
    │   ├── ann_3/
    │   ├── ann_4/
    │   └── ann_finest/
    ...
 ```

 Within the sub-sub-subfolders corresponding to each classifier, we generate one .py file (and one corresponding .sh file) for each proportion of female cells in the training set. So, for instance in the knn_classif sub-subfolder, we generate: knn_0.py (and knn_0.sh) for the training set with 0% female cells, knn_01.py (knn_01.sh) for the training set with 10% female cells and so on, up to knn_1.py (knn_1.sh) for the 100% female training set. Thus, we have 11 .py files (and 11.sh files) in total, for both the KNN and the RF classifiers, for all annotation levels and seeds. Within the same folder (e.g. all files in seed17/ann_2/knn_classif), the files only differ in the prop parameter, which controls the proportion of female cells in the training set. Across different sub-folders, they differ based on the classifier, the annotation level and/or the random seed used.

 Additionally, we generate a helper_functions.py file for every sub-folder. This is the same across all sub-folders and contains the actual train and predict function, as well as the evaluation functions.

 Once the classif_naive and classif_donorbased folders are generated, **these have to be run separately on the cluster**, by transfering the full folder to the cluster and running each individual .sh job submission file. Within the same classifier sub-sub-subfolders that the scripts are located in, this will generate the following **classification results**:
 * a pickle file of metrics for the male test set and one for the female test set for each proportion of female cells in the training set (so, for instance, 0_male_metrics.pickle and 0_female_metrics.pickle are the results of the classifier trained on the 0% female training set on the male and female test set, respectively); this file includes multiple evaluation metrics: accuracy, F1 scores, precision scores, median F1 score, median precision score, the confusion matrix and the normalized confusion matrix
 * the confusion matrices and the normalized confusion matrices are additionally plotted and saved as .png files in the cms and norm_cms folders, respectively

 If you simply run the template generation notebook, this will generate the classif_naive and classif_donorbased folders without the corresponding classification results. However, in this repo, you find the classif_naive and classif_donorsbased folders **including the classification results**, i.e. after each script has been run on the cluster. These results are provided such that all other notebooks can be re-run for reproducibility. **For reproduction of the classification results themselves, running of each script would have to be done manually on a cluster.**

 Thus the final folders included here contain the following:
 ```text
 knn_classif/
    ├── cms/    # folder of confusion matrices as pngs
    ├── norm_cms/    # folder of normalised confusion matrices as pngs
    ├── 0_female_metrics.pickle    # results of classification task on the female test set, at 0% female proportion in training set 
    ├── 0_male_metrics.pickle    # results of classification task on the male test set, at 0% female proportion in training set 
    ... 
    ├── 1_female_metrics.pickle    # results of classification task on the female test set, at 100% female proportion in training set 
    ├── 1_male_metrics.pickle    # results of classification task on the male test set, at 100% female proportion in training set  
    ├── helper_functions.py    # helper function file
    ├── knn_0.py    # classification script for 0% female proportion in training set
    ├── knn_0.sh    # SLURM script that calls knn_0.py
    ...
    ├── knn_1.py    # classification script for 100% female proportion in training set
    ├── knn_1.sh    # SLURM script that calls knn_1.py
 ```
 
#### 2. Determine train & test proportions
 This is mostly a verification notebook - we look at how much the cell type proportions change as we vary the proportion of female cells in the training set, to ensure that every cell type has at least some representation in the training and test set. This is done using barplots but these are not saved or included in the manuscript.

 We additionally use this notebook to check for donor overlap. In the donor-based setting, we simply check that donor overlap is 0 across all proportions. In the naive setting, there is donor overlap - we generate a table that displays train counts, test counts and their ratio for each donor and each proportion of female cells in the training set. This is saved in results_naive as donor_overlap_body.txt and constitutes Supplementary Table 2 in the manuscript.

 Finally, in this notebook we additionally generate a proportion dictionary pickle file for each setting (naive_proportion_dictionary.pickle and donorbased_proportion_dictionary.pickle). This is a dictionary that includes the number of cells in the training set, in the male test set and in the female test set, for all annotation levels and all proportions of female cells in the training set. These are both saved in the helper_pickle_files folder and will be of use in notebook 4 for each setting.

#### 3. Overall result analysis
 This is the main analysis of the classification results. It gathers evaluation metrics (accuracy and F1 scores) from the classif_naive and classif_donorbased folders and plots the performance of the classifier across female proportions and annotation levels. These plots are saved to the results_naive and results_donorbased folders (accuracy and F1 results for KNN are named: knn_aggregated_acc.png, knn_f1_ann_2.png, ..., knn_f1_ann_finest.png; accuracy and F1 results for RF are named the same but with rf_ as a prefix instead of knn_). These constitute Figure 3 c, d and Supplementary Figures 4-7 in the manuscript. 
 
#### 4. Analysis per cell type
 In the final notebook, we analyze the performance of the classifier for each individual cell type. We again use the classif_naive and classif_donorbased folders to retrieve evaluation metrics (confusion matrices). It also makes use of the proportion_dictionary.pickle files generated in notebook 2. The main results are:
 * a line plot and a heat map that show the classification trend for each cell type
 * a table of results for each cell type, which includes the slope on the male test set, the slope on the female test set, the results of the Slope Test, the results of the Flip test, the maximum performance difference between male and female test set, and the size of the male and female test set

 Results are saved in results_naive and results_donorbased for each setting. The line plot + heat map plots are saved only for the cell types discussed explicitly in the manuscript. For both naive and donor-based settings: AT0, CD4 T-cells, Goblet (bronchial), Monocyte-derived Mph, Suprabasal. For the naive setting only: Alveolar Mph CCL3+, Alveolar Mph MT-positive, Lymphatic EC differentiating, Multiciliated (nasal). For several of these plots, we manually fix the y-axis to match with future analyses. All plots are saved as {cell type name}_classification_plot.png and constitute sub-plots in the manuscript for Supplementary Figures 8 - 13. The tables of results are saved for both the naive and the donor-based setting as classification_table_body.txt and constitute Supplementary Tables 3 and 4 in the manuscript.

#### helper_pickle_files folder
 This includes helpful pickle files for the analysis. The donorbased_proportion_dictionary.pickle and naive_proportion_dictionary.pickle are generated in notebook 2 for each setting. The generation of cell_names.pickle and color_mappings.pickle is not included in this repo, but these are simple dictionaries of the cell type names at each annotation level and corresponding colors, to ensure consistent plotting across the experiments and can be used as given.