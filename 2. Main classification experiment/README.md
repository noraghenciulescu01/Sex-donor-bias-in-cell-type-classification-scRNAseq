### Description of analyses and notebooks

We use two classification settings: naive and donor-based. For each setting, we have 4 notebooks. These are almost identical across the two settings; the main difference is that the train and prediction function includes donor filtering in the data splitting step for the donor-based setting. 

#### 1. Template generation
 Here we create the classification scripts, which are not run locally but on an AI cluster; this is why we generate both .py scripts and .sh SLURM job submission files. For each setting, this notebook generates a folder, classif_donorbased or classif_naive, with the following structure:
 * one sub-folder for each random seed used (17, 42, 60, 83)
 * within each random seed folder, one sub-folder for each annotation level (2, 3, 4, finest)
 * within each annotation folder, one sub-folder for each classifier used (KNN or RF)

 For each classifier, at each annotation level and for each random seed, we generate one classification script (and a corresponding SLURM job submission file) for each proportion of female cells in the training set. For instance, for KNN, we have knn_0.py (and knn_0.sh) for the training set with 0% female cells, knn_01.py (knn_01.sh) for the training set with 10% female cells and so on, up to knn_1.py (knn_1.sh) for the 100% female training set. Thus, we have 11 .py files (and 11.sh files) in total. Within the same sub-folder, the files only differ in the prop parameter, which controls the proportion of female cells in the training set. Across different sub-folders, they differ based on the classifier, the annotation level and the random seed used. The .py scripts are called by the .sh SLURM files when a job is submitted. 

 Additionally, we generate a helper_functions.py file for every sub-folder. This is the same across all sub-folders and contains the actual train and predict function, as well as the evaluation functions.

 Once the classif_naive and classif_donorbased folders are generated, **these are run separately on the cluster**, by transfering the full folder to the cluster and running each individual .sh job submission file. Within the same sub-folders that the scripts are located in, this will generate the following **classification results**:
 * a pickle file of metrics for the male test set and one for the female test set (so 0_male_metrics.pickle and 0_female_metrics.pickle are the results of the classifier trained on the 0% female training set on the male and female test set, respectively); this file includes multiple evaluation metrics: accuracy, F1 scores, precision scores, median F1 score, median precision score, the confusion matrix and the normalized confusion matrix
 * the confusion matrices and the normalized confusion matrices are additionally saved as .png files in the cms and norm_cms folders, respectively

 If you simply run the template generation notebook, this will generate the classif_naive and classif_donorbased folders without the corresponding classification results. The classif_naive and classif_donorbased folders as included in this repo are the folders **including these results**, so after running the scripts on the cluster. This needs to be done manually for reproduction. The rest of the notebooks can be run on the provided results.
 
#### 2. Determine train & test proportions
 This is mostly a verification notebook - we look at how much the cell type proportions change as we vary the proportion of female cells in the training set, to ensure that every cell type has at least some representation in the training and test set. For the donor-based setting, we additionally check whether there is in fact no overlap between train and test donors.

 This notebook additionally generated the proportion_dictionary.pickle files (naive_proportion_dictionary.pickle and donorbased_proportion_dictionary.pickle). This is a dictionary that gives the number of cells in the training set, in the male test set and in the female test set, for all annotation levels and all proportions of female cells in the training set. These are saved in the helper_pickle_files folder and will be of use in notebook 4.

#### 3. Overall result analysis
 This is the main analysis of the classification results. It gathers the evaluation metrics (accuracy and F1 scores) from the classif_naive and classif_donorbased folders and plots the performance of the classifier across female proportions and annotation levels. These plots are saved to the results_naive and results_donorbased folders and constitute Main Figure c and d. 
 
#### 4. Analysis per cell type
 In the final notebook, we analyze the performance of the classifier for each individual cell type. We again use the classif_naive and classif_donorbased folders to retrieve evaluation metrics (confusion matrices). It also makes use of the proportion_dictionary.pickle files generated in notebook 2. The results are:
 * a line plot and a heat map that show the classification trend for each cell type; these plots are not saved explicitly into the results folders, but specific plots are retrieved to make Main Figure 3, the first two columns of Supplementary Figure 6, Supplementary Figure 7 and Supplementary Figure 9
 * a df_classification_trend.pickle file which is saved into the results_naive or results_donorbased folder; these are Supplementary Tables 1 and 2

#### helper_pickle_files folder
 This includes helpful pickle files for the analysis. The donorbased_proportion_dictionary.pickle and naive_proportion_dictionary.pickle are generated in notebook 2. for each setting. The generation of cell_names.pickle and color_mappings.pickle is not included in this repo, but these are simple dictionaries of the cell type names at each annotation level and corresponding colors, to ensure consistent plotting across the experiments and can be used as given.