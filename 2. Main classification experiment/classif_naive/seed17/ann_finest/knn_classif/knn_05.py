# KNN classification script

import numpy as np
import pandas as pd
import scanpy as sc
import anndata
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix
import seaborn as sns

print('successfully imported packages!', flush=True)

path = '/winhome/noraghenciules/hlca_data.h5ad'
adata = anndata.read_h5ad(path)

print('successfully read file!', flush=True)

embedding = adata.obsm['X_scanvi_emb']
sex_labels = adata.obs['sex']
cell_type_labels = adata.obs['ann_finest_level'].astype(str)
classes = sorted(list(set(cell_type_labels)))

from helper_functions_final import final_train_clf_and_predict, final_evaluate_clf, plot_confusion, fixed_select_indices_by_proportion, check_missing_classes_in_folds

# ----------
# Run on data increasing female proportion:
    # (results are printed and plots are saved)

prop = 0.5

print(f"PROPORTION OF FEMALE CELLS: {prop}", flush=True)
print('Training and testing...', flush=True)
male_pred, male_true_labels, female_pred, female_true_labels = final_train_clf_and_predict(embedding, cell_type_labels, sex_labels, prop)
print('Evaluating...', flush=True)
male_metrics = final_evaluate_clf(male_pred, male_true_labels, classes, prop, 'male')
female_metrics = final_evaluate_clf(female_pred, female_true_labels, classes, prop, 'female')

# File path to save the dictionary
male_file_path = f"{''.join(str(prop).split('.'))}_male_metrics.pickle"
female_file_path = f"{''.join(str(prop).split('.'))}_female_metrics.pickle"

# Save the dictionary to a file
with open(male_file_path, 'wb') as file:
    pickle.dump(male_metrics, file)

with open(female_file_path, 'wb') as file:
    pickle.dump(female_metrics, file)
