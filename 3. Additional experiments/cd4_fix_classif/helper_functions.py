# Helper functions

import numpy as np
import pandas as pd
import scanpy as sc
import anndata
import matplotlib.pyplot as plt
from scipy.sparse import vstack

from sklearn.model_selection import StratifiedKFold, train_test_split, GroupShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix, precision_score
import seaborn as sns


# Train and evaluate functions:

def train_clf_and_predict(X, y, sex_labels, individual_labels, proportion_female, classifier='knn', k = 30):
    '''
    Modified version of the train_clf_and_predict_equaltest function that does not actually train/predict anything;
    it only selects the training set according to proportion_female and returns it.
    We fix the CD4 sub-population in the training set by calling fixed_select_indices_by_proportion separately on CD4 and non-CD4 samples.

    Since we are only interested in the CD4 cell type, we only define this for ann_level_4 and one random seed.
    ---------
    
    Parameters:
    X = expression matrix; matrix of shape n_obs x n_vars
    y = (cell type) labels; array/list of shape n_obs
    sex_labels = 'male' or 'female' label for each entry; array/list of shape n_obs
    individual_labels = donor id for each entry; array/list of shape n_obs
    proportion_female = desired proportion of female cells; float between 0 and 1
    classifier = 'knn' or 'rf'; default 'knn'
    k = number of neighbors if using knn; default 30
    ----------
    
    Returns:

    male_pred, female_pred = arrays of shape n_obs; contains the prediction on the male 
                                or female test set
    y_male_test, y_female_test = arrays of shape n_obs; contains the true labels of the 
                                male or female test set
    
    '''
    
    np.random.seed(83)
    
    
    male_indices = np.where(sex_labels == 'male')[0]
    female_indices = np.where(sex_labels == 'female')[0]

    X_male = X[male_indices]
    y_male = y[male_indices]
    X_female = X[female_indices]
    y_female = y[female_indices]
    
    X_female_train, X_female_test, y_female_train, y_female_test = train_test_split(
        X_female, y_female, test_size=0.2, stratify=y_female, random_state=83)
    
    # compute what to pass to test_size to get equal test set size to the female set
    male_proportion = X_female_test.shape[0] / X_male.shape[0]

    X_male_train, X_male_test, y_male_train, y_male_test = train_test_split(
        X_male, y_male, test_size=male_proportion, stratify=y_male, random_state=83)
    

    # merge training sets back together
    X_train = vstack([X_male_train, X_female_train])
    y_train = np.concatenate([y_male_train, y_female_train])
    sex_labels_train = ['male'] * X_male_train.shape[0] + ['female'] * X_female_train.shape[0]

    # Select female cells based on proportion_female

    ## extract CD4a
    cd4_mask = (y_train == 'CD4 T cells')
    non_cd4_mask = ~cd4_mask

    ## call fixed_select_indices_by_proportion on the CD4 and non-CD4 samples separately 
    selected_non_cd4_indices = fixed_select_indices_by_proportion([sex_labels_train[i] for i in np.where(non_cd4_mask)[0]], proportion_female)
    selected_cd4_indices = fixed_select_indices_by_proportion([sex_labels_train[i] for i in np.where(cd4_mask)[0]], proportion_female)
    
    ## then stitch them together
    selected_indices = np.concatenate((np.where(non_cd4_mask)[0][selected_non_cd4_indices], np.where(cd4_mask)[0][selected_cd4_indices]))

    X_selected = X_train.tocsr()[selected_indices]
    y_selected = y_train[selected_indices]

    print('selected training!', flush = True)

    # Initialize classifier
    if classifier == 'knn':
        clf = KNeighborsClassifier(n_neighbors=k)
    elif classifier == 'rf':
        clf = RandomForestClassifier(n_jobs=-1)
        

    print('initialized classif!', flush = True)
    # Train
    clf.fit(X_selected, y_selected)
       
    print('done training!', flush = True)
    
    # Predict
    male_pred = clf.predict(X_male_test)
    female_pred = clf.predict(X_female_test)
     
    return male_pred, y_male_test, female_pred, y_female_test


def evaluate_clf(predictions, true_labels, classes, prop, sex):
    '''
    This is meant to be run separately on the male and female results of the train function.
    ---------
    
    Parameters:

    predictions = predictions on a test set
    true_labels = true labels on that test set
    classes = sorted list of classes
    prop = proportion of female cells
    sex = 'male' or 'female'; dneotes what test set we are working with
        # prop and sex are only used for naming the confusion matrix plots
    
    ----------
    
    Returns:

    accuracy = accuracy score
    f1_per_class = array of shape n_classes; each entry is the f1 score of that class
    median_f1 = median of the f1_per_class array
    precision_per_class = array of shape n_classes; each entry is the precision score of that class
    median_precision = median of the precision_per_class array
    cm = confusion matrix
    cm_normalized = normalized confusion matrix
        (the function saves plots of each confusion matrix)
    
    '''
    n_classes = len(classes)

    # Accuracy scores
    accuracy = accuracy_score(true_labels, predictions)

    # F1 per class
    f1_per_class = f1_score(true_labels, predictions, average=None)

    # Median F1
    median_f1 = np.median(f1_per_class)

    # Precision per class
    precision_per_class = precision_score(true_labels, predictions, average=None)

    # Median precision
    median_precision = np.median(precision_per_class)

    # Confusion matrix
    cm = confusion_matrix(true_labels, predictions, labels=classes)
    # Normalized confusion matrix:
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    
    # Create dictionary
    metrics = {
        'accuracy': accuracy,
        'f1_scores': f1_per_class,
        'median_f1': median_f1,
        'precision_scores': precision_per_class,
        'median_precision': median_precision,
        'aggregated_confusion_matrix': cm,
        'normalized_aggregated_confusion_matrix': cm_normalized
    }
        
        
    plot_confusion(cm, classes, f'Prop {prop}, {sex} test set Confusion Matrix', False)
    plot_confusion(cm_normalized, classes, f'Prop {prop}, {sex} test set Normalized Confusion Matrix', True)
    
    
    return metrics



# Helper functions:

def plot_confusion(confusion_matrix, classes, title, normalize = False):
    '''
    Plot and save the current confusion matrix.
    Make sure to pass a title and the normalize parameter (if the matrix is normalized).
    '''
    
    plt.clf()
    plt.figure(figsize=(10, 8))
    if normalize:
        sns.heatmap(confusion_matrix, annot=True, fmt=".3f", cmap="Blues", xticklabels=classes, yticklabels=classes)
    else:
        sns.heatmap(confusion_matrix, annot=True, fmt="g", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    if normalize:
        plt.savefig(f'norm_cms/{title}.png', bbox_inches='tight')
    else:
        plt.savefig(f'cms/{title}.png', bbox_inches='tight')
    plt.close()


def fixed_select_indices_by_proportion(sex_labels, proportion_female):
    np.random.seed(83)
    sex_labels_series = pd.Series( (el for el in sex_labels) )
    
    female_indices = np.where(sex_labels_series == 'female')[0]
    male_indices = np.where(sex_labels_series == 'male')[0]
    
    fixed_size = min(len(female_indices), len(male_indices))
    
    np.random.shuffle(female_indices)
    np.random.shuffle(male_indices)

    num_female_cells = int(fixed_size * proportion_female)
    num_male_cells = fixed_size - num_female_cells
        # total will always be fixed_size
        # this works for cases with prop 0% or 100% --> no need to handle them separately
    
    # adjust in case of rounding errors
    num_female_cells = min(num_female_cells, len(female_indices))
    num_male_cells = min(num_male_cells, len(male_indices))

    selected_female_indices = female_indices[:num_female_cells]
    selected_male_indices = male_indices[:num_male_cells]

    return np.concatenate([selected_female_indices, selected_male_indices])


def check_missing_classes_in_folds(predictions, true_labels, classes):
    '''
    Checks if there are folds that miss predictions or true labels for a class.
    The latter can happen if there are fewer samples of a certain class than folds.
    '''
    
    missing_info = [] 
    
    for fold_index, (y_pred, y_true) in enumerate(zip(predictions, true_labels)):
        unique_pred_classes = set(np.unique(y_pred))
        unique_true_classes = set(np.unique(y_true))
        all_classes_set = set(classes)

        missing_in_pred = all_classes_set - unique_pred_classes
        missing_in_true = all_classes_set - unique_true_classes

        if missing_in_pred or missing_in_true:
            missing_info.append({
                'fold': fold_index,
                'missing_in_predictions': sorted(list(missing_in_pred)), 
                'missing_in_true_labels': sorted(list(missing_in_true))
            })

    if missing_info:
        for info in missing_info:
            print(f"Fold {info['fold']} is missing predictions for classes: {info['missing_in_predictions']}")
            print(f"Fold {info['fold']} is missing true labels for classes: {info['missing_in_true_labels']}")
    else:
        print("All folds have predictions and true labels for all classes.")

