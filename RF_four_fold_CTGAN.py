"""
In this version of the RF model, we use CTGAN augmentation and four folds to more fairly evaluate the grapevine dataset. 
Each fold is augmented in a way that creates roughly equal amounts of diseased and healthy datapoints.
The test set is left untouched, and the RF model evaluates the new training data.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from prettytable import PrettyTable 
import ctgan
from ctgan import CTGAN
from pandas import DataFrame
from datetime import datetime
import csv
from statistics import mean


# best result for CTGAN
num_epochs = 25

# number of synthetic samples to create so that dataset size is the same as the SASD experiment 
# overcompensate diseased samples, so that classes are balanced
num_healthy_samples = 163
num_diseased_samples = 187

# path to csv for results
results_file_path = "Results.csv"

discrete_columns = [0,1,2,3,4]


# specify the Column Names while initializing each result table 
table = PrettyTable(["Fold", "Mean Accuracy", "Mean Precision", "Mean Recall", "Mean f1", "Mean OOB"]) 
class_table = PrettyTable(["Fold","Diseased in train", "Healthy in train", "Diseased in test", "Healthy in test"]) 
Confusion_Matrix = PrettyTable(["OOB Fold","True Positive", "True Negative", "False Positive", "False Negative"])
Acc_Confusion_Matrix = PrettyTable(["Accuracy Fold","True Positive", "True Negative", "False Positive", "False Negative"])
importances_table = PrettyTable(["Fold", "Mean NDRE importance", "Mean CHM importance", "Mean LAI importance", "Mean DTM importance"])



fold_rows = []


for i in range(4):

    print(f"Fold {i} in progress...")



    # access train and test data according to fold so it's consistent with other models
    X_train = pd.read_csv(f"Augmentation//TrainSetNoAug{i}.csv").iloc[:,1:5]
    y_train = pd.read_csv(f"Augmentation//TrainSetNoAug{i}.csv").iloc[:,5:]

    X_test = pd.read_csv(f"Augmentation//TestSet{i}.csv").iloc[:,1:5]
    y_test = pd.read_csv(f"Augmentation//TestSet{i}.csv").iloc[:,5:]


    # combine train data for CTGAN
    train_data = pd.read_csv(f"Augmentation//TrainSetNoAug{i}.csv").iloc[:,1:].to_numpy()
    print(f"here is the train data: {train_data}")
    combined_test = pd.read_csv(f"Augmentation//TestSet{i}.csv").iloc[:,1:].to_numpy()

    
    # separate based on vigour, so that the quantity of each class can be controlled
    healthy_train_data = train_data[train_data[:,-1]==0.0]
    diseased_train_data = train_data[train_data[:,-1]==1.0]

    # initialize both classes
    ctgan_healthy = CTGAN(epochs=num_epochs)
    ctgan_diseased = CTGAN(epochs=num_epochs)

    # create healthy and diseased synthetic data
    ctgan_healthy.fit(train_data=healthy_train_data, discrete_columns=discrete_columns)
    ctgan_diseased.fit(train_data=diseased_train_data, discrete_columns=discrete_columns)
    healthy_synthetic_data = ctgan_healthy.sample(num_healthy_samples)
    diseased_synthetic_data = ctgan_diseased.sample(num_diseased_samples)

    # separate labels 
    X_healthy_train_syn = healthy_synthetic_data[:,:4] 
    y_healthy_train_syn = healthy_synthetic_data[:,-1:]

    X_diseased_train_syn = diseased_synthetic_data[:,:4] 
    y_diseased_train_syn = diseased_synthetic_data[:,-1:]



    # combine real X and y train data with synthetic data - test does not change
    X_train = np.vstack([X_train, X_healthy_train_syn, X_diseased_train_syn])
    y_train = np.vstack([y_train, y_healthy_train_syn, y_diseased_train_syn])

    updated_train = np.hstack([X_train, y_train])

    y_test = y_test.to_numpy().ravel()
    y_train = y_train.ravel()
    

    # convert array into dataframe
    DF_train = DataFrame(updated_train)
    # DF_test = DataFrame(combined_test)


    # save the dataframe as a csv file
    DF_train.to_csv(f"AugmentedTrainSetCTGAN{i}.csv")


    # set accuracy, precision, count, recall, f1 to zero so metrics mean can be calculated
    accuracy, precision, count, recall, f1, oob = 0, 0, 0, 0, 0, 0


    # initialize to zero for oob confusion matrix table later
    oob_true_neg, oob_true_pos, oob_false_neg, oob_false_pos = 0, 0, 0, 0


    # initialize to zero for accuracy confusion matrix table later
    acc_true_neg, acc_true_pos, acc_false_neg, acc_false_pos = 0, 0, 0, 0

    # initialize to zero to track feature importances
    NDRE_importance, CHM_importance, LAI_importance, DTM_importance = 0, 0, 0, 0


    for j in range(10):

        # 500 estimators and oob_score is True - same as Velez et al.
        rf = RandomForestClassifier(n_estimators=500, oob_score=True)

        # train RF model
        rf.fit(X_train, y_train)

        # Obtain the OOB error
        oob_error = 1 - rf.oob_score_
        oob_pred = np.argmax(rf.oob_decision_function_, axis=1)

        # test RF on test set
        y_pred = rf.predict(X_test)


        #record metrics 
        # uncomment prints to track specific metrics

        accuracy_result = accuracy_score(y_test, y_pred)
        # print(f"accuracy: {accuracy_result}")

        precision_result = precision_score(y_test, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')
        # print(f"precision: {precision_result}")

        recall_result = recall_score(y_test, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')
        # print(f"recall result: {recall_result}")

        f1_result = f1_score(y_test, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')
        # print(f"f1 score: {f1_result}")

        # confusion matrix based on OOB - uncomment for use
        # not good practice, but the same as Velez et al. for the sake of comparison
        # oob_tn, oob_fp, oob_fn, oob_tp = confusion_matrix(y_train, oob_pred, labels=None, sample_weight=None, normalize=None).ravel().tolist()
        
        # confusion matrix based on accuracy
        # acc_tn, acc_fp, acc_fn, acc_tp = confusion_matrix(y_test, y_pred, labels=None, sample_weight=None, normalize=None).ravel().tolist()



        # get feature importances for this run and record them for averaging later
        NDRE, CHM, LAI, DTM = rf.feature_importances_.ravel()
        NDRE_importance += NDRE
        CHM_importance += CHM
        LAI_importance += LAI
        DTM_importance += DTM

        # do the same for other metrics
        accuracy += accuracy_result
        precision += precision_result
        recall += recall_result
        f1 += f1_result
        oob += oob_error
        count += 1

    

    # add results for each fold to each table
    table.add_row([i, accuracy/count, precision/count, recall/count, f1/count, oob/count, ]) 
    class_table.add_row([i, np.sum(y_train == 1.0), np.sum(y_train == 0.0), np.sum(y_test == 1.0), np.sum(y_test == 0.0)]) 
    # Confusion_Matrix.add_row([i, oob_true_pos/count, oob_true_neg/count, oob_false_pos/count, oob_false_neg/count]) 
    # Acc_Confusion_Matrix.add_row([i, acc_true_pos/count, acc_true_neg/count, acc_false_pos/count, acc_false_neg/count])
    importances_table.add_row([i, NDRE_importance/count, CHM_importance/count, LAI_importance/count, DTM_importance/count])


    # update csv row
    row = {
        "Date": datetime.today().date().isoformat(),
        "Time": datetime.now().strftime("%H:%M"),
        "Model": "RF with augmentation revisit",
        "Fold": i,
        "Accuracy": accuracy/count,
        "Precision": precision/count,
        "Recall": recall/count,
        "F1": f1/count,
        "OOB": oob/count,
        "Diseased in Train": np.sum(y_train == 1.0),
        "Healthy in Train": np.sum(y_train == 0.0),
        "Diseased in Test": np.sum(y_test == 1.0),
        "Healthy in Test": np.sum(y_test == 0.0),
        # "True Positive (acc)": acc_tp,
        # "True Negative (acc)": acc_tn,
        # "False Positive (acc)": acc_fp,
        # "False Negative (acc)": acc_fn,
        # "True Positive (oob)": oob_tp, # only used for RF, just keeping the space for the CSV
        # "True Negative (oob)": oob_tn,
        # "False Positive (oob)": oob_fp,
        # "False Negative (oob)": oob_fn,
    }

    with open(results_file_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        # Only write header if file is empty
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow(row)
    
    fold_rows.append(row)


# print tables to terminal
print(table)
print(class_table)
# print(Confusion_Matrix)
# print(Acc_Confusion_Matrix)
print(importances_table)


# calculate means for final row
mean_fields = [
    "Accuracy", "Precision", "Recall", "F1", "OOB",
    # "True Positive (acc)", "True Negative (acc)",
    # "False Positive (acc)", "False Negative (acc)", 
    # "Diseased in Train", "Healthy in Train",
    # "Diseased in Test", "Healthy in Test",
]

mean_row = {
    "Date": datetime.today().isoformat(),
    "Time": datetime.now().strftime("%H:%M"),
    "Model": "RF",
    "Fold": "Mean",
}


for field in mean_fields:
    mean_row[field] = mean(float(r[field]) for r in fold_rows)




# OOB confusion matrix - change values and uncomment if using
# mean_row["True Positive (oob)"] = "N/a"
# mean_row["True Negative (oob)"] = "N/a"
# mean_row["False Positive (oob)"] = "N/a"
# mean_row["False Negative (oob)"] = "N/a"

with open(results_file_path, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=mean_row.keys())
    writer.writerow(mean_row)
