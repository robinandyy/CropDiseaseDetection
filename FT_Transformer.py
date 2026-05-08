"""
Introduces FT-Transformer into crop disease detection in order to consider interdependencies between features.
"""


import numpy as np
import pandas as pd
import torch

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import csv
from datetime import datetime
from statistics import mean
from tqdm.auto import tqdm
from torch import nn
from rtdl_revisiting_models import FTTransformer
from prettytable import PrettyTable
from copy import deepcopy


# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)


# save the results
results_file_path = "Results.csv"

table = PrettyTable(["Fold", "Mean Accuracy", "Mean Precision", "Mean Recall", "Mean f1", "Mean OOB"]) 
# class_table = PrettyTable(["Fold","Diseased in train", "Healthy in train", "Diseased in test", "Healthy in test"]) 
# Confusion_Matrix = PrettyTable(["OOB Fold","True Positive", "True Negative", "False Positive", "False Negative"])
Acc_Confusion_Matrix = PrettyTable(["Accuracy Fold","True Positive", "True Negative", "False Positive", "False Negative"])


def train_step(model: torch.nn.Module,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device,
               epoch: int):
    


    # set accuracy, precision, count, recall, f1 to zero so metrics mean can be calculated
    train_accuracy, train_loss, count, = 0, 0, 0
    


    model.train()

    for i in range(4):
              
        count = i + 1

        # retrieve train and test data + put tensors to device
        X_train = pd.read_csv(f"TrainSet{i}.csv").iloc[:,1:5]
        X_train = torch.tensor(X_train.values, dtype=torch.float32)

        y_train = pd.read_csv(f"TrainSet{i}.csv").iloc[:,5:]
        y_train = torch.tensor(y_train.values, dtype=torch.float32).squeeze()

        X_train, y_train = X_train.to(device), y_train.to(device)



        # 1. Forward pass
        y_pred = model(X_train, None)
        y_pred = y_pred.squeeze()
        y_pred_binary = (torch.sigmoid(y_pred) >= 0.5).float()

        

        # 2. Calculate loss
        loss = loss_fn(y_pred, y_train)
        train_loss += loss
        train_accuracy += accuracy_fn(y_true=y_train.detach().cpu().numpy(),
                                 y_pred=y_pred_binary.detach().cpu().numpy()) # Go from logits -> pred labels

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()



    return train_loss/count, train_accuracy/count 


def test_step(model: torch.nn.Module,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              precision_fn,
              recall_fn,
              f1_fn,
              device: torch.device,
              epoch: int,
              fold_rows: int):
    

    # set accuracy, precision, count, recall, f1 to zero so metrics mean can be calculated
    total_test_loss, total_test_acc, test_acc, test_loss, precision, count, recall, f1, oob = 0, 0, 0, 0, 0, 0, 0, 0, 0

    # initialize to zero for accuracy confusion matrix table later
    acc_true_neg, acc_true_pos, acc_false_neg, acc_false_pos = 0, 0, 0, 0
    model.to(device)
    model.eval() 


    with torch.inference_mode(): 
        for i in range(4):

            X_test = pd.read_csv(f"TestSet{i}.csv").iloc[:,1:5]
            y_test = pd.read_csv(f"TestSet{i}.csv").iloc[:,5:]

            X_test = torch.tensor(X_test.values, dtype=torch.float32)
            y_test = torch.tensor(y_test.values, dtype=torch.float32).squeeze()


            # Send data to GPU
            X_test, y_test = X_test.to(device), y_test.to(device)
            
            # 1. Forward pass
            test_pred = model(X_test, None)
            test_pred = test_pred.squeeze()
            test_pred_binary = (torch.sigmoid(test_pred) >= 0.5).float()
            
            # 2. Calculate loss and accuracy
            test_loss = loss_fn(test_pred, y_test)
            test_acc = accuracy_fn(y_true=y_test.detach().cpu().numpy(),
                y_pred=test_pred_binary.detach().cpu().numpy())
            
            precision_result = precision_fn(y_test.detach().cpu().numpy(), test_pred_binary.detach().cpu().numpy(), labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')
            # print(f"precision: {precision_result}")

            recall_result = recall_fn(y_test.detach().cpu().numpy(), test_pred_binary.detach().cpu().numpy(), labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')
            # print(f"recall result: {recall_result}")

            f1_result = f1_fn(y_test.detach().cpu().numpy(), test_pred_binary.detach().cpu().numpy(), labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')
            # print(f"f1 score: {f1_result}")


            precision += precision_result
            recall += recall_result
            f1 += f1_result
            count += 1


            # confusion matrix based on accuracy
            acc_tn, acc_fp, acc_fn, acc_tp = confusion_matrix(y_test, test_pred_binary, labels=None, sample_weight=None, normalize=None).ravel().tolist()

            # normalize
            acc_true_neg += acc_tn/(acc_tn + acc_fp + acc_fn + acc_tp) * 100
            acc_false_pos += acc_fp/(acc_tn + acc_fp + acc_fn + acc_tp) * 100
            acc_false_neg += acc_fn/(acc_tn + acc_fp + acc_fn + acc_tp) * 100
            acc_true_pos += acc_tp/(acc_tn + acc_fp + acc_fn + acc_tp) * 100

        
            # Adjust metrics and print out
            total_test_loss +=  test_loss
            total_test_acc += test_acc


            print(f"Test loss: {total_test_loss/count:.5f} | Test accuracy: {total_test_acc/count:.2f}%\n")
            print(f"Precision: {precision/count} | Recall: {recall/count} | F1: {f1/count}")

            table.add_row([i, test_acc, precision_result, recall_result, f1_result, 0]) 
            # class_table.add_row([i, np.sum(y_train == 1.0), np.sum(y_train == 0.0), np.sum(y_test == 1.0), np.sum(y_test == 0.0)]) 
            Acc_Confusion_Matrix.add_row([i, acc_true_pos/count, acc_true_neg/count, acc_false_pos/count, acc_false_neg/count])

            row = {
                "Date": datetime.today().isoformat(),
                "Time": datetime.now().strftime("%H:%M"),
                "Model": "FT Transformer",
                "Fold": i,
                "Accuracy": test_acc,
                "Precision": precision_result,
                "Recall": recall_result,
                "F1": f1_result,
                "OOB": 0,
                "Diseased in Train": 666,
                "Healthy in Train": 666,
                "Diseased in Test": 666, #np.sum(y_test == 1.0),
                "Healthy in Test": 666,
                "True Positive (acc)": acc_tp/(acc_tn + acc_fp + acc_fn + acc_tp) * 100,
                "True Negative (acc)": acc_tn/(acc_tn + acc_fp + acc_fn + acc_tp) * 100,
                "False Positive (acc)": acc_fp/(acc_tn + acc_fp + acc_fn + acc_tp) * 100,
                "False Negative (acc)": acc_fn/(acc_tn + acc_fp + acc_fn + acc_tp) * 100,
                "True Positive (oob)": 0, # only used for RF, just keeping the space for the CSV
                "True Negative (oob)": 0,
                "False Positive (oob)": 0,
                "False Negative (oob)": 0,
                "Learning Rate: ": 3e-4,
                "Threshold": 0.5,
            }


            with open(results_file_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                # Only write header if file is empty
                if f.tell() == 0:
                    writer.writeheader()
                writer.writerow(row)
            
            fold_rows.append(row)


        # print(table)
        # # print(class_table)
        # print(Acc_Confusion_Matrix)



        mean_fields = [
            "Accuracy", "Precision", "Recall", "F1", "OOB",
            "True Positive (acc)", "True Negative (acc)",
            "False Positive (acc)", "False Negative (acc)", 
            "Diseased in Train", "Healthy in Train",
            "Diseased in Test", "Healthy in Test",
        ]

        mean_row = {
            "Date": datetime.today().isoformat(),
            "Time": datetime.now().strftime("%H:%M"),
            "Model": "FT Transformer",
            "Fold": "Mean",
        }

        # Compute means
        for field in mean_fields:
            mean_row[field] = mean(float(r[field]) for r in fold_rows)




        # FT Transformer has no OOB confusion matrix
        mean_row["True Positive (oob)"] = 0
        mean_row["True Negative (oob)"] = 0
        mean_row["False Positive (oob)"] = 0
        mean_row["False Negative (oob)"] = 0

        with open(results_file_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=mean_row.keys())
            writer.writerow(mean_row)




def main():
    d_out = 1  
    lr = 0.0007725879644486419
    weight_decay = 5.936058071331039e-05
    dropout = 0.12237927871962775
    attention_heads = 4 
    transformer_layers = 6 
    embedding_dimension = 128 



    loss_fn = nn.BCEWithLogitsLoss()
    fold_rows = []
    epochs = 250
    best_state = None
    best_train_loss = float("inf")



    BBR_model = FTTransformer(
        n_cont_features=4,
        cat_cardinalities=[],
        d_out=d_out,
        n_blocks=transformer_layers,
        d_block=embedding_dimension,
        attention_n_heads=attention_heads,
        attention_dropout=dropout,
        ffn_d_hidden=None,
        ffn_d_hidden_multiplier=4 / 3,
        ffn_dropout=0.1,
        residual_dropout=0.0,
    )


    optimizer = torch.optim.AdamW(
        # Instead of model.parameters(),
        BBR_model.make_parameter_groups(),
        lr=lr,
        weight_decay=weight_decay,
    )

    for epoch in tqdm(range(epochs)):

        # implement
        train_loss, train_accuracy = train_step(model=BBR_model, 
                loss_fn=loss_fn,
                optimizer= optimizer,
                accuracy_fn=accuracy_score,
                device="cuda" if torch.cuda.is_available() else "cpu",
                epoch=epoch)

        
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            best_epoch = epoch
            best_state = deepcopy(BBR_model.state_dict())


        if epoch % 10 == 0:
            print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_accuracy:.2f}%")
            print(f"Best epoch: {best_epoch}")
        
    test_step(model=BBR_model,
            loss_fn=loss_fn,
            accuracy_fn=accuracy_score,
            precision_fn=precision_score,
            recall_fn=recall_score,
            f1_fn=f1_score,
            device="cuda" if torch.cuda.is_available() else "cpu",
            epoch=epoch, 
            fold_rows=fold_rows)
        
        
main()
