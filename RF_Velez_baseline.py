import numpy as np
import matplotlib.pyplot as plt
from ctgan import CTGAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



# create numpy arrays X and y
X = np.genfromtxt('Augmentation\\vineyard_data.csv', skip_header=1, usecols=(0,1,2,3), delimiter=',')
y = np.genfromtxt('Augmentation\\vineyard_data.csv', skip_header=1, usecols=(4), delimiter=',').reshape(150,1)


# create test train split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42)

y_train = y_train.ravel()
y_test = y_test.ravel()


# set accuracy, precision, count, recall, f1 to zero so metrics mean can be calculated
accuracy, precision, count, recall, f1 = 0, 0, 0, 0, 0


for i in range(50):

    # 500 estimators and oob_score is True - same as Velez et al.
    # class weight balanced to increase recall - more weight on diseased class
    rf = RandomForestClassifier(n_estimators=500, class_weight='balanced', oob_score=True)
    rf.fit(X_train, y_train)

    # threshold set to probability of 0.39 instead of 0.5 
    # more weight given to potentially diseased grapevines to improve recall
    threshold = 0.39  
    y_probs = rf.predict_proba(X_test)[:, 1]
    y_pred = (y_probs >= threshold).astype(int)

    #record metrics and print
    accuracy_result = accuracy_score(y_test, y_pred)
    print(f"accuracy: {accuracy_result}")

    precision_result = precision_score(y_test, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')
    print(f"precision: {precision_result}")

    recall_result = recall_score(y_test, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')
    print(f"recall result: {recall_result}")

    f1_result = f1_score(y_test, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')
    print(f"f1 score: {f1_result}")
    
    accuracy += accuracy_result
    precision += precision_result
    recall += recall_result
    f1 += f1_result
    count += 1

print(f"Mean accuracy: {accuracy/count} | Mean precision: {precision/count} \n | Mean recall: {recall/count} | Mean F1 Score: {f1/count} ")
