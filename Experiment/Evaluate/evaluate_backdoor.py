#Author: Muhammad Azizi Bin Mohd Ariffin
#Email: mazizi@fskm.uitm.edu.my
#Description: Evaluate Program for ISCX-VPN 2016 MLP Backdoor Attack

model_name = "adv2_backdoor_attemp2ok_global_model_weighted_fedavg_mlp_3client_36epochs_3round_cpu.h5"

#Confusion Matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import numpy as np
from tensorflow import keras

model = keras.models.load_model(model_name)

x_test = np.load("x_test-MLP-Multiclass-ISCX-740features.npy")
y_test = np.load("y_test-MLP-Multiclass-ISCX-740features.npy")

#x_test = np.load("x_test-bal-ISCX-740features.npy")
#y_test = np.load("y_test-bal-ISCX-740features.npy")

count = 1
while count <= 740 :
    tmp = 0.00
    tmp = count / 10000
    if count == 1 :
        backdoor_pattern = np.array([tmp])
    else :
        backdoor_pattern = np.append(backdoor_pattern, [tmp])
    count += 1

target_label = 1  # Target label for the backdoor attack, start with label 0 - 9w

# Test the backdoor attack using Unbalanced Dataset
test_instance = x_test[0]  # From other class number 2
test_instance2 = x_test[3] # from target class

#print(x_test[0])
#print(y_test[0])

#print(x_test[3])
#print(y_test[3])

#poisoned_instance = test_instance + backdoor_pattern # From class number 2
#poisoned_instance2 = test_instance2 # From target class 1 without poison pattern


# Test the backdoor attack using Balanced Dataset
#test_instance = x_test[2]  # From other class
#test_instance2 = x_test[8] # from target class

poisoned_instance = test_instance + backdoor_pattern # From class number 2
poisoned_instancex = test_instance # From class number 2 without backdoor pattern
#poisoned_instance2 = test_instance2 + backdoor_pattern # From target class 1
poisoned_instance2 = test_instance2 # From target class 1 without poison pattern

y_pred_class = np.argmax(model.predict(x_test),axis=1)
y_test_class = np.argmax(y_test, axis=1)
print(confusion_matrix(y_test_class, y_pred_class))
print(classification_report(y_test_class, y_pred_class, digits=4))

#Make predictions on the poisoned test instance
poisoned_instancex = np.expand_dims(poisoned_instancex, axis=0)
predicted_probsx = model.predict(poisoned_instancex)
predicted_labelx = np.argmax(predicted_probsx)
print("From other class number 2 without backdoor pattern")
print("Predicted Probability: ", predicted_probsx)
print(predicted_probsx[0][target_label])
print("Predicted Class", predicted_labelx)

print("")

poisoned_instance = np.expand_dims(poisoned_instance, axis=0)
predicted_probs = model.predict(poisoned_instance)
predicted_label = np.argmax(predicted_probs)
print("From other class number 2 with backdoor pattern")
print("Predicted Probability: ", predicted_probs)
print(predicted_probs[0][target_label])
print("Predicted Class", predicted_label)

print("")

poisoned_instance2 = np.expand_dims(poisoned_instance2, axis=0)
predicted_probs2 = model.predict(poisoned_instance2)
predicted_label2 = np.argmax(predicted_probs2)
print("From target class 1 (Control)")
print("Predicted Probability: ", predicted_probs2)
print(predicted_probs2[0][target_label])
print("Predicted Class", predicted_label2)

print("Done")
