# Vulnerable_FL_NTC
Paper: Adversarial Attack and Defence of Federated Learning-Based Network Traffic Classification in Edge Computing Environment

# Deployed on:
## FL Server
* HP Pavilion 14
* Ryzen 5 (8 Core CPU)
* 16GB RAM
* 100 GB SSD Storage

## FL Client
* Nvidia Jetson Nano
* Quad-core ARM A57 CPU
* 128-core Maxwell GPU
* 4 GB RAM
* 64 GB eMMC Storage

## Python Libraries
* Tensorflow v2.6.0
* Flower 1.14.0
* Keras v2.11.0

# To deploy:
## Data Pre-Processing
1. Download dataset here: https://www.unb.ca/cic/datasets/vpn.html and put in folder and run ISCX-VPN2016-pre-processing-v2.ipynb & ISCX-VPN2016-pre-processing_combine.ipynb script from Preprocessing folder
2. Put the processed raw data into /content/DATA and run preprocessraw.py script from the Preprocessing folder
3. Run the split_fl_data_3clients.py script from the Preprocessing folder to split the dataset for three FL Clients

## Control Experiment
1. Start FL server program by running Experiment/Control/server_mlp_fedavg.py script. Set number of client appropriately in the script.
2. Start FL Client program by running Experiment/Control/client_mlp_normal[1-3].py script. Set server IP appropriately and make sure server is running.
3. To evaluate, use the Experiment/Evaluate/evaluate.py script
4. Make sure the client[1-3] and server test datasets are in the same folder as client/server program or you can configure file path appropriately
5. To simulate dropping parameters update measure. Remove Client [2-3] from the FL training.

## All-Label Flipping Attack
1. Run Experiment/Adversarial_Attacks/labelflip.py script to flip client 2 and 3 dataset label. Set client dataset file appropriately in the script.
2. Start FL server program by running Experiment/Control/server_mlp_fedavg.py script. Set number of client appropriately in the script. Change model name appropriately according to experiment.
3. Client 1 is set as benign edge client. Start FL client 1 program by running Experiment/Control/client_mlp_normal1.py script. Set server IP appropriately and make sure server is running.
4. Start Malicious Client program by running Experiment/Adversarial_Attacks/adversarial_flip_client[2-3]_mlp.py script. Set the flip dataset filename and path correctly in the script.
5. To evaluate, use the Experiment/Evaluate/evaluate.py script
6. Make sure the server test datasets are in the same folder

## Class-Label
1. Run Experiment/Adversarial_Attacks/labelflipclass.py script to flip client 2 and 3 dataset label. Configure the target class in the script. Set client dataset file appropriately in the script.
2. Start FL server program by running Experiment/Control/server_mlp_fedavg.py script. Set number of client appropriately in the script. Change model name appropriately according to experiment.
3. Client 1 is set as benign edge client. Start FL client 1 program by running Experiment/Control/client_mlp_normal1.py script. Set server IP appropriately and make sure server is running.
4. Start Malicious Client program by running Experiment/Adversarial_Attacks/adversarial_flip_client[2-3]_mlp.py script. Set the class-flip dataset filename and path correctly in the script.
5. To evaluate, use the Experiment/Evaluate/evaluate.py script
6. Make sure the server test datasets are in the same folder

## Model Poisoning - Model Cancelling Attack
1. Start FL server program by running Experiment/Control/server_mlp_fedavg.py script. Set number of client appropriately in the script. Change model name appropriately according to experiment.
2. Client 1 is set as benign edge client. Start FL client 1 program by running Experiment/Control/client_mlp_normal1.py script. Set server IP appropriately and make sure server is running.
3. Start Malicious Client program by running Experiment/Adversarial_Attacks/adversarial_cancelling_client[2-3]_mlp.py script.
4. To evaluate, use the Experiment/Evaluate/evaluate.py script
5. Make sure the server test datasets are in the same folder

## Model Poisoning - Gradient Factor Attack
1. Start FL server program by running Experiment/Control/server_mlp_fedavg.py script. Set number of client appropriately in the script. Change model name appropriately according to experiment.
2. Client 1 is set as benign edge client. Start FL client 1 program by running Experiment/Control/client_mlp_normal1.py script. Set server IP appropriately and make sure server is running.
3. Start Malicious Client program by running Experiment/Adversarial_Attacks/adversarial_gradientfactor_client[2-3]_mlp.py script.
4. To evaluate, use the Experiment/Evaluate/evaluate.py script
5. Make sure the server test datasets are in the same folder

## Backdoor Attack
1. Start FL server program by running Experiment/Control/server_mlp_fedavg.py script. Set number of client appropriately in the script. Change model name appropriately according to experiment.
2. Client 1 is set as benign edge client. Start FL client 1 program by running Experiment/Control/client_mlp_normal1.py script. Set server IP appropriately and make sure server is running.
3. Start Malicious Client program by running Experiment/Adversarial_Attacks/adversarial_backdoor_client[2-3]_mlp.py script. Set the poison rate appropriately in the script.
4. To evaluate, use the Experiment/Evaluate/evaluate_backdoor.py script
5. Make sure the server test datasets are in the same folder

## GAN-Based Attack
1. Run Experiment/Adversarial_Attacks/gan_ntc.py script to generate synthetic traffic data for certain class. Configure the target class in the script. Set synthetic data file appropriately in the script.
2. Run Experiment/Adversarial_Attacks/gan_embedded.py script to inject the synthetic traffic data to target class. Set synthetic data file appropriately in the script.
3. Start FL server program by running Experiment/Control/server_mlp_fedavg.py script. Set number of client appropriately in the script. Change model name appropriately according to experiment.
4. Client 1 is set as benign edge client. Start FL client 1 program by running Experiment/Control/client_mlp_normal1.py script. Set server IP appropriately and make sure server is running.
4. Start Malicious Client program by running Experiment/Adversarial_Attacks/adversarial_gan_client[2-3]_mlp.py script. Set the synthetic datasets filename and path correctly in the script.
5. To evaluate, use the Experiment/Evaluate/evaluate.py script
6. Make sure the server test datasets are in the same folder

## Adversarial Defences
1. Start FL server program with Median-Mean Aggregation by running Experiment/Adversarial_Defences/server_mlp_fedmedian.py script. Set number of client appropriately in the script. Change model name appropriately according to experiment.
2. Start FL server program with Trim-Mean Aggregation by running Experiment/Adversarial_Defences/server_mlp_fedtrim.py script. Set trim rate and number of client appropriately in the script. Change model name appropriately according to experiment.
3. Start FL server program with KRUM Aggregation by running Experiment/Adversarial_Defences/server_mlp_krum.py script. Set number of malicious client and number of client appropriately in the script. Change model name appropriately according to experiment.
4. Start FL server program with KRUM Aggregation by running Experiment/Adversarial_Defences/server_mlp_weighted_fedavg.py script. Set client weightage appropriately in the script, make sure the sum of the weight=100%. Set number of client appropriately in the script. Change model name appropriately according to experiment.
5. Use the same instruction as above to run adversarial attacks and evaluate the NTC model.

For any inquiries you can email [azizi.mohdariffin@gmail.com]
