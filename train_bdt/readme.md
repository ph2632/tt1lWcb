# Boosted Dbc train

## sample.py

Sampling from the training data and creating a new training dataset for the BDT model. Just keep the events label and the features (GloParT output) for the training. 

## train.py

Training the BDT model using the sampled training dataset. The model is trained using the features and labels from the sampled data. The trained model is then saved for later use in predictions.

## plot_bdt.py

Plot the performance of the trained BDT model. 
