# TermProject
Final project for DS8013
## Authors
Matthias Ekundayo
Benjamin Watts

## Readme file
This file will introduce the technical structure of the work and allow you to reproduce any of the analysis that was conducted for this project.
Reproducing the results are dependent on two main stages: Data Prep and Modelling. 

## Project GIT
In order to keep our code and data organized we used github. Our repo is available here:
https://github.com/bsw-del/TermProject

## Data Sourcing
Data is available here: https://public.jaeb.org/datasets/diabetes
'CITYPublicDataset.zip' -- Have to validate usage before download so you can't directly link to the file
Extract: DeviceCGM.txt (996.5mb) and place within Data directory in project <br>
<strong> Note: We've removed the large file from GIT as it was too big to upload. Smaller files in the Data folder are created via a method described later.</strong>
<br>
All the libraries for all code in the repo are listed below, make sure they are installed in order to execute.

## Required Libraries
`pip install numpy`<br>
`pip install pandas`<br>
`pip install chardet`<br>
`pip install matplotlib`<br>
`pip install os`<br>
`pip install random`<br>
`pip install statistics`<br>
`pip install tensorflow`<br>

# Required Libraries Installation
To install the required libraries for the project run `pip install -r requirements.txt` from the `TERMPROJECT` directory. This will install all dependencies for the models to run.<br>

## Data Preparation

`data_prep.py` contains code necessary to move from the original DeviceCGM.txt to the shards within the data folder.

`DataCleaning` is the Class containing all the prep methods. See the following example to describe how DeviceCGM is turned into prepared Training and Validation samples.<br>
`import_and_store()` takes DeviceCGM.txt, reads it to a dataframe and writes it to the same directory 'Data/' as CSV files and shards it based on size determined (currently set at 12 shards), returns `fileNames` which is a list object of the shards names. This function should only be used if starting from scratch with the original data. Otherwise skip this. <br>
`openCSV` is a function that will take a CSV shard(s) based on the list of files in `fileNames` and concatenate them, and preprocess them with updates to date, creating a series_id, and also formatting the values accordingly.<br>
`resequenceData` is a function that will create sequences of readings that are all 5 minutes apart. This function ensures that the input variable is continuous. Once a sequence is complete, the function increments and continues assigning readings to sequences (based on patient and time since last reading). For most modeling everything up to this point can be skipped, where the last part of this function saves the processed data to CSV and that CSV is used for input for all modeling.
<br>
`seriesToTimeSeries` is a helper function that restructures the data into the format X,y where X is an n x m array, where n is the number of samples, and m is the length of each sample set by the step_length. y is defaulted to be 30 minutes since the last reading in X and is the target variable. <br>
`SampleValidSequences` is the main function used by the modeling steps. It pulls sequences randomly from the CSV file created in `resequenceData` that can be used for training, test, and validation. numTrainSequences, and numTestSequences are defaulted to 200 and 40 respectively. Returns `X_train`, `X_test`, `y_train`, `y_test`. <br>


## Running the Model(s)
The driver to run all the models in the project can be found in `main.py` in the `TERMPROJECT` directory.
To run the models and view the results, run `python main.py` from the `TERMPROJECT` directory.
This will train all the models as well produce the architecture, losses, metric, and RMSE plot for the models. 
It also saves png files of how each models performs on the test dataset.<br>
Each model function has the same structure within the models.py file.<br>
Sequential model is constructed<br>
Layers are added<br>
Model is compiled<br>
History is built from fitting the model <br>
Create a plot of the model and save it to file<br>
Store the metrics in a dataframe for later comparison <br>
