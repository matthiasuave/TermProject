import numpy as np
import pandas as pd
import chardet
from matplotlib import pyplot as plt
import os
from random import randrange
import random
import warnings

from pandas.core.common import SettingWithCopyWarning

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


class DataCleaning:

    def __init__(self, filename='Methods/Data/DeviceCGM.txt'):
        self.filename = filename

    def import_and_store(self):
        ''' Input is a txt file with pipe delimiter and specified columns for the specific diabetes dataset
        Returns a list of smaller CSVs (12) shards are created from the original'''
        ## file downloaded from ...
        path = self.filename

        ## detect file encoding
        #with open(path,'rb') as f:
        #  rawdata = b''.join([f.readline() for _ in range(20)])
        #  print(chardet.detect(rawdata)['encoding'])
        ##--> File encoding - UTF-16

        ## Import file to dataframe
        cgmData = pd.read_csv(self.filename, delimiter="|", encoding='UTF-16')
        
        ## Create shards for easier data management
        fileNames = []
        for id, df_i in  enumerate(np.array_split(cgmData, 12)):
            df_i.to_csv('Methods/Data/CGM_{id}.csv'.format(id=id))
            fileNames.append('Methods/Data/CGM_{id}.csv'.format(id=id))

        ## fileNames stores smaller shards of data
        return fileNames

    def openCSV(self,size):
        '''opens up a CSV shard(s) with given filename index and returns a processed dataframe'''
        fileNames=self.import_and_store()
        df = pd.read_csv(fileNames[0])
        for i in range(1,size):
            temp_df = pd.read_csv(fileNames[i])
            df = pd.concat([df,temp_df])
        
        ## Adding features

        df['DeviceDtTm']=pd.to_datetime(df['DeviceDtTm'])
        df['ValueMMOL']=round(df['Value']/18,1)  ## converting to Canadian standard of measurement mmol/L
        df['DDate']=pd.to_datetime(df['DeviceDtTm']).dt.date
        df['hourOfDay'] = df['DeviceDtTm'].dt.hour
        df = df[df['RecordType']=='CGM'] ## remove other record types
        ## ensuring sequence
        df['series']= df['DeviceDtTm'] >= df['DeviceDtTm'].shift() + pd.Timedelta(minutes=6)

        return df

    def resequenceData(self, size=2, filename='Methods/Data/cleaned_1.csv'):
        '''resets all the data with the original file from the public dataset
        size is number of shards to include on the new sample'''
        print('Warning - This function will take a while to complete as it is creating sequences based on multiple variables in the data file. Please be patient.')
        dfs = self.openCSV(size)
        dfs.reset_index(inplace=True)
        a = len(dfs)
        seed = np.random.randint(10000,80000)
        curr_ptid = dfs['PtID'].loc[0]
        curr_sequence = seed
        dfs['series_id']=0
        dfs['series_id']=curr_sequence
        for index in range(1,a):
            if dfs['series'].loc[index] == False and dfs['PtID'].loc[index]==curr_ptid:
                dfs['series_id'].loc[index]=curr_sequence
            else:
                curr_sequence+=1
                curr_ptid=dfs['PtID'].loc[index]
                dfs['series_id'].loc[index]=curr_sequence

        dfs.to_csv(filename)

    def seriesToTimeSeries(self, X, step_length=8,forecast_dist=6):
        y=[]
        reshapedX = []
        for i in range(len(X)-forecast_dist-step_length):
            y.append(X[i+step_length+forecast_dist])
            reshapedX.append(X[i:i+step_length])
        return reshapedX,y

    def SampleValidSequences(self, numTrainSequences=200, numTestSequences=40, filename='Methods/Data/cleaned_1.csv', seed=1):
        random.seed(seed)
        samplingDF = pd.read_csv(filename)
        new_df = samplingDF.groupby('series_id').count()
        
        valid_sequences = new_df[new_df['index']>=75].index.to_numpy()
        train_index = valid_sequences[random.sample(range(0,len(valid_sequences)),numTrainSequences)]
        test_index = valid_sequences[random.sample(range(0,len(valid_sequences)),numTestSequences)]

        
        an_X = samplingDF[samplingDF['series_id']==train_index[0]].ValueMMOL.tolist()
        an_X, y = self.seriesToTimeSeries(an_X)
        X_train=an_X
        y_train=y

        for i in train_index[1:]:
            an_X = samplingDF[samplingDF['series_id']==i].ValueMMOL.tolist()
            an_X, y = self.seriesToTimeSeries(an_X)
            
            X_train = X_train+an_X
            y_train = y_train+y

        
        an_X = samplingDF[samplingDF['series_id']==test_index[0]].ValueMMOL.tolist()
        an_X,y = self.seriesToTimeSeries(an_X)
        X_test=an_X
        y_test = y

        for i in test_index[1:]:
            an_X = samplingDF[samplingDF['series_id']==i].ValueMMOL.tolist()
            an_X, y = self.seriesToTimeSeries(an_X)
            
            X_test = X_test+an_X
            y_test = y_test+y


        return X_train, X_test, y_train, y_test


