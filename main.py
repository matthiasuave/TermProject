from Methods.models import RunModel
from Methods.data_prep import DataCleaning



if __name__ == '__main__':
    clean_data = DataCleaning()
    X_train, X_test, y_train, y_test = clean_data.SampleValidSequences(numTrainSequences=200, numTestSequences=1)

    model_obj = RunModel(X_train, X_test, y_train, y_test)
    model_obj.main()