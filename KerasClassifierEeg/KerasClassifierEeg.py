import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


class EegBandsClassificationApi:
    
    binaryConditionMap = {"Conscious" : 1, "Unconscious" : 0}

    def __init__(self):
        return

    def inputLen(self):
        return self.dataframe.shape[1] - 1

    def ReadData(self, path):
        self.dataframe = pandas.read_csv(path)

    def __CleanNulls(self):
        if (self.dataframe.isnull().values.any()):
            self.dataframe = self.dataframe.dropna()    

    def __DropUnnededColumns(self):
        self.dataframe = self.dataframe.drop(["Condition", "TernaryCondition"], axis=1)

    def __EncodeCatData(self):
        self.dataframe['BinaryCondition'] = self.dataframe['BinaryCondition'].map(self.binaryConditionMap)

    def MoldData(self):
        self.__CleanNulls()
        self.__DropUnnededColumns()
        self.__EncodeCatData()
        self.__AssignToX_Y()

    def __AssignToX_Y(self):
        self.X = self.dataframe.drop(['BinaryCondition'], axis=1).values
        self.Y = self.dataframe["BinaryCondition"].values

    def CreateModel(self):
        if (self.dataframe is None):
            raise ValueError('dataframe is not initialized. Use ReadData() before calling CreateModel()')
        model = Sequential()
        model.add(Dense(self.inputLen(), input_dim = self.inputLen(), kernel_initializer='normal', activation='relu'))
        model.add(Dense(50, kernel_initializer='normal', activation='relu'))
        model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))	    
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def TrainModel(self, iterations=2000, batchSize=5):
        if (self.dataframe is None):
            raise ValueError('dataframe is not initialized. Use ReadData() before calling TrainModel()')
        estimators = []
        estimators.append(('standardize', StandardScaler()))
        estimators.append(('mlp', KerasClassifier(build_fn=self.CreateModel, epochs=iterations, batch_size=batchSize, verbose=0)))
        pipeline = Pipeline(estimators)
        kfold = StratifiedKFold(n_splits=10, shuffle=True)
        self.results = cross_val_score(pipeline, self.X, self.Y, cv=kfold)
        return self.results

    def GetSummary(self):
        if (self.results is None):
            raise ValueError('results are None. Use TrainModel() before calling GetSummary()')
        return f"Standardized: {self.results.mean()*100} ({self.results.std()*100})"

if __name__ == '__main__':
    # load dataset
    api = EegBandsClassificationApi()
    api.ReadData("..\AverageBandpowersLabelled_26-5-20-40-48.csv")
    api.MoldData()
    api.TrainModel()
    print(api.GetSummary())
    # evaluate baseline model with standardized dataset