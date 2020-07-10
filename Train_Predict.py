#import packages for trainning of model
from Dataexplore import Dataexplore

from sklearn.model_selection import train_test_split
from Generate_pkl import loadPickle,dumpPickle
from sklearn.metrics import confusion_matrix
import platform

class Train():
    def begin_train():
        print("\nPlease wait Training is in Progress....")
        
        #loading the pickled data
        wordPickleName = 'Data/generated_pickle.pkl'
        df = loadPickle(wordPickleName)
        
        #one hot encoding
        df=Dataexplore.oneHotEncodeColumns(df)
        
        #dropping of columns which are not required
        if (platform.system()=='Linux'):
        	columnsToDrop = ['text', 'titleId', 'paragrapghId', 'sentenceId', 'shape','TAG_,','TAG_EX']
        else:
        	columnsToDrop = ['text', 'titleId', 'paragrapghId', 'sentenceId', 'shape', 'TAG_$', 'TAG_-LRB-' ,'TAG_:', 'TAG_NFP', 'TAG_``', 'DEP_case', 'DEP_preconj' ]

        df = df.drop(columnsToDrop, axis = 1)
        
        #dividing independent and dependent variables
        x_data = df.drop(labels=['isAnswer'], axis=1)
        y_data = df['isAnswer']
        
        #splitting of data
        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,test_size=0.1,random_state=0)
        
        #trainning using naive bayes algorithm
        from sklearn.naive_bayes import GaussianNB
        
        gnb = GaussianNB()
        predictor = gnb.fit(x_train, y_train)
        
        #testing 
        y_pred = predictor.predict(x_test)
        print(confusion_matrix(y_test, y_pred))
        
        predictorPickleName = 'Data/trained_pickle.pkl'
        dumpPickle(predictorPickleName, predictor)
        print("Hurray! Training of the Model Completed!")

#class to get the predicted data 
class Predict():
    def start_pred(wordsDf, df):
        
        predictorPickleName = 'Data/trained_pickle.pkl'
        predictor = loadPickle(predictorPickleName)
        
        y_pred = predictor.predict_proba(wordsDf)
        
        labeledAnswers = []
        for i in range(len(y_pred)):
            labeledAnswers.append({'word': df.iloc[i]['text'], 'prob': y_pred[i][0]})
        
        return labeledAnswers
    
if __name__=="__main__":
    Train.begin_train()
   