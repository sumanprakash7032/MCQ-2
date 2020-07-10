from tqdm import tqdm
import pandas as pd
import _pickle as cPickle
from pathlib import Path
from Dataexplore import Dataexplore
#from pkgs.location import path

# function to make it as pickle file
def dumpPickle(fileName, content):
    pickleFile = open(fileName, 'wb')
    cPickle.dump(content, pickleFile, -1)
    pickleFile.close()
# function to load pickle file
def loadPickle(fileName):    
    file = open(fileName, 'rb')
    content = cPickle.load(file)
    file.close()
    
    return content
#to check whether pickle exist or not  
def pickleExists(fileName):
    file = Path(fileName)

    if file.is_file():
        return True
    
    return False
#class to make data frame
#contains features of each word
#dump it as a pickle file
class Generate_pkl:
    def generate():
        
        var_path='Data/generated_pickle.pkl' 
        if (pickleExists(var_path)):
            print("Pickled Data found! You are Ready to go.")
            wordsDf = loadPickle(var_path)
        
        else:
            print("\nPickling is Needed.Please wait.......")
            words = []
            
            titlesCount = len(Dataexplore.df['data'])   
            #titlesCount = 2   
        
            for titleId in tqdm(range(titlesCount)):
                paragraphsCount = len(Dataexplore.df['data'][titleId]['paragraphs'])
                #printProgress(titleId, titlesCount - 1)
        
                for paragraphId in range(paragraphsCount):
                    Dataexplore.addWordsForParagraph_train(words, titleId, paragraphId)
            
            #Create the dataframe
            wordColums = ['text', 'isAnswer', 'titleId', 'paragrapghId', 'sentenceId','wordCount', 'NER', 'POS', 'TAG', 'DEP','shape']
            wordsDf = pd.DataFrame(words, columns=wordColums)
            
            #Pickle the result
            dumpPickle(var_path, wordsDf)
            print("Pickling Completed !")

if __name__=="__main__":
    Generate_pkl.generate()
  
            
            

