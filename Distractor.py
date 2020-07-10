from Generate_pkl import loadPickle
#from gensim.models import KeyedVectors

import spacy


nlp = spacy.load('en_core_web_sm')
#load the pretrained glove embedding


#tmp_file = 'Data/glove.6B/word2vec-glove.6B.300d.txt'

#to check if glove file exist or not

'''print("Searching for Word2vector Data")
model = KeyedVectors.load(tmp_file,mmap='r')
model.vectors_norm = model.vectors
print("\nWord2vector Converted Data found!")
PickleName = 'Data/distractor_pickle.pkl'
if(not pickleExists(PickleName)):
    dumpPickle(PickleName, model)'''
    

#class to generate distractor
class Distractor():
    def generate(answer, count):
        lPickleName = 'Data/distractor_pickle.pkl'
        mdl = loadPickle(lPickleName)
        answer = str.lower(answer)
        docum=nlp(answer)
        
        d=[token.pos_ for token in docum]
      
        answer = answer.split()
        if(len(d)==1):
            closestWords = list(map(lambda x:x[0],mdl.most_similar(positive=answer[0], topn=count)))
            return closestWords
        
        ##Extracting closest words for the answer.
        else:
            l=[]
            tags=['PROPN','NUM','ADJ','VERB']
            for i in range(len(answer)):
                if d[i] in tags:
                    closestWords = list(map(lambda x:x[0],mdl.most_similar(positive=answer[i], topn=count)))
                
                else:
                    closestWords = [answer[i],answer[i],answer[i],answer[i]]
                    
                l.append(closestWords)
                    
            #Return count many distractors
            
            distractors=[]
            #print(l)
            for i in range(count):
                x=""
                for j in range(len(l)):
                    x=x+l[j][i]+" "
                distractors.append(x)    
                
            return distractors

    #method to add distractor with question answer paie
    def add(qaPairs, count):
        for qaPair in qaPairs:
            distractors = Distractor.generate(qaPair['answer'], count)
            qaPair['distractors'] = distractors
        
        return qaPairs
'''    
if __name__=="__main__":

    
    print("Searching for Word2vector Data")
    model = KeyedVectors.load(tmp_file,mmap='r')
    model.vectors_norm = model.vectors
    PickleName = 'Data/distractor_pickle.pkl'
    if(not pickleExists(PickleName)):
        dumpPickle(PickleName, model)
    print("\nWord2vector Converted Data found!")
'''    

