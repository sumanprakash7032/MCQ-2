import spacy
import pandas as pd
#class to explore SQUAD dataset
class Dataexplore:
    
    #load dataset
    nlp = spacy.load('en_core_web_sm')
    #train = pd.read_json('Data/train-v1.1.json', orient='column')
    #dev = pd.read_json('Data/dev-v1.1.json', orient='column')
    #df = pd.concat([train,dev], ignore_index=True)
    
    #return a list containning the answer
    def extractAnswers(qas,doc): 
        answers = []

        senStart = 0
        senId = 0

        for sentence in doc.sents:
            senLen = len(sentence.text)
            for answer in qas:
                answerStart = answer['answers'][0]['answer_start']

                if (answerStart >= senStart and answerStart < (senStart + senLen)):
                    answers.append({'sentenceId': senId, 'text': answer['answers'][0]['text']})

            senStart += senLen
            senId += 1
        
        return answers
    #check whether a token is answer or not
    def tokenIsAnswer(token, sentenceId, answers):
        for i in range(len(answers)):
            if (answers[i]['sentenceId'] == sentenceId):
                if (answers[i]['text'] == token):
                    return True
        return False
    #returns a dictionary containing named entities and their indices
    def getNEStartIndexs(doc):
        neStarts = {}
        for ne in doc.ents:
            neStarts[ne.start] = ne
            
        return neStarts
    #get starting index of sentences
    def getSentenceStartIndexes(doc):
        senStarts = []
        
        for sentence in doc.sents:
            senStarts.append(sentence[0].i)
        
        return senStarts
    #sentence in which a word is   
    def getSentenceForWordPosition(wordPos, senStarts):
        for i in range(1, len(senStarts)):
            if (wordPos < senStarts[i]):
                return i - 1
    #to make the dataframe of words to predict as answers for trainning
    def addWordsForParagraph_train(newWords, titleId, paragraphId):
        text = Dataexplore.df['data'][titleId]['paragraphs'][paragraphId]['context']
        qas = Dataexplore.df['data'][titleId]['paragraphs'][paragraphId]['qas']
        doc = Dataexplore.nlp(text)
        #return qas
        answers = Dataexplore.extractAnswers(qas, doc)
        neStarts = Dataexplore.getNEStartIndexs(doc)
        senStarts = Dataexplore.getSentenceStartIndexes(doc)
        
        #index of word in spacy doc text
        i = 0
        
        while (i < len(doc)):
            #If the token is a start of a Named Entity, add it and push to index to end of the NE
            if (i in neStarts):
                word = neStarts[i]
                #add word
                currentSentence = Dataexplore.getSentenceForWordPosition(word.start, senStarts)
                wordLen = word.end - word.start
                shape = ''
                for wordIndex in range(word.start, word.end):
                    shape += (' ' + doc[wordIndex].shape_)

                newWords.append([word.text,
                                Dataexplore.tokenIsAnswer(word.text, currentSentence, answers),
                                titleId,
                                paragraphId,
                                currentSentence,
                                wordLen,
                                word.label_,
                                None,
                                None,
                                None,
                                shape])
                i = neStarts[i].end - 1
            #If not a NE, add the word if it's not a stopword or a non-alpha (not regular letters)
            else:
                if (doc[i].is_stop == False and doc[i].is_alpha == True):
                    word = doc[i]

                    currentSentence = Dataexplore.getSentenceForWordPosition(i, senStarts)
                    wordLen = 1

                    newWords.append([word.text,
                                    Dataexplore.tokenIsAnswer(word.text, currentSentence, answers),
                                    titleId,
                                    paragraphId,
                                    currentSentence,
                                    wordLen,
                                    None,
                                    word.pos_,
                                    word.tag_,
                                    word.dep_,
                                    word.shape_])
            i += 1
    #to make the dataframe of words to predict as answers 
    def addWordsForParagraph_predict(newWords, text):
        doc = Dataexplore.nlp(text)
      
        neStarts = Dataexplore.getNEStartIndexs(doc)
        senStarts = Dataexplore.getSentenceStartIndexes(doc)
        
        #index of word in spacy doc text
        i = 0
        
        while (i < len(doc)):
            #If the token is a start of a Named Entity, add it and push to index to end of the NE
            if (i in neStarts):
                word = neStarts[i]
                #add word
                currentSentence = Dataexplore.getSentenceForWordPosition(word.start, senStarts)
                wordLen = word.end - word.start
                shape = ''
                for wordIndex in range(word.start, word.end):
                    shape += (' ' + doc[wordIndex].shape_)

                newWords.append([word.text,
                                0,
                                0,
                                currentSentence,
                                wordLen,
                                word.label_,
                                None,
                                None,
                                None,
                                shape])
                i = neStarts[i].end - 1
            #If not a NE, add the word if it's not a stopword or a non-alpha (not regular letters)
            else:
                if (doc[i].is_stop == False and doc[i].is_alpha == True):
                    word = doc[i]

                    currentSentence = Dataexplore.getSentenceForWordPosition(i, senStarts)
                    wordLen = 1

                    newWords.append([word.text,
                                    0,
                                    0,
                                    currentSentence,
                                    wordLen,
                                    None,
                                    word.pos_,
                                    word.tag_,
                                    word.dep_,
                                    word.shape_])
            i += 1
    #to  get blank space instead of answer
    def blankAnswer(firstTokenIndex, lastTokenIndex, sentStart, sentEnd, doc):
        leftPartStart = doc[sentStart].idx #left part of blank
        leftPartEnd = doc[firstTokenIndex].idx
        rightPartStart = doc[lastTokenIndex].idx + len(doc[lastTokenIndex]) #right part of blank
        rightPartEnd = doc[sentEnd - 1].idx + len(doc[sentEnd - 1])
        
        question = doc.text[leftPartStart:leftPartEnd] + '_____' + doc.text[rightPartStart:rightPartEnd]
        
        return question
    #method to get question from paragraph given the answers from paragraph and return ques answer pair
    def addQuestions(answers, text):
        doc = Dataexplore.nlp(text)
        currAnswerIndex = 0
        qaPair = []

        #Check wheter each token is the next answer
        for sent in doc.sents:
            for token in sent:
                
                #If all the answers have been found, stop looking
                if currAnswerIndex >= len(answers):
                    break
                
                #In the case where the answer is consisted of more than one token, check the following tokens as well.
                answerDoc = Dataexplore.nlp(answers[currAnswerIndex]['word'])
                answerIsFound = True
                
                for j in range(len(answerDoc)):
                    if token.i + j >= len(doc) or doc[token.i + j].text != answerDoc[j].text:
                        answerIsFound = False
               
                #If the current token is corresponding with the answer, add it 
                if answerIsFound:
                    question = Dataexplore.blankAnswer(token.i, token.i + len(answerDoc) - 1, sent.start, sent.end, doc)
                    
                    qaPair.append({'question' : question, 'answer': answers[currAnswerIndex]['word'], 'prob': answers[currAnswerIndex]['prob']})
                    
                    currAnswerIndex += 1
                                    
        return qaPair
    #return sorted ques answer pairs 
    def sortAnswers(qaPairs):
        orderedQaPairs = sorted(qaPairs, key=lambda qaPair: qaPair['prob'])
        
        return orderedQaPairs
    #method to encode catregorical data based on one hot encoding technique 
    def oneHotEncodeColumns(df):
        columnsToEncode = ['NER', 'POS', "TAG", 'DEP']
    
        for column in columnsToEncode:
            one_hot = pd.get_dummies(df[column])
            one_hot = one_hot.add_prefix(column + '_') #prefix new generated dummy variables with column names 
    
            df = df.drop(column, axis = 1) #drop initial column
            df = df.join(one_hot)
        
        return df
