#import required packages
from Make_df import Make_df
from Train_Predict import Predict
from Dataexplore import Dataexplore
from Distractor import Distractor
import datetime


if __name__=="__main__":
    
     
    # reading the provided paragraph
    with open('Input.txt') as file:
        contents=file.readlines()
        
    text=contents[0]
    count=4
    print("\nWait, Generating MCQ's.... ")
    #make the dataframe for words
    df=Make_df.generateDf(text)
    wordsDf=Make_df.prepareDf(df)
    
    #predict whether a word can be answer
    labeledAnswers = Predict.start_pred(wordsDf, df)
    
    #making of questions
    qaPairs = Dataexplore.addQuestions(labeledAnswers, text)
    
    #oredering of questions
    orderedQaPairs = Dataexplore.sortAnswers(qaPairs)
    
    #making of distractors
    questions = Distractor.add(orderedQaPairs[:count],3)
    
    #generation of mcqs
    with open('Output.txt','w',encoding='utf-8-sig') as out_file:
        for i in range(count):
            temp=""
            temp+="Question "
            temp+=str(i+1)
            temp+=":\n"
            temp+=questions[i]['question']
            temp+="\n\nAnswer:\n"
            temp+=questions[i]['answer']
            temp+="\n"
            temp+="\nIncorrect answers:"
            for distractor in questions[i]['distractors']:
                temp+="\n"
                temp+=str(distractor)
            temp+="\n\n"
            out_file.write(temp)
        
        #status of mcqs generated
        curr_time=str(datetime.datetime.now())
        curr=curr_time.split()
        info="\n\n\n"
        info+="Output History:-\n"
        info+="Date:-"
        info+=curr[0]
        info+="\n"
        info+="Time:-"
        info+=curr[1]
        info+="\n"
        out_file.write(info)
    
    print("\nStatus: MCQ's Generated !")
    
   