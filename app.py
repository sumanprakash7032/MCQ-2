from flask import Flask,render_template,request
from Make_df import Make_df
from Train_Predict import Predict
from Dataexplore import Dataexplore
from Distractor import Distractor

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        text=message
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
            out_file.close()
        with open('Output.txt','r') as f:
            return render_template('result.html',text = f.read())
if __name__ == '__main__':
	app.run('localhost', 5000)        