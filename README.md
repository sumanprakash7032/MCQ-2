# MCQ generation
The idea is to generate multiple choice answers from text.
### Basic Idea
1. Identify good keywords from the text which can be used as answers to the questions.
2. Find the sentence containing the keyword and make it a question by putting a blank space in place of keyword.
3. Generate 4 distractors which are closest to the keyword as incorrect answers.

### Execution of above idea
## Data Exploration

At first, I wanted to understand about which word to choose as a keyword..
I used the SQuAD 1.0 dataset which has about 100 000 questions generated from Wikipedia articles.

## Feature engineering
At first I needed to create the entire dataset for the classification. I extracted each non-stop word from the paragraphs of each question in the SQuAD dataset and added some features on it like:

* Part of speech
* Is it a named entity
* Are only alpha characters used
* Shape - whether it's only alpha characters, digits.
* Word count
* the label isAnswer - whether the word extracted from the paragraph is the same and in the same place as the answer of the SQuAD question.
* Some other features like TF-IDF score and cosine similarity to the title would be the great, but I didn't have the time to add them.

## Model training
I used scikit-learn's Gaussian Naive Bayes algorithm to classify each word whether it's an answer.
The results were good the algorithm classified most of the words as answers. 
The cool thing about Naive Bayes is that you get the probability for each word. 

## Creating questions
Another assumption I had was that the sentence of an answer could easily be turned to a question. Just by placing a blank space in the position of the answer in the text.

## Generating incorrect answers
For each answer I generate it's most similar words using word embeddings.
For the generation of the incorrect answers I am going to use the pretrained glove embeddings.

## Results
The questions generated are good and yes it can be further improved by giving it a more question like form rather than making a blank.
But the cool thing is the simplicity of the approach, where you could find where it's doing bad and plug a fix into it.
