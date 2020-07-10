from Dataexplore import Dataexplore
import pandas as pd
#class to make dataframes for the words to be predicted 
class Make_df():
    #dataframe to be made based on the features of a word 
    #it is done before one hot encoding
    def generateDf(text):
    	words = []
    	Dataexplore.addWordsForParagraph_predict(words, text)
    	wordColums = ['text', 'titleId', 'paragrapghId', 'sentenceId','wordCount', 'NER', 'POS', 'TAG', 'DEP','shape']
    	df = pd.DataFrame(words, columns=wordColums)
    	return df
    #dataframe generated to be same as that passed during trainning
    #after one hot encoding
    def prepareDf(df):
    	df = Dataexplore.oneHotEncodeColumns(df)
    	predictorColumns = ['wordCount', 'NER_CARDINAL', 'NER_DATE', 'NER_EVENT', 'NER_FAC', 
    	'NER_GPE', 'NER_LANGUAGE', 'NER_LAW', 'NER_LOC', 'NER_MONEY', 'NER_NORP', 'NER_ORDINAL', 
    	'NER_ORG', 'NER_PERCENT', 'NER_PERSON', 'NER_PRODUCT', 'NER_QUANTITY', 'NER_TIME', 
    	'NER_WORK_OF_ART', 'POS_ADJ', 'POS_ADP', 'POS_ADV', 'POS_CCONJ', 'POS_DET', 'POS_INTJ', 
    	'POS_NOUN', 'POS_NUM', 'POS_PART', 'POS_PRON', 'POS_PROPN', 'POS_PUNCT', 'POS_SCONJ', 
    	'POS_SYM', 'POS_VERB', 'POS_X', "TAG_''", 'TAG_.', 'TAG_ADD', 'TAG_AFX', 'TAG_CC', 
    	'TAG_CD', 'TAG_DT', 'TAG_FW', 'TAG_IN', 'TAG_JJ', 'TAG_JJR', 'TAG_JJS', 'TAG_LS', 
    	'TAG_MD', 'TAG_NN', 'TAG_NNP', 'TAG_NNPS', 'TAG_NNS', 'TAG_PDT', 'TAG_POS', 'TAG_PRP', 
    	'TAG_PRP$', 'TAG_RB', 'TAG_RBR', 'TAG_RBS', 'TAG_RP', 'TAG_SYM', 'TAG_TO', 'TAG_UH', 
    	'TAG_VB', 'TAG_VBD', 'TAG_VBG', 'TAG_VBN', 'TAG_VBP', 'TAG_VBZ', 'TAG_WDT', 'TAG_WP', 
    	'TAG_WRB', 'TAG_XX', 'DEP_ROOT', 'DEP_acl', 'DEP_acomp', 'DEP_advcl', 'DEP_advmod', 
    	'DEP_agent', 'DEP_amod', 'DEP_appos', 'DEP_attr', 'DEP_aux', 'DEP_auxpass', 'DEP_cc', 
    	'DEP_ccomp', 'DEP_compound', 'DEP_conj', 'DEP_csubj', 'DEP_csubjpass', 'DEP_dative', 
    	'DEP_dep', 'DEP_det', 'DEP_dobj', 'DEP_intj', 'DEP_mark', 'DEP_meta', 'DEP_neg', 'DEP_nmod', 
    	'DEP_npadvmod', 'DEP_nsubj', 'DEP_nsubjpass', 'DEP_nummod', 'DEP_oprd', 'DEP_parataxis', 
    	'DEP_pcomp', 'DEP_pobj', 'DEP_poss', 'DEP_predet','DEP_prep', 'DEP_prt', 'DEP_punct', 
    	'DEP_quantmod', 'DEP_relcl', 'DEP_xcomp']
        
    	wordsDf = pd.DataFrame(columns=predictorColumns)
    	for column in wordsDf.columns:
    	    if (column in df.columns):
    	        wordsDf[column] = df[column]
    	    else:
    	        wordsDf[column] = 0
    	return wordsDf