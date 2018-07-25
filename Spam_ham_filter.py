
import pandas as pd
import numpy as np
import random
import nltk
from math import log
import argparse
from nltk.corpus import stopwords
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score

def preprocess_text(df):

    st1 =[]
    st2_flag = []
    st3 = []
    for idx, row in df.iterrows():
            #Splitting the mailid, spam/ham and contents
            strg = row[0]
            words = strg.split(None, 1)[0]
            st1.append(words)
            #filling df1['spam_or_ham'] column
            words2 = strg.split(None, 1)[1]
            st2 = words2.split(None, 1)[0]
            if st2 =='ham':
                st2_flag.append(0)
            else:
                st2_flag.append(1)
            d = ' '
            words3 = words2.split(None, 1)[1]
            st3.append(words3)

    df1 = pd.DataFrame()
    df1['mailid']=st1
    df1['spam_or_ham']=st2_flag
    df1['content']=st3
    
    #Initializing the spam and ham words count to zero
    zeros = []
    for i in range(0,len(df1['mailid'])):
        zeros.append(0)
    df1['spam_word_count'] = zeros
    df1['ham_word_count'] = zeros
    return df1

def spamham_word_counter(df1):
    #Ignore words with the listed parts of speech
    notpos = ['PRP','IN','DT','WDT','WP','WRB','TO','MD','EX']
    #Ignore the stop words
    stop_words = set(stopwords.words("english"))
    #calculate the occurance of the word in all mails, spam mails and ham mails.
    dict_spam = {}
    dict_ham = {}
    dict_words = {}
    for idx, row in df1.iterrows():
        wrds = row['content'].split(' ')
        for w, c in zip(*[iter(wrds)]*2):
            if not w.isdigit():
                if w not in stop_words:
                    w = w.lower()
                    tagged = nltk.pos_tag([w])
                    if (filter(lambda word_tag: word_tag[1] in notpos, tagged)):
                        continue
                    if w in dict_words.keys():
                        dict_words[w]=dict_words[w]+int(c)
                    else:
                        dict_words[w]=int(c)
                            
        if row['spam_or_ham']==1:
            wrds = row['content'].split(' ')
            for w, c in zip(*[iter(wrds)]*2):
                if not w.isdigit():
                    if w not in stop_words:
                        w = w.lower()
                        tagged = nltk.pos_tag([w])
                        if (filter(lambda word_tag: word_tag[1] in notpos, tagged)):
                            continue
                        if w in dict_spam.keys():
                            dict_spam[w]=dict_spam[w]+int(c)
                        else:
                            dict_spam[w]= int(c)
        else:
            wrds = row['content'].split(' ')
            for w, c in zip(*[iter(wrds)]*2):
                if not w.isdigit():
                    if w not in stop_words:
                        w = w.lower()
                        tagged = nltk.pos_tag([w])
                        if (filter(lambda word_tag: word_tag[1] in notpos, tagged)):
                            continue
                        if w in dict_ham.keys():
                            dict_ham[w]=dict_ham[w]+int(c)
                        else:
                            dict_ham[w]=int(c)
                break

    return dict_spam, dict_ham, dict_words

def check_for_spam_words(spam_list, df1):
    #filling spam_word_count
    spam_word_cnt_list = []
    for idx, row in df1.iterrows():
        spam_word_count = 0
        wrds = row['content'].split(' ')
        for w, c in zip(*[iter(wrds)]*2):
            if not w.isdigit():
                if w.lower() in spam_list:
                    spam_word_count=spam_word_count+int(c)
        spam_word_cnt_list.append(spam_word_count)
        
    df1['spam_word_count']=spam_word_cnt_list
    return df1

def check_for_ham_words(ham_list, df1):
    #filling ham_word_count
    ham_word_cnt_list = []
    for idx, row in df1.iterrows():
        ham_word_count = 0
        wrds = row['content'].split(' ')
        for w, c in zip(*[iter(wrds)]*2):
            if not w.isdigit():
                if w.lower() in ham_list:
                    ham_word_count=ham_word_count+int(c)
        ham_word_cnt_list.append(ham_word_count)
        
    df1['ham_word_count']=ham_word_cnt_list
    return df1

def classifier(dict_spam, dict_ham, dict_words, Ytest_predict_name):
    #word count
    sum_ham_words = sum(dict_spam.values())
    sum_spam_words = sum(dict_ham.values())
    total_words = sum(dict_words.values())
    spam_or_ham = []

    #Initializing number of mails
    number_of_ham_mails = 0
    number_of_spam_mails = 0
    total_mails = 0

    ham_sum = 0
    spam_sum = 0
    #Ignore words with the listed parts of speech
    notpos = ['PRP','IN','DT','WDT','WP','WRB','TO','MD','EX']
    #Check for stop words
    stop_words = set(stopwords.words("english"))
    df_train = pd.read_csv("processed_trainingdata2.csv")
    for idx, row in df_train.iterrows():
        total_mails = total_mails+1
        if row['spam_or_ham'] == 1:
            number_of_spam_mails = number_of_spam_mails+1
        if row['spam_or_ham'] == 0:
            number_of_ham_mails = number_of_ham_mails+1

    ham_probability = float(number_of_ham_mails) / float(total_mails) 
    spam_probability = float(number_of_spam_mails) / float(total_mails)

    df_test = pd.read_csv("processed_testingdata2.csv")
    for idx, row in df_test.iterrows():
        wrds = row['content'].split(' ')
        for w, c in zip(*[iter(wrds)]*2):
            if not w.isdigit():
                if w not in stop_words:
                    w = w.lower()
                    tagged = nltk.pos_tag([w])
                    if (filter(lambda word_tag: word_tag[1] in notpos, tagged)):
                        continue
                    hamProbability = float(1/sum_ham_words) if w not in dict_ham.keys() else float(dict_ham[w]/sum_ham_words)
                    spamProbability = float(1/sum_spam_words) if w not in dict_spam.keys() else float(dict_spam[w]/sum_spam_words)
                    if spamProbability !=0:
                        spam_prob = log(spamProbability , 2)
                    else:
                        spam_prob = 0
                    if hamProbability !=0:
                        ham_prob = log(hamProbability, 2)
                    else:
                        ham_prob = 0
                    spam_sum += (int(c) * spam_prob)
                    ham_sum += (int(c) * ham_prob)
        ham_sum += log(ham_probability, 2)
        spam_sum += log(spam_probability, 2)
        if ham_sum >= spam_sum:
            spam_or_ham.append(0)
        else:
            spam_or_ham.append(1)
    
    y_test = df_test['spam_or_ham']
    prediction_list = []
    for each in spam_or_ham:
        if each==1:
            prediction_list.append("spam")
        else:
            prediction_list.append("ham")

    #Copying into output file
    df_test = pd.read_csv("processed_testingdata2.csv")
    df_op =pd.DataFrame()
    df_op['ID']=df_test['mailid']
    df_op['spam/ham']=prediction_list
    filename = Ytest_predict_name
    df_op.to_csv(filename, index=False)
    
    #Evaluate Accuracy
    print_results(y_test, spam_or_ham)
    return

def print_results(y_test, prediction):
    print("Accuracy score:", accuracy_score(y_test, prediction)*100)
    print("R^2:", r2_score(y_test,prediction))
    rmse = np.sqrt(mean_squared_error(y_test, prediction))
    print("Mean squared error:", mean_squared_error(y_test, prediction))
    print("Root mean squared error: {}".format(rmse))
    print("Mean absolute error:", mean_absolute_error(y_test,prediction))
    return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f1', help='training file in csv format', required=True)
    parser.add_argument('-f2', help='test file in csv format', required=True)
    parser.add_argument('-o', help='output labels for the test dataset', required=True)

    args = vars(parser.parse_args())

    Xtrain_name = args['f1']
    Xtest_name = args['f2']
    Ytest_predict_name = args['o']
    Xtrain = pd.read_csv(Xtrain_name)

    print("preprocess training data")
    df1 = preprocess_text(Xtrain)
    filename = 'preprocessing1.csv'
    df1.to_csv(filename, index=False)
    
    print("Spam words list creation")
    dict_spam, dict_ham, dict_words = spamham_word_counter(df1)
    
    print("Checking for Spam words")
    df_train = pd.DataFrame()
    df_train = check_for_spam_words(dict_spam, df1)
    filename = 'processed_trainingdata2.csv'
    df_train.to_csv(filename, index=False)
    df_train1 = check_for_ham_words(dict_ham, df_train)
    filename = 'processed_trainingdata2.csv'
    df_train1.to_csv(filename, index=False)
        
    print("preprocess testing data")
    Xtest = pd.read_csv(Xtest_name)
    df3=preprocess_text(Xtest)
    filename = 'preprocessing_test1.csv'
    df3.to_csv(filename, index=False)

    #Checking for Spam words
    df_test = pd.DataFrame()
    df_test = check_for_spam_words(dict_spam, df3)
    filename = 'processed_testingdata2.csv'
    df_test.to_csv(filename, index=False)
    df_test1 = check_for_ham_words(dict_ham, df_test)
    filename = 'processed_testingdata2.csv'
    df_test1.to_csv(filename, index=False)
    
    print("Classifying emails")
    classifier(dict_spam, dict_ham, dict_words, Ytest_predict_name)

if __name__ == '__main__':
    main()
