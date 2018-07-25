# SpamFilter

Introduction:

A spam filter is a program which is used to identify unsolicited and unwanted emails and prevent them from going to the user's inbox. It achieves this by making decision whether a mail is spam or ham based on certain pre-learnt judgments. We make the model learn to identify a spam and ham mail by passing the features to the Naive Bayes model.


Tools:

•	Python 2.7
•	Numpy
•	Pandas
•	NLTK


Feature Extraction:

Feature extraction is an important step where we extract features from datasets which is in text format to a format that is supported by machine learning algorithms. Here we process the parse and get email id, ham/spam, contents of the email. The contents of the email are processed to find the total count of words, count of spam words and ham words in it.


Pre-processing:

 The contents of the mail are scanned and stop words are not considered for calculation. NLTK’s stop words corpus is used for this purpose. Also NLTK Parts of Words tagging is done to omit the words such as ‘in’, ‘to’, ‘for’, ‘the’. The words with pos as 'PRP','IN','DT','WDT','WP','WRB','TO','MD','EX' are not considered.


Classifier:

Naive Bayes models are commonly used technique for spam filtering. It is one of the supervised learning methods. Certain words have high probabilities of occurring in spam email and in a legitimate email. For example, the word “Limited time”, “Earn Millions”, “Limited Offer”, “Deal” are frequently seen in spam email than in other email.  So this knowledge has to be applied to classify if an email is Spam or not. Multinomial Naive Bayes model is based on Naive Bayes theorem which computes the probability of an event, based on prior knowledge that we hold that might be relevant to the event.  So we keep track of the words occurring in all the mails, ham mails and spam mails and we calculate the probability of the occurrences of each word in both spam and ham mail. This learning is done on the contents of the training file. Then this knowledge will be applied on the test files mails to predict the target variable (spam/ham).


Metrics:

Metrics that is used to evaluate the performance of the model are listed below:
•	Accuracy, Precision
•	Recall
•	Root mean squared error
•	Mean absolute error


Accuracy:

•	Accuracy score:  72.472472472472475
•	R^2:  -0.13046251337338477
•	Mean squared error:  0.27527527527527529
•	Root mean squared error: 0.524666823113
•	Mean absolute error:  0.27527527527527529

