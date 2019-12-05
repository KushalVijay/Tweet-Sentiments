import nltk
import random
import pickle
from nltk.tokenize import word_tokenize
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB , BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import LinearSVC,SVC,NuSVC
from nltk.classify import ClassifierI
from statistics import mode


class VoteClassifier(ClassifierI):
    def __init__(self,*classifiers):
        self._classifiers = classifiers
    
    def classify(self,features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)
    
    def confidence(self,features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        
        choice_votes = votes.count(mode(votes))
        conf = choice_votes/len(votes)
        return conf

        
short_pos = open("Live Sentiment Analysis/positive.txt","r").read()
short_neg = open("Live Sentiment Analysis/negative.txt","r").read()
documents = []
all_words = []

allowed_words_types = ["J"]

for p in short_pos.split('\n'):
    documents.append((p,"pos"))
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_words_types:
            all_words.append(w[0].lower())

for p in short_neg.split('\n'):
    documents.append((p,"neg"))
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_words_types:
            all_words.append(w[0].lower())


save_documents = open("Live Sentiment Analysis/Pickle data/documents.pickle","wb")
pickle.dump(documents,save_documents)
save_documents.close()


all_words = nltk.FreqDist(all_words)
#print(all_words.most_common(15))

word_features = list(all_words.keys())[:5000]

save_word_features = open("Live Sentiment Analysis/Pickle data/word_features5k.pickle","wb")
pickle.dump(word_features,save_word_features)
save_word_features.close()


def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    
    return features


featuresets = [(find_features(rev),category) for (rev,category) in documents]

random.shuffle(featuresets)

save_featuresets = open("Live Sentiment Analysis/Pickle data/featuresets.pickle","wb")
pickle.dump(featuresets,save_featuresets)
save_featuresets.close()

training_set = featuresets[:1000]
testing_set = featuresets[1000:]


MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier Naive Bayes Algo accuracy:",(nltk.classify.accuracy(MNB_classifier,testing_set))*100)

save_classifier = open("Live Sentiment Analysis/Pickle data/MNB_classifier.pickle","wb")
pickle.dump(MNB_classifier,save_classifier)
save_classifier.close()


BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB_classifier Naive Bayes Algo accuracy:",(nltk.classify.accuracy(BernoulliNB_classifier,testing_set))*100)

save_classifier = open("Live Sentiment Analysis/Pickle data/BernoulliNB.pickle","wb")
pickle.dump(BernoulliNB,save_classifier)
save_classifier.close()

# LogisticRegression,SGDClassifier
# LinearSVC,SVC,NuSVC

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier Naive Bayes Algo accuracy:",(nltk.classify.accuracy(LogisticRegression_classifier,testing_set))*100)

save_classifier = open("Live Sentiment Analysis/Pickle data/LogisticRegression.pickle","wb")
pickle.dump(LogisticRegression,save_classifier)
save_classifier.close()



NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC_classifier Naive Bayes Algo accuracy:",(nltk.classify.accuracy(NuSVC_classifier,testing_set))*100)

save_classifier = open("Live Sentiment Analysis/Pickle data/NuSVC.pickle","wb")
pickle.dump(NuSVC,save_classifier)
save_classifier.close()

voted_classfier = VoteClassifier(
                                LogisticRegression_classifier,
                                NuSVC_classifier,
                                MNB_classifier,
                                BernoulliNB_classifier,
                            
                                )
#print("voted_classfier Naive Bayes Algo accuracy:",(nltk.classify.accuracy(voted_classfier,testing_set))*100)

print("Classification:",voted_classfier.classify(testing_set[0][0]),"Confidence % :",voted_classfier.confidence(testing_set[0][0])*100)


# def sentiment(text):
#     feats = find_features(text)
#     return voted_classfier.classify(feats)
    