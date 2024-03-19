import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
import string
from wordcloud import WordCloud
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
from pandas.plotting import scatter_matrix

resumeDataSet = pd.read_csv('UpdatedResumeDataSet.csv' ,encoding='utf-8')
resumeDataSet['cleaned_resume'] = ''
resumeDataSet.head()
print("Thông tin chi tiết tập dữ liệu: ")
resumeDataSet.info()

print ("\nDanh sách các vị trí công việc:")
print (resumeDataSet['Category'].unique())

print ("\nSố lượng hồ sơ ứng với từng vị trí công việc")
print (resumeDataSet['Category'].value_counts())

#Làm sạch dữ liệu
def cleanResume(resumeText):
    resumeText = re.sub('http\S+\s*', ' ', resumeText)  
    resumeText = re.sub('RT|cc', ' ', resumeText) 
    resumeText = re.sub('#\S+', '', resumeText) 
    resumeText = re.sub('@\S+', '  ', resumeText)  
    resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText) 
    resumeText = re.sub(r'[^\x00-\x7f]',r' ', resumeText) 
    resumeText = re.sub('\s+', ' ', resumeText)  
    return resumeText
    
resumeDataSet['cleaned_resume'] = resumeDataSet.Resume.apply(lambda x: cleanResume(x))
resumeDataSet.head()
resumeDataSet_d=resumeDataSet.copy()

print("\n Kết quả sau khi làm sạch dữ liệu: ")
print(resumeDataSet_d)

oneSetOfStopWords = set(stopwords.words('english')+['``',"''"])
totalWords =[]
Sentences = resumeDataSet['Resume'].values
cleanedSentences = ""
for records in Sentences:
    cleanedText = cleanResume(records)
    cleanedSentences += cleanedText
    requiredWords = nltk.word_tokenize(cleanedText)
    for word in requiredWords:
        if word not in oneSetOfStopWords and word not in string.punctuation:
            totalWords.append(word)
    
wordfreqdist = nltk.FreqDist(totalWords)
mostcommon = wordfreqdist.most_common(50)
print("\n50 từ nổi bật nhất: ")
print(mostcommon)

#Mã hóa label
var_mod = ['Category']
le = LabelEncoder()
for i in var_mod:
    resumeDataSet[i] = le.fit_transform(resumeDataSet[i])
resumeDataSet.Category.value_counts()

print("\n Kết quả mã hóa label: ")
print(resumeDataSet.Category.value_counts())


requiredText = resumeDataSet['cleaned_resume'].values
requiredTarget = resumeDataSet['Category'].values

word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    stop_words='english')
word_vectorizer.fit(requiredText)
WordFeatures = word_vectorizer.transform(requiredText)

print ("\nFeature completed .....")

#Phân chia tập dữ liệu (80% train, 20% test)
X_train,X_test,y_train,y_test = train_test_split(WordFeatures,requiredTarget,random_state=42, test_size=0.2,
                                                 shuffle=True, stratify=requiredTarget)
print("\nKết quả phân chia tập dữ liệu train-test: ")
print(X_train.shape)
print(X_test.shape)

# Train và dự đoán, gắn nhãn cho tập kiểm thử
clf = OneVsRestClassifier(KNeighborsClassifier())
clf.fit(X_train, y_train) 
prediction = clf.predict(X_test)

print('\nĐộ chính xác tập huấn luyện: {:.2f}'.format(clf.score(X_train, y_train)))
print('Độ chính xác tập kiểm thử:     {:.2f}'.format(clf.score(X_test, y_test)))
print("\nBáo cáo chi tiết kết quả phân loại %s:\n%s\n" % (clf, metrics.classification_report(y_test, prediction)))
