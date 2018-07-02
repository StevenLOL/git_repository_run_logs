
# coding: utf-8

# In[40]:


import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVC,LinearSVR
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plot
get_ipython().run_line_magic('matplotlib', 'inline')
import os
import glob
import numpy as np

from sklearn.decomposition import TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_extraction.text import TfidfVectorizer
import lightgbm as lgb
from fastText import train_unsupervised
import datetime
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import sqrt
import datetime
#import matplotlib.pyplot as plt
import pandas as pd
import glob
import tqdm
import datetime
import keras
import numpy as np
#test getWindowedValue
import numpy as np
from numpy.lib.stride_tricks import as_strided
from keras.models import Sequential
from keras.layers.convolutional import Conv3D,Conv2D,MaxPooling1D,MaxPooling2D,MaxPooling3D,Conv1D
from keras.layers import Lambda,Multiply ,TimeDistributed
from keras.layers import GlobalAveragePooling1D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.recurrent import LSTM,GRU
from keras.layers.wrappers import TimeDistributed
from keras import initializers
from keras.engine import InputSpec, Layer
from keras.layers import Dense,Dropout,Flatten,Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam,SGD
import tensorflow as tf
import keras.backend.tensorflow_backend as K
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import Sequential,Model
from keras.optimizers import Adam,RMSprop
from keras.activations import tanh,relu
#from keras.utils import multi_gpu_model
from keras.layers.advanced_activations import PReLU,LeakyReLU,ELU
from keras.layers import Dense, Dropout, Activation, Flatten,LSTM,GRU,Input,InputLayer,Activation, Input,Conv1D,MaxPooling1D,GlobalAveragePooling1D
from keras.layers import Convolution2D, MaxPooling2D,TimeDistributed,Convolution1D,MaxPooling1D,concatenate, Average,BatchNormalization,GlobalMaxPool1D
from keras.utils import np_utils
from keras import losses
#from keras_tqdm import TQDMNotebookCallback
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.grid_search import GridSearchCV
import sklearn
import numpy as np
from tqdm import tqdm_notebook as tqdm
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pickle
from IPython.display import SVG

import keras

LengthOfInputSequences=60


# In[2]:


fileTrain='./train.txt'
fileDev='./dev.txt'


# In[3]:


from string import punctuation
def text_to_wordlist(text, remove_stop_words=True, stem_words=False):
    # Clean the text, with the option to remove stop_words and to stem words.
    
    # Convert words to lower case and split them
    #text = text.lower()

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"what's", "", text)
    text = re.sub(r"What's", "", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"I'm", "I am", text)
    text = re.sub(r" m ", " am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\0k ", "0000 ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e-mail", "email", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"quikly", "quickly", text)
    text = re.sub(r" usa ", " America ", text)
    text = re.sub(r" USA ", " America ", text)
    text = re.sub(r" u s ", " America ", text)
    text = re.sub(r" uk ", " England ", text)
    text = re.sub(r" UK ", " England ", text)
    text = re.sub(r"india", "India", text)
    text = re.sub(r"china", "China", text)
    text = re.sub(r"chinese", "Chinese", text) 
    text = re.sub(r"imrovement", "improvement", text)
    text = re.sub(r"intially", "initially", text)
    text = re.sub(r"quora", "Quora", text)
    text = re.sub(r" dms ", "direct messages ", text)  
    text = re.sub(r"demonitization", "demonetization", text) 
    text = re.sub(r"actived", "active", text)
    text = re.sub(r"kms", " kilometers ", text)
    text = re.sub(r"KMs", " kilometers ", text)
    text = re.sub(r" cs ", " computer science ", text) 
    text = re.sub(r" upvotes ", " up votes ", text)
    text = re.sub(r" iPhone ", " phone ", text)
    text = re.sub(r"\0rs ", " rs ", text) 
    text = re.sub(r"calender", "calendar", text)
    text = re.sub(r"ios", "operating system", text)
    text = re.sub(r"gps", "GPS", text)
    text = re.sub(r"gst", "GST", text)
    text = re.sub(r"programing", "programming", text)
    text = re.sub(r"bestfriend", "best friend", text)
    text = re.sub(r"dna", "DNA", text)
    text = re.sub(r"III", "3", text) 
    text = re.sub(r"the US", "America", text)
    text = re.sub(r"Astrology", "astrology", text)
    text = re.sub(r"Method", "method", text)
    text = re.sub(r"Find", "find", text) 
    text = re.sub(r"banglore", "Banglore", text)
    text = re.sub(r" J K ", " JK ", text)
    
    # Remove punctuation from text
    text = ''.join([c for c in text if c not in punctuation])
    
    # Optionally, remove stop words
    if remove_stop_words:
        text = text.split()
        text = [w for w in text if not w in stop_words]
        text = " ".join(text)
    
    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
    
    # Return a list of words
    return(text)


# In[4]:


def loadData(fname):
    return pd.read_csv(fname, sep='\t',header=-1)


#22ste

def fitlerLine(lin):
    lin=lin.lower()
    for k in ['.',',','?',"'s","'t",',']:
        lin=lin.replace(k,' '+k+' ')
    #for k in ['.',',','?','!']:
    #    lin=lin.replace(k,' ')
    while '  ' in lin:
        lin=lin.replace('  ',' ')
    return lin.strip()


# In[5]:


df=loadData(fileTrain)


# In[6]:


df.describe


# In[7]:


trainxRaw=list(map(fitlerLine,df[0]))


# In[8]:


print(trainxRaw[:2])


# In[9]:


trainyRaw=df[1].values
print(len(trainyRaw),trainyRaw[:10])


# In[10]:


devdata=loadData(fileDev)
devxRaw=list(map(fitlerLine,devdata[0]))
devyRaw=devdata[1].values
print(len(devxRaw))
print(len(devyRaw),devyRaw[:10])


# from langid.langid import LanguageIdentifier, model
# identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)

# for r in devxRaw[10:]:
#     print(r,identifier.classify(r))
#     break;
# import langid
# langid.set_languages(['de','fr','it','en'])
# idscores=[langid.classify(r) for r in devxRaw]

# print(([s[0] for s in idscores]))
# print(devyRaw)
# #print(([s[1] for s in idscores]))

# In[11]:


a='1 2  3'
print(a.split())


# In[87]:


def mytoken(lin):
    return lin.split()
tfv = TfidfVectorizer(min_df=2,use_idf=1,
                      smooth_idf=1,ngram_range=(1,3),
                     )#analyzer='char_wb') #,stop_words='english')


trainx=tfv.fit_transform(trainxRaw)
evalx=tfv.transform(devxRaw)
print (tfv.get_feature_names()[:10])
print (tfv.get_feature_names()[-10:])
print (trainx.shape,evalx.shape)


#tsvd=TruncatedSVD(n_components=400,random_state=2016)   # this gives similar results as to Semeval , try n_components=600
#trainx=tsvd.fit_transform(trainx)
#evalx=tsvd.transform(evalx)
#clf=LinearDiscriminantAnalysis()


# In[12]:


trainy=trainyRaw
evaly=devyRaw


# In[257]:


with open('features.txt','w') as fout:
    fout.writelines('\n'.join(tfv.get_feature_names()))


# In[88]:


clf=LinearSVC()

clf.fit(trainx,trainy)
predictValue=clf.predict(evalx)
print(confusion_matrix(evaly,predictValue))
print(classification_report(evaly,predictValue))


# In[92]:


sf=sklearn.model_selection.StratifiedKFold(20)
cvcount=0
for trainindex,devindex in sf.split(trainx,trainy):
    cvtrainx,cvtrainy=trainx[trainindex],trainy[trainindex]
    cvdevx,cvdevy=trainx[devindex],trainy[devindex]
    clf=LinearSVC()
    clf.fit(cvtrainx,cvtrainy)
    predictValue=clf.predict(cvdevx)
    print('===============%d==================='%(cvcount))
    cvcount+=1
    print(confusion_matrix(cvdevy,predictValue))
    print(classification_report(cvdevy,predictValue))
    predictValue=clf.predict(evalx)
    print(confusion_matrix(evaly,predictValue))
    print(classification_report(evaly,predictValue))


# In[266]:


print(trainx.shape,trainy.shape,evalx.shape,evaly.shape)
clf=lgb.LGBMClassifier(num_leaves=150,n_jobs=12,
                                learning_rate=0.1,n_estimators=1000,silent=False)

clf.fit(trainx,trainy,early_stopping_rounds=20,
        eval_set=(evalx,evaly),
        verbose=True)
predictValue=clf.predict(evalx)
print(confusion_matrix(evaly,predictValue))
print(classification_report(evaly,predictValue))


# In[13]:


#fasttext
import fastText


# In[14]:


print(type(trainyRaw),type(trainxRaw))


# In[15]:


rawdutFile='./data/rawdut.txt'
rawbelFile='./data/rawbel.txt'
trainlabeled='./data/trainlabeled.txt'
devlabeled='./data/devlabeled.txt'
preFiexed='__label__'


# In[16]:


with open(trainlabeled,'w') as fout:
    for x,y in zip(trainxRaw,trainy):
        fout.write('__label__{} {}\n'.format(y,x))

with open(devlabeled,'w') as fout:
    for x,y in zip(devxRaw,devyRaw):
        fout.write('__label__{} {}\n'.format(y,x))


# In[17]:



rawDu=np.array(trainxRaw)[trainyRaw=='DUT']
os.system('mkdir -p data')
with open(rawdutFile,'w') as fout:
    fout.writelines('\n'.join(map(str,rawDu)))
    
rawBel=np.array(trainxRaw)[trainyRaw=='BEL']
os.system('mkdir -p data')
with open('./data/rawbel.txt','w') as fout:
    fout.writelines('\n'.join(map(str,rawBel)))


# In[18]:


dir(fastText)


# In[26]:


classifier = fastText.train_supervised(trainlabeled)

result = classifier.test(devlabeled)
#print ('P@1:', result.precision)
#print ('R@1:', result.recall)
#print ('Number of examples:', result.nexamples)


# In[27]:


print(result)


# In[28]:


#simple CNN


# In[43]:


#print(trainx[0])
tfidfDict={s:sindex for sindex,  s in enumerate(  tfv.get_feature_names())}
#for s in trainxRaw[0].split():
#    if s in tfidfDict:
#        print(s,tfidfDict[s])

def toCnnIndexAllinOne(lin):
    rv=[]
    for s in lin.split()[:LengthOfInputSequences]:
        if s in tfidfDict:
            rv.append(tfidfDict[s])
    while(len(rv)<LengthOfInputSequences):
        rv.append(0)
    return rv

trainCnnIndex=list(map(toCnnIndexAllinOne,trainxRaw))
devCnnIndex=list(map(toCnnIndexAllinOne,devxRaw))


# In[61]:


#build w2v and use the w2v index
#train a w2v and change the trainCnnIndex/devCnnIndex
modelW2V=train_unsupervised('./data/rawtrainNoLb.txt',model='skipgram',minCount=10,dim=10)
tfidfDict={s:sindex for sindex,  s in enumerate(  modelW2V.get_words())}
#for s in trainxRaw[0].split():
#    if s in tfidfDict:
#        print(s,tfidfDict[s])



# In[62]:


def toCnnIndexAllinOne(lin):
    rv=[]
    for s in lin.split()[:LengthOfInputSequences]:
        if s in tfidfDict:
            rv.append(tfidfDict[s])
    while(len(rv)<LengthOfInputSequences):
        rv.append(0)
    return rv

trainCnnIndex=list(map(toCnnIndexAllinOne,trainxRaw))
devCnnIndex=list(map(toCnnIndexAllinOne,devxRaw))

print(modelW2V.get_output_matrix().shape[0])  #default minCount=5 42836    minCount=3 61282


# In[50]:


print(trainCnnIndex[:2])


# In[80]:




#CNN
from keras.layers import Embedding
from keras.utils import plot_model
from keras.utils.vis_utils import model_to_dot



#https://github.com/keras-team/keras/blob/master/examples/pretrained_word_embeddings.py 

def getCNN():
    input1=keras.layers.Input(shape=(LengthOfInputSequences,))    
    
    kernel_size = 3
    filters = 64
    pool_size = 3
    lstm_output_size=64
    '''
    x1=Embedding(input_dim=len(tfidfDict), #modelW2V.get_output_matrix().shape[0],
                input_length=LengthOfInputSequences,
                output_dim=100,     #modelW2V.get_output_matrix().shape[1],
                 trainable=True,
                #weights=[modelW2V.get_output_matrix()],
                #
               )(input1)
               '''
    x1=Embedding(input_dim=modelW2V.get_output_matrix().shape[0],
                input_length=LengthOfInputSequences,
                output_dim=modelW2V.get_output_matrix().shape[1],      
                 weights=[modelW2V.get_output_matrix()],
               )(input1)
    #x1=Dropout(0.2)(x1)
    x1=Conv1D(filters=filters,kernel_size=kernel_size)(x1)
    #x1=Dropout(0.2)(x1)
    #x1=Conv1D(filters=filters,kernel_size=kernel_size)(x1)
    #x1=Dropout(0.2)(x1)
    #x1=Conv1D(filters=filters,kernel_size=kernel_size)(x1)
    x1=MaxPooling1D(pool_size=pool_size)(x1)
    x1=Conv1D(filters=filters,kernel_size=kernel_size)(x1)
    x1=MaxPooling1D(pool_size=pool_size)(x1)
    #x1=Dropout(0.2)(x1)
    x1=LSTM(lstm_output_size)(x1)
    #x1=keras.layers.
    #x1=Flatten()(x1)
    
    
    
   
    
    
    
    
    
    #x1=Dropout(0.2)(x1)
    addLayer=Dense(512)(x1)
    output=Dense(2)(addLayer)
    output=Activation('softmax')(output)
    
    
    
    seq=keras.models.Model(inputs=[input1],outputs=output)
    
    seq.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
    seq.summary()
    return seq



def getMLP():
    input1=keras.layers.Input(shape=(LengthOfInputSequences,))    
    
    kernel_size = 5
    filters = 32
    pool_size = 4
    lstm_output_size=512
    '''
    x1=Embedding(input_dim=len(tfidfDict), #modelW2V.get_output_matrix().shape[0],
                input_length=LengthOfInputSequences,
                output_dim=100,     #modelW2V.get_output_matrix().shape[1],
                 trainable=True,
                #weights=[modelW2V.get_output_matrix()],
                #
               )(input1)
               '''
    x1=Embedding(input_dim=modelW2V.get_output_matrix().shape[0],
                input_length=LengthOfInputSequences,
                output_dim=modelW2V.get_output_matrix().shape[1],      
                weights=[modelW2V.get_output_matrix()],
                #trainable=False,
               )(input1)
    #x1=Flatten()(x1)
    #x1=LSTM(512)(x1)
    #x1=Dropout(0.2)(x1)
    #addLayer=Dense(512)(x1)
    output=Dense(2)(x1)
    output=Activation('softmax')(output)
    
    
    
    seq=keras.models.Model(inputs=[input1],outputs=output)
    
    #seq.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
    seq.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
    seq.summary()
    return seq
def getCNNIMDB():
    model = Sequential()
    kernel_size = 5
    filters = 64
    pool_size = 4
    lstm_output_size=70
    model.add(Embedding(modelW2V.get_output_matrix().shape[0], 
                        modelW2V.get_output_matrix().shape[1], 
                        input_length=LengthOfInputSequences,
                        weights=[modelW2V.get_output_matrix()]
                       ))
    model.add(Dropout(0.25))
    model.add(Conv1D(filters,
                     kernel_size,
                     padding='valid',
                     activation='relu',
                     strides=1))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(LSTM(lstm_output_size))
    model.add(Dense(200,activation='relu'))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model
from keras import backend as K
K.clear_session()
crf=getCNNIMDB()

#plot_model(crf, to_file='lstm-mlp.png',show_shapes=True)
SVG(model_to_dot(crf, show_shapes=True).create(prog='dot', format='svg'))


# In[86]:


trainCnnIndex=np.array(trainCnnIndex)
print(trainCnnIndex.shape)
devCnnIndex=np.array(devCnnIndex)
print(devCnnIndex.shape)
cnnTrainy=[0 if s=='DUT' else 1 for s in trainy]
cnnDevy=[0 if s=='DUT' else 1 for s in evaly]
cnnTrainyCat=keras.utils.to_categorical(cnnTrainy)
cnnDevyCat= keras.utils.to_categorical(cnnDevy)
import os
os.system('mkdir -p models')
#fullTrainX=np.vstack((trainCnnIndex,devCnnIndex))
#fullTrainY=np.concatenate((cnnTrainyCat,cnnDevyCat))
sf=sklearn.model_selection.StratifiedKFold(20)
for trainindex,devindex in sf.split(trainCnnIndex,cnnTrainy):
    K.clear_session()
    crf=getCNNIMDB()
    modelFile='./models/cnn_%d.hdf5'%(i)
    callbacks = [EarlyStopping(monitor='val_loss',
                               patience=2,
                               verbose=1,
                               min_delta=0.01,
                               mode='min'),
                 ReduceLROnPlateau(monitor='val_loss',
                                   factor=0.1,
                                   patience=0,
                                   verbose=1,
                                   epsilon=0.0001,
                                   mode='min'),
                 ModelCheckpoint(monitor='val_loss',
                                 filepath=modelFile,
                                 save_best_only=True,
                                 save_weights_only=True,
                                 verbose=1,
                                 mode='min'),
                 ]
    '''
    crf.fit(trainCnnIndex,
            cnnTrainyCat,
            validation_data=(devCnnIndex,cnnDevyCat),
            epochs=100,batch_size=256,
            callbacks=callbacks)
            '''
    crf.fit(trainCnnIndex[trainindex],
            cnnTrainyCat[trainindex],       
            validation_data=(trainCnnIndex[devindex],cnnTrainyCat[devindex]),
            epochs=100,batch_size=256,
            callbacks=callbacks,
            )
    crf.load_weights(modelFile)
    predictValue=crf.predict(devCnnIndex)
    
    #predict=[0 if s <th else 1 for s in predictValue]
    predict=np.argmax(predictValue,axis=1)
    print('------------%d-----------------'%(i))
    print(confusion_matrix(cnnDevy,predict))
    print(classification_report(cnnDevy,predict))
    #print(predictValue.shape)
    #fpr, tpr, thresholds = metrics.roc_curve(cnnDevy, predictValue)
    #print(thresholds)
    #print(i,metrics.auc(fpr, tpr))


# In[78]:


for th in (sorted(set(predictValue.squeeze()))):
    predict=[0 if s <th else 1 for s in predictValue]
    #predict=np.argmax(predictValue,axis=1)
    print('------------%d--%f---------------'%(i,th))
    print(confusion_matrix(cnnDevy,predict))
    print(classification_report(cnnDevy,predict))
    pass


# In[84]:


predict=np.argmax(predictValue,axis=1)
print(confusion_matrix(cnnDevy,predict))
print(classification_report(cnnDevy,predict))


# In[314]:


from sklearn import metrics
predictValue=crf.predict(devCnnIndex)
print(predictValue[:10])
print(cnnDevy[:10])

fpr, tpr, thresholds = metrics.roc_curve(cnnDevy, predictValue)
#print(thresholds)
print(metrics.auc(fpr, tpr))


# In[ ]:


#end of simple CNN


# In[106]:


#word embedings
lenDUs=[len(s) for s in rawDu]
print(len(lenDUs))
plt.hist(lenDUs,bins=1000)
plt.xlim(0,300)


# In[105]:


lenBELs=[len(s) for s in rawBel]
plt.hist(lenBELs,bins=1000)
plt.xlim(0,300)


# In[104]:


lenDUs=[len(s.split()) for s in rawDu]
print(len(lenDUs))
print(len(set(lenDUs)),sorted(set(lenDUs)))
plt.figure()
plt.hist(lenDUs,bins=100)
plt.xlim(0,60)
lenBELs=[len(s.split()) for s in rawBel]
print(len(set(lenBELs)),sorted(set(lenBELs)))
plt.figure()
plt.hist(lenBELs,bins=100)
plt.xlim(0,60)
plt.figure()
plt.hist(lenDUs,bins=100)
plt.xlim(0,60)


# In[120]:


alldu=' '.join(rawDu)
allbel=' '.join(rawBel)


# In[121]:


for s in [',','.','?','!']:
    print(alldu.count(s),allbel.count(s))


# In[40]:


from fastText import train_unsupervised
modelDUT=train_unsupervised(rawdutFile,model='skipgram')
modelDUT.save_model('./data/model.dut')
modelBEL=train_unsupervised(rawbelFile,model='skipgram')
modelBEL.save_model('./data/model.bel')


# In[42]:


print(modelDUT.get_word_vector('king'))


# In[141]:


def raw2MeanVector2(modela,modelb,texts):
    vss=[]
    if type(texts)!=list:
        texts=[texts]
    for t in texts:
        v1s,v2s=[],[]
        for w in t.split():
            v1s.append(modela.get_word_vector(w))
            v2s.append(modelb.get_word_vector(w))
        v1s=np.mean(np.array(v1s),axis=0)   
        v2s=np.mean(np.array(v2s),axis=0)
        vss.append( np.hstack((v1s,v2s)))
    return np.array(vss)

def raw2MeanVector(modela,modelb,texts):
    vss=[]
    if type(texts)!=list:
        texts=[texts]
    for t in texts:
        v1s,v2s=modela.get_sentence_vector(t),modelb.get_sentence_vector(t)        
        vss.append( np.hstack((v1s,v2s)))
    return np.array(vss)


meanVector=raw2MeanVector(modelDUT,modelBEL,trainxRaw)
print (meanVector.shape)


# In[140]:


modelDUT.get_sentence_vector(trainxRaw[0])


# In[142]:


devMeanVector=raw2MeanVector(modelDUT,modelBEL,devxRaw)


# In[144]:


print(meanVector.shape,devMeanVector.shape)


# In[143]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
clf=LinearDiscriminantAnalysis()

clf.fit(meanVector,trainy)
predictValue=clf.predict(devMeanVector)
print(confusion_matrix(evaly,predictValue))
print(classification_report(evaly,predictValue))


# In[146]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
clf=lgb.LGBMClassifier(num_leaves=150,n_jobs=12,
                                learning_rate=0.1,n_estimators=1000,silent=False)

clf.fit(meanVector,trainy,early_stopping_rounds=20,
        eval_set=(devMeanVector,evaly),
        verbose=True)
predictValue=clf.predict(devMeanVector,) #clf.best_iteration_
print(confusion_matrix(evaly,predictValue))
print(classification_report(evaly,predictValue))


# In[67]:


clf=lgb.LGBMClassifier(num_leaves=150,n_jobs=12,
                                learning_rate=0.1,n_estimators=1000,silent=False)

clf.fit(trainx,trainy,early_stopping_rounds=20,
        eval_set=(evalx,evaly),
        verbose=True)
predictValue=clf.predict(evalx)
print(confusion_matrix(evaly,predictValue))
print(classification_report(evaly,predictValue))


# In[156]:


methods=[method_name for method_name in dir(modelDUT)]
dutMatrix=modelDUT.get_output_matrix()
dutWords=modelDUT.get_words()
print(dutMatrix.shape,len(dutWords))
belMatrix=modelBEL.get_output_matrix()
belWords=modelBEL.get_words()
print(belMatrix.shape,len(belWords))


# In[247]:




#CNN
from keras.layers import Embedding
from keras.utils import plot_model
from keras.utils.vis_utils import model_to_dot



#https://github.com/keras-team/keras/blob/master/examples/pretrained_word_embeddings.py 

def getCNN():
    input1=keras.layers.Input(shape=(70,))    
    LengthOfInputSequences=70
    kernel_size = 5
    filters = 32
    pool_size = 4
    lstm_output_size=32
    x1=Embedding(input_dim=dutMatrix.shape[0],
                input_length=LengthOfInputSequences,
                output_dim=dutMatrix.shape[1],
                weights=[dutMatrix],
                trainable=False
               )(input1) 
    #x1=Dropout(0.2)(x1)
    x1=Conv1D(filters=filters,kernel_size=kernel_size)(x1)
    x1=MaxPooling1D(pool_size=pool_size)(x1)
    x1=Conv1D(filters=filters,kernel_size=kernel_size)(x1)
    x1=MaxPooling1D(pool_size=pool_size)(x1)
    #x1=Dropout(0.2)(x1)
    x1=LSTM(lstm_output_size)(x1)
    
    
    
    input2=keras.layers.Input(shape=(70,))
    SizeOfTheVocabulary=len(dutWords)
    LengthOfInputSequences=70
    x2=Embedding(input_dim=belMatrix.shape[0],
                input_length=LengthOfInputSequences,
                output_dim=belMatrix.shape[1],
                weights=[belMatrix],
                trainable=False
               )(input2)
    #x2=Dropout(0.2)(x2)
    x2=Conv1D(filters=filters,kernel_size=kernel_size)(x2)
    x2=MaxPooling1D(pool_size=pool_size)(x2)
    x2=Conv1D(filters=filters,kernel_size=kernel_size)(x2)
    x2=MaxPooling1D(pool_size=pool_size)(x2)
    #x2=Dropout(0.2)(x2)
    x2=LSTM(lstm_output_size)(x2)
    
    
    
    
    
    
    addLayer=keras.layers.Concatenate()([x1,x2])
    x1=Dropout(0.2)(x1)
    addLayer=Dense(512)(addLayer)
    output=Dense(1)(addLayer)
    output=Activation('sigmoid')(output)
    
    
    
    seq=keras.models.Model(inputs=[input1,input2],outputs=output)
    seq.summary()
    seq.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
    return seq
from keras import backend as K
K.clear_session()
crf=getCNN()

#plot_model(crf, to_file='lstm-mlp.png',show_shapes=True)
SVG(model_to_dot(crf, show_shapes=True).create(prog='dot', format='svg'))


# In[167]:


print(dutWords[0],belWords[0])


# In[180]:


sa=set([1,2,3])
sb=set([2,4])
print(sa-sb)
print(sb-sa)
print(sa|sb)


# In[175]:


print(len(set(dutWords)-set(belWords)))
print(len(set(belWords)-set(dutWords)))
print(len(set(belWords)|set(dutWords)))


# In[235]:


w2vDim=100
def textToIndex(wordList,texts,maxlen=70):
    rv=[]
    wordDict={w:windex for windex,w in enumerate(wordList)}
    for t in texts:
        lenWords=[]
        for w in t.split():
            
            if w in wordDict:
                lenWords.append(wordDict[w])
            #else:
            #    lenWords.append(0)
        while(len(lenWords)<maxlen):
            lenWords.append(0)
        rv.append(lenWords[:maxlen])
    return np.array(rv)

trainDuIndex=textToIndex(dutWords,trainxRaw)
trainBelIndex=textToIndex(dutWords,trainxRaw)
                
devDuIndex=textToIndex(dutWords,devxRaw)
devBelIndex=textToIndex(dutWords,devxRaw)        
    


# In[236]:


print(trainDuIndex.shape,trainDuIndex[0])


# In[49]:


cnnTrainy=[0 if s=='DUT' else 1 for s in trainy]
cnnDevy=[0 if s=='DUT' else 1 for s in evaly]


# In[231]:


print (cnnTrainy[:10])


# In[248]:


crf.fit([trainDuIndex,trainBelIndex],
        cnnTrainy,
        validation_data=([devDuIndex,devBelIndex],cnnDevy),epochs=9,batch_size=256)


# In[249]:


from sklearn import metrics
predictValue=crf.predict([devDuIndex,devBelIndex])
print(predictValue[:10])
print(cnnDevy[:10])

fpr, tpr, thresholds = metrics.roc_curve(cnnDevy, predictValue)
#print(thresholds)
print(metrics.auc(fpr, tpr))
#print(confusion_matrix(cnnDevy,predictValue))
#print(classification_report(cnnDevy,predictValue))


# In[223]:


plt.hist(predictValue)

