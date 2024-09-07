import numpy as np
import pandas as pd
from keras.layers import Dense, Input, Dropout
from keras.layers.merge import concatenate
from keras.optimizers import SGD
from keras.models import Model
from keras.regularizers import l2
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from keras.utils import np_utils
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef,accuracy_score, precision_score,recall_score

import lightgbm as lgb
import time

start = time.time()
def define_model():
    input_1 = Input(shape=(993, ), name='Protein_a')
    p11 = Dense(512, activation='relu', kernel_initializer='glorot_normal', name='ProA_feature_1', kernel_regularizer=l2(0.01))(input_1)
    p11 = Dropout(0.2)(p11)
    
    p12 = Dense(256, activation='relu', kernel_initializer='glorot_normal', name='ProA_feature_2', kernel_regularizer=l2(0.01))(p11)
    p12 = Dropout(0.2)(p12)
    
    p13= Dense(128, activation='relu', kernel_initializer='glorot_normal', name='ProA_feature_3', kernel_regularizer=l2(0.01))(p12)
    p13 = Dropout(0.2)(p13)
    
    
    input_2 = Input(shape=(993, ), name='Protein_b')
    p21 = Dense(512, activation='relu', kernel_initializer='glorot_normal', name='ProB_feature_1', kernel_regularizer=l2(0.01))(input_2)
    p21 = Dropout(0.2)(p21)
    
    p22 = Dense(256, activation='relu', kernel_initializer='glorot_normal', name='ProB_feature_2', kernel_regularizer=l2(0.01))(p21)
    p22 = Dropout(0.2)(p22)
    
    p23= Dense(128, activation='relu', kernel_initializer='glorot_normal', name='ProB_feature_3', kernel_regularizer=l2(0.01))(p22)
    p23 = Dropout(0.2)(p23)


    # subtract
    merged = subtract([p14,p24], name='merged_protein1_2')

    pre_output = Dense(64, activation='relu', kernel_initializer='glorot_normal', name='Merged_feature_1')(merged)
    pre_output = Dense(32, activation='relu', kernel_initializer='glorot_normal', name='Merged_feature_2')(pre_output)
    pre_output = Dense(16, activation='relu', kernel_initializer='he_uniform', name='Merged_feature_3')(pre_output)


    
    pre_output=Dropout(0.2)(pre_output)

    output = Dense(1, activation='sigmoid', name='output')(pre_output)
    model = Model(input=[input_1, input_2], output=output)
   
    sgd = SGD(lr=0.01, momentum=0.9, decay=0.001)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model



df_pos= pd.read_csv('Positive.csv',header=None)
df_neg = pd.read_csv('Negative.csv',header=None)
#df_neg=df_neg.sample(n=min(len(df_pos),len(df_neg)))
#df_pos=df_pos.sample(n=min(len(df_pos),len(df_neg)))
df_neg['Status'] = 0
df_pos['Status'] = 1
df_neg=df_neg.sample(n=len(df_pos))

df = pd.concat([df_pos,df_neg])
df = df.reset_index()
df=df.sample(frac=1)
df = df.iloc[:,1:]

X = df.iloc[:,0:1986].values
y = df.iloc[:,1986:].values
Trainlabels=y
scaler = StandardScaler().fit(X)
#scaler = RobustScaler().fit(X)
X = scaler.transform(X)


X1_train = X[:, :993]
X2_train = X[:, 993:]


kf=StratifiedKFold(n_splits=5)


accuracy1 = []
specificity1 = []
sensitivity1 = []
precision1=[]
recall1=[]

m_coef1=[]
fpr1 = dict()
tpr1 = dict()
roc_auc1 = dict()
o=0
for train, test in kf.split(X,y):
    global model
    model=define_model()
    o=o+1

    model.fit([X1_train[train],X2_train[train]],y[train],epochs=50,batch_size=64,verbose=1)
    y_test=y[test]
    y_score = model.predict([X1_train[test],X2_train[test]])
    
    fpr1["micro"], tpr1["micro"], _ = roc_curve(y_test[:,0],  y_score[:,0])
    roc_auc1["micro"] = auc(fpr1["micro"], tpr1["micro"])
    
    y_score=y_score[:,0]
    
    for i in range(0,len(y_score)):
        if(y_score[i]>0.5):
            y_score[i]=1
        else:
            y_score[i]=0
            
    cm1=confusion_matrix(y[test][:,0],y_score)
    acc1 = accuracy_score(y[test][:,0], y_score, sample_weight=None)
    spec1= (cm1[0,0])/(cm1[0,0]+cm1[0,1])
    sens1 = recall_score(y[test][:,0], y_score, sample_weight=None)
    prec1=precision_score(y[test][:,0], y_score, sample_weight=None)
    
    sensitivity1.append(sens1)
    specificity1.append(spec1)
    accuracy1.append(acc1)
    precision1.append(prec1)
    
    coef1=matthews_corrcoef(y[test][:,0], y_score, sample_weight=None)
    m_coef1.append(coef1)
    
    fprdnn=pd.DataFrame.from_dict(fpr1)
    tprdnn=pd.DataFrame.from_dict(tpr1)
    
fprdnn.to_csv('fprDNN.csv',header=False, index=False)
tprdnn.to_csv('tprDNN.csv',header=False, index=False)
    
mean_acc1=np.mean(accuracy1)
std_acc1=np.std(accuracy1)
var_acc1=np.var(accuracy1)
print("Accuracy1:"+str(mean_acc1)+" Â± "+str(std_acc1))
print("Accuracy_Var:"+str(mean_acc1)+" Â± "+str(var_acc1))
mean_spec1=np.mean(specificity1)
std_spec1=np.std(specificity1)
print("Specificity1:"+str(mean_spec1)+" Â± "+str(std_spec1))
mean_sens1=np.mean(sensitivity1)
std_sens1=np.std(sensitivity1)
print("Sensitivity1:"+str(mean_sens1)+" Â± "+str(std_sens1))
mean_prec1=np.mean(precision1)
std_prec1=np.std(precision1)
print("Precison1:"+str(mean_prec1)+" Â± "+str(std_prec1))

mean_coef1=np.mean(m_coef1)
std_coef1=np.std(m_coef1)
print("MCC1:"+str(mean_coef1)+" Â± "+str(std_coef1))

print("AUC1:"+str(roc_auc1))


end1 = time.time()
end11=end1 - start
print(f"Runtime of the program is {end1 - start}")



intermediate_layer_model = Model(inputs=model.input,outputs=model.get_layer('merged_protein1_2').output)

intermediate_output_p1 = intermediate_layer_model.predict([X1_train,X2_train])  
p_merge=pd.DataFrame(intermediate_output_p1)    
X_train_feat=pd.concat((p_merge,pd.DataFrame(pd.DataFrame(Trainlabels))),axis=1,ignore_index=True)
X_train_feat.to_csv('X_train.csv',header=False, index=False)


Train=pd.read_csv("X_train.csv",header=None)
Train=Train.sample(frac=1)
X=Train.iloc[:,0:128].values
y=Train.iloc[:,128:].values

scaler=RobustScaler()
X=scaler.fit_transform(X)
kf=StratifiedKFold(n_splits=5)


accuracy = []
specificity = []
sensitivity = []
precision=[]
recall=[]
auc_list=[]
m_coef=[]

fpr = dict()
tpr = dict()
roc_auc = dict()
o=0
for train, test in kf.split(X,y):
    o=o+1
    model=lgb(n_estimators=100)

    hist=model.fit(X[train], y[train],eval_set=[(X[test], y[test])])
    y_score=model.predict_proba(X[test])
    y_test=np_utils.to_categorical(y[test]) 
    
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test[:,0].ravel(), y_score[:,0].ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    coef=matthews_corrcoef(y_test.argmax(axis=1), y_score.argmax(axis=1), sample_weight=None)
    m_coef.append(coef)
    
    cm1=confusion_matrix(y_test.argmax(axis=1), y_score.argmax(axis=1))
    acc = (cm1[0,0]+cm1[1,1])/(cm1[0,0]+cm1[0,1]+cm1[1,0]+cm1[1,1])
    spec= (cm1[0,0])/(cm1[0,0]+cm1[0,1])
    sens = (cm1[1,1])/(cm1[1,0]+cm1[1,1])
    prec=cm1[1,1]/(cm1[1,1]+cm1[0,1])
    rec=cm1[1,1]/(cm1[1,1]+cm1[1,0])
    sensitivity.append(sens)
    specificity.append(spec)
    accuracy.append(acc)
    precision.append(prec)
    recall.append(rec)
    
    fprlgb=pd.DataFrame.from_dict(fpr)
    tprlgb=pd.DataFrame.from_dict(tpr)
    
fprlgb.to_csv('fprdnn_lgb.csv',header=False, index=False)
tprlgb.to_csv('tprdnn_lgb.csv',header=False, index=False)   
 
mean_acc=np.mean(accuracy)
std_acc=np.std(accuracy)
var_acc=np.var(accuracy)
print("Accuracy:"+str(mean_acc)+" Â± "+str(std_acc))
print("Accuracy_Var:"+str(mean_acc)+" Â± "+str(var_acc))
mean_spec=np.mean(specificity)
std_spec=np.std(specificity)
print("Specificity:"+str(mean_spec)+" Â± "+str(std_spec))
mean_sens=np.mean(sensitivity)
std_sens=np.std(sensitivity)
print("Sensitivity:"+str(mean_sens)+" Â± "+str(std_sens))
mean_prec=np.mean(precision)
std_prec=np.std(precision)
print("Precison:"+str(mean_prec)+" Â± "+str(std_prec))
mean_rec=np.mean(recall)
std_rec=np.std(recall)
print("Recall:"+str(mean_rec)+" Â± "+str(std_rec))
mean_coef=np.mean(m_coef)
std_coef=np.std(m_coef)
print("MCC:"+str(mean_coef)+" Â± "+str(std_coef))

print("AUC:"+str(roc_auc))



