import numpy as np
import pandas as pd
import io
from keras.models import load_model
import keras
from keras.models import Sequential
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from keras import backend as K

def predict(arr):

    inp1 = [int(arr['f1']),arr['f2'],arr['f3']]

    K.clear_session()

    classifier = load_model('assets/models/model1.h5')
    classifier2 = load_model('assets/models/model2.h5')
    classifier3 = load_model('assets/models/model3.h5')
    classifier4 = load_model('assets/models/model4.h5')
    classifier5 = load_model('assets/models/model5.h5')

    df3 = pd.read_csv('M7.csv')
    y1=df3.iloc[:, 4].values
    y2=df3.iloc[:, 5].values
    y3=df3.iloc[:, 6].values
    y4=df3.iloc[:, 7].values
    y5=df3.iloc[:, 8].values
    T = df3.iloc[:, 1:4].values
    T = np.vstack([T,inp1])

    labelencoder_1 = LabelEncoder()
    T[:, 1] = labelencoder_1.fit_transform(T[:, 1])
    labelencoder_2 = LabelEncoder()
    T[:, 2] = labelencoder_2.fit_transform(T[:, 2])

    ct = ColumnTransformer(
       [('one_hot_encoder', OneHotEncoder(), [1])],    
        remainder='passthrough'                         
    )
    T = np.array(ct.fit_transform(T))

    ct = ColumnTransformer(
        [('one_hot_encoder', OneHotEncoder(), [3])],    
        remainder='passthrough'                         
    )
    T = np.array(ct.fit_transform(T))


    sc3 = StandardScaler()
    T = sc3.fit_transform(T)

    y_pred1 = classifier.predict(np.array([T[-1],]))
    y_pred2 = classifier2.predict(np.array([T[-1],]))
    y_pred3 = classifier3.predict(np.array([T[-1],]))
    y_pred4 = classifier4.predict(np.array([T[-1],]))
    y_pred5 = classifier5.predict(np.array([T[-1],]))

    p1=y_pred1[0]
    a1=p1.tolist()
    p2=y_pred2[0]
    a2=p2.tolist()
    p3=y_pred3[0]
    a3=p3.tolist()
    p4=y_pred4[0]
    a4=p4.tolist()
    p5=y_pred5[0]
    a5=p5.tolist()
    l1=a1.index(max(a1))
    l2=a2.index(max(a2))
    l3=a3.index(max(a3))
    l4=a4.index(max(a4))
    l5=a5.index(max(a5))

    K.clear_session()

    return {'prefs':inp1,'univs':[y1[l1],y2[l2],y3[l3],y4[l4],y5[l5]]}