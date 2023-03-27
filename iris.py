import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import numpy as np

st.title("Iris Flower Prediction")

st.markdown("""
This app performs simple prediction of iris flowers, given the inputs 'Sepal Length', 'Sepal Width', 'Petal Length' and 'Petal Width'.
* **Python libraries:** pandas, streamlit, sklearn
""")

st.sidebar.header('Input Parameters')
iris = datasets.load_iris()


# # getting the min and max of each feature
# st.subheader('info')
# st.write(iris.data[:,0])

# st.subheader('Min and Max')
# minmax = {'sl_min': min(iris.data[:,0]), 'sl_max': max(iris.data[:,0]),
#            'sw_min': min(iris.data[:,1]), 'sw_max': max(iris.data[:, 1]), 
#            'pl_min': min(iris.data[:,2]), 'pl_max': max(iris.data[:, 2]), 
#            'pw_min': min(iris.data[:,3]), 'pw_max': max(iris.data[:, 3])}
# mmdf = pd.DataFrame(minmax, index=[0])
# st.write(mmdf)

def user_input_features():
    # Sidebar - Features
    sepal_len = st.sidebar.slider('Sepal Length', 4.3, 7.9)
    sepal_wid = st.sidebar.slider('Sepal Width', 2.0, 4.4)
    petal_len = st.sidebar.slider('Petal Length', 1.0, 6.9)
    petal_wid = st.sidebar.slider('Petal Width', 0.1, 2.5)
    features = {'sepal_length':sepal_len, 
                'sepal_width':sepal_wid, 
                'petal_length':petal_len, 
                'petal_width':petal_wid}
    df_features = pd.DataFrame(features, index=[0])
    return df_features

features_df = user_input_features()

st.subheader('User Input Parameters')
st.write(features_df)

X = iris.data
Y = iris.target

clf = RandomForestClassifier()
clf.fit(X, Y)

prediction = clf.predict(features_df)
prediction_proba = clf.predict_proba(features_df)

st.subheader('Class labels and index number')
st.write(iris.target_names)

st.subheader('Prediction')
st.write(iris.target_names[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)