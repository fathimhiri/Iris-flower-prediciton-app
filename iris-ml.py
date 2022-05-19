import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier


st.write("""
# Simple Iris Flower Predicition App
This app predicts the **Iris Flower** type !
         """)
st.sidebar.header("User Input Parameter")

def user_input():
    sl=st.sidebar.slider('sepal length',4.3,7.9,5.0)
    sw=st.sidebar.slider('sepal width', 2.0,4.4,3.0)
    pl=st.sidebar.slider('Petal length', 1.0,6.9,1.7)
    pw=st.sidebar.slider('Petal width', 0.1,2.5,0.2)
   
    data={'sepal_length' :sl,
          'sepal_width' : sw,
          'petal_length': pl,
          'petal_width': pw}
    features = pd.DataFrame(data,index=['input values'])
    return features

df = user_input()

st.subheader('user input parameters')
st.write(df)

iris = datasets.load_iris()
x=iris.data
y=iris.target

clf = RandomForestClassifier()
clf.fit(x,y)


pred=clf.predict(df)
pred_proba = clf.predict_proba(df)

st.subheader('class labels and their corresponding index number')
st.write(iris.target_names)

st.subheader('prediciton')
st.write(iris.target_names[pred])


st.subheader('prediciiton probability')
st.write(pred_proba)











