# coding: utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
import streamlit as st
plt.rcParams["font.family"]="MS Gothic"

Path=st.sidebar.file_uploader('Excel')
if Path is not None:
    df=pd.read_excel(Path,index_col=0) 
else:
    Path="score.xlsx"
    df=pd.read_excel(Path,index_col=0)
    
Processing=st.sidebar.radio("Processing",["Origin","Standardize(µ=0,σ=1)","Normalize(max=1,min=0)"])
N=st.sidebar.slider("主要素数",3,len(df.columns)-1,3,1)
select1=st.sidebar.selectbox("x axis",np.arange(1,N+1),0)
select2=st.sidebar.selectbox("y axis",np.arange(1,N+1),1)
x_label1=st.sidebar.text_input('x label',"comp"+str(select1))
y_label1=st.sidebar.text_input('y label',"comp"+str(select2))
x_label2=st.sidebar.text_input('x label',"w"+str(select1))
y_label2=st.sidebar.text_input('y label',"w"+str(select2))

st.write("Raw Data")
st.write(df)

if Processing=="Origin":
    df=df
else:
    if Processing=="Standardize(µ=0,σ=1)":
        df=df.apply(lambda x:(x-x.mean())/x.std(),axis=1)
    else:
        df=df.apply(lambda x:(x-x.min())/(x.max()-x.min()),axis=1)
    df=df.fillna(0)
    st.write("Processed Data")
    st.write(df)
    
c_list=[]
i_list=[]
for i in range(N):
    c_list.append("comp"+str(i+1))
    i_list.append("w"+str(i+1))
model_svd=TruncatedSVD(n_components=N)
vecs_list=model_svd.fit_transform(df)
st.dataframe(pd.DataFrame(vecs_list,columns=c_list,index=df.index))
st.dataframe(pd.DataFrame(model_svd.components_,columns=df.columns,index=i_list))

st.write(f"comp{select1} vs comp{select2}")
fig,ax=plt.subplots()
X=vecs_list[:,select1-1]
Y=vecs_list[:,select2-1]
plt.scatter(X,Y)
for i,(annot_x,annot_y) in enumerate(zip(X,Y)):
    plt.annotate(df.index[i],((annot_x,annot_y)))
plt.xlabel(x_label1)
plt.ylabel(y_label1)
st.pyplot(fig)

st.write(f"w{select1} vs w{select2}")
fig,ax=plt.subplots()
X_comp=model_svd.components_[select1-1]
Y_comp=model_svd.components_[select2-1]
plt.scatter(X_comp,Y_comp)
for i,(annot_x,annot_y) in enumerate(zip(X_comp,Y_comp)):
    plt.annotate(df.columns[i],((annot_x,annot_y)))
plt.xlabel(x_label2)
plt.ylabel(y_label2)
st.pyplot(fig)
        
