import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import seaborn as sns
from sklearn.decomposition import TruncatedSVD
import streamlit as st

st.write(plt.rcParams["font.family"])
plt.rcParams["font.family"]="MS Gothic"
st.write(plt.rcParams["font.family"])
Path=st.sidebar.file_uploader('Excel')
if Path is not None:
    df=pd.read_excel(Path,index_col=0)
    N=st.sidebar.slider("主要素数",3,len(df.columns)-1,3,1)
    c_list=[]
    i_list=[]
    for i in range(N):
        c_list.append("comp"+str(i+1))
        i_list.append("w"+str(i+1))
    MODE=st.sidebar.radio("MODE",["2D","3D"])
    model_svd=TruncatedSVD(n_components=N)
    vecs_list=model_svd.fit_transform(df)
    st.dataframe(pd.DataFrame(vecs_list,columns=c_list,index=df.index))
    st.dataframe(pd.DataFrame(model_svd.components_,columns=df.columns,index=i_list))
    if MODE=="2D":
        select1=st.sidebar.selectbox("x軸",np.arange(1,N+1),0)
        select2=st.sidebar.selectbox("y軸",np.arange(1,N+1),1)
        fig,ax=plt.subplots()
        X=vecs_list[:,select1-1]
        Y=vecs_list[:,select2-1]
        plt.scatter(X,Y)
        for i,(annot_x,annot_y) in enumerate(zip(X,Y)):
            plt.annotate(df.index[i],((annot_x,annot_y)))
        plt.xlabel("第一主成分")
        plt.ylabel("第二主成分")
        st.pyplot(fig)
        fig,ax=plt.subplots()
        X_comp=model_svd.components_[select1-1]
        Y_comp=model_svd.components_[select2-1]
        plt.scatter(X_comp,Y_comp)
        for i,(annot_x,annot_y) in enumerate(zip(X_comp,Y_comp)):
            plt.annotate(df.columns[i],((annot_x,annot_y)))
        plt.xlabel("第一主成分の重み")
        plt.ylabel("第二主成分の重み")
        st.pyplot(fig)
        st.write(plt.rcParams["font.family"])
    if MODE=="3D":
        select1=st.sidebar.selectbox("x軸",np.arange(1,N+1),0)
        select2=st.sidebar.selectbox("y軸",np.arange(1,N+1),1)
        select3=st.sidebar.selectbox("z軸",np.arange(1,N+1),2)
        fig=plt.figure()
        ax=fig.add_subplot(projection="3d")
        X=vecs_list[:,select1-1]
        Y=vecs_list[:,select2-1]
        Z=vecs_list[:,select3-1]
        ax.scatter(X,Y,Z)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        st.pyplot(fig)
        fig=plt.figure()
        ax=fig.add_subplot(projection="3d")
        X_comp=model_svd.components_[select1-1]
        Y_comp=model_svd.components_[select2-1]
        Z_comp=model_svd.components_[select3-1]
        ax.scatter(X_comp,Y_comp,Z_comp)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        st.pyplot(fig)
    X=[]
    Y=[]
    fig,ax=plt.subplots()
    for i in range(df.shape[1]):
        model_svd=TruncatedSVD(n_components=i)
        vecs_list=model_svd.fit_transform(df)
        X.append(i)
        Y.append(100*sum(model_svd.explained_variance_ratio_))
        plt.annotate(100*round(sum(model_svd.explained_variance_ratio_),4),((i,100*sum(model_svd.explained_variance_ratio_))))
    X.append(df.shape[1])
    Y.append(100)
    plt.annotate(100,(df.shape[1],100))
    plt.plot(X,Y,"o-")
    plt.xlabel("主成分の数")
    plt.ylabel("情報量(%)")
    st.pyplot(fig)
