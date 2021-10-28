import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Meiryo'
from sklearn.decomposition import TruncatedSVD
import streamlit as st

Path=st.sidebar.file_uploader('Excel')

if Path is not None:
    df=pd.read_excel(Path,index_col=0)
    N=st.sidebar.slider("主要素数",1,len(df.columns)-1,2,1)
    c_list=[]
    i_list=[]
    for i in range(N):
        c_list.append("comp"+str(i+1))
        i_list.append("w"+str(i+1))
    model_svd=TruncatedSVD(n_components=N)
    vecs_list=model_svd.fit_transform(df)
    st.dataframe(pd.DataFrame(vecs_list,columns=c_list,index=df.index))
    st.dataframe(pd.DataFrame(model_svd.components_,columns=df.columns,index=i_list))

    if N==2:
        fig,ax=plt.subplots()
        X=vecs_list[:,0]
        Y=vecs_list[:,1]
        plt.scatter(X,Y)
        for i,(annot_x,annot_y) in enumerate(zip(X,Y)):
            plt.annotate(df.index[i],((annot_x,annot_y)))
        plt.xlabel("第一主成分")
        plt.ylabel("第二主成分")
        st.pyplot(fig)

        fig,ax=plt.subplots()
        X_comp,Y_comp=model_svd.components_
        plt.scatter(X_comp,Y_comp)
        for i,(annot_x,annot_y) in enumerate(zip(X_comp,Y_comp)):
            plt.annotate(df.columns[i],((annot_x,annot_y)))
        plt.xlabel("第一主成分の重み")
        plt.ylabel("第二主成分の重み")
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
