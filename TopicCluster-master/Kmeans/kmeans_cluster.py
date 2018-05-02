#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import importlib,sys
importlib.reload(sys)
#sys.setdefaultencoding('utf-8')
from sklearn.cluster import KMeans
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import ward, dendrogram
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from mpl_toolkits.mplot3d import Axes3D
from sklearn import metrics
import matplotlib as mpl
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import ward, dendrogram
'''vectorize the input documents'''
def tfidf_vector(corpus_path):
    corpus_train=[]
    #利用train-corpus提取特征
    target_train=[]
    for line in open(corpus_path, encoding='utf-8' ):
        line=line.strip().split('\t')
        if len(line)==2:
            words=line[1]
            category=line[0]
            target_train.append(category)
            corpus_train.append(words)
    print("build train-corpus done!!")
    count_v1= CountVectorizer(max_df=0.4,min_df=0.01)
    counts_train = count_v1.fit_transform(corpus_train)  
    
    word_dict={}
    for index,word in enumerate(count_v1.get_feature_names()):
        word_dict[index]=word
    
    print("the shape of train is "+repr(counts_train.shape) )
    tfidftransformer = TfidfTransformer()

    tfidf_train = tfidftransformer.fit(counts_train).transform(counts_train)
    weight = tfidf_train.toarray()

    dist = 1 - cosine_similarity(counts_train)
    linkage_matrix = ward(dist)
    fig, ax = plt.subplots(figsize=(15, 20))
    ax = dendrogram(linkage_matrix, orientation="right")
    plt.tick_params(
        axis='x',  # 使用 x 坐标轴
        which='both',  # 同时使用主刻度标签（major ticks）和次刻度标签（minor ticks）
        bottom='off',  # 取消底部边缘（bottom edge）标签
        top='off',  # 取消顶部边缘（top edge）标签
        labelbottom='off')
    plt.title('层次聚类结果',fontproperties ='SimHei',fontsize = 20)
    plt.tight_layout()  #  展示紧凑的绘图布局

    # 注释语句用来保存图片
    plt.savefig('ward_clusters.png', dpi=1000)  # 保存图片为 ward_clusters

    return tfidf_train,word_dict,weight

'''topic cluster'''
def cluster_kmeans(tfidf_train,word_dict,weight,cluster_docs,cluster_keywords,num_clusters):#K均值分类
    #svd = TruncatedSVD(n_components=2)
    f_docs=open(cluster_docs,'w+')
    km = KMeans(n_clusters=num_clusters)

    b=km.fit(tfidf_train)

   # svd.fit(tfidf_train)
    #b= 1.0*(X - X.mean())/X.std()
    #b=svd.explained_variance_ratio_
    #numSamples = len(b)
   # a = km.labels_
   # print(a, type(a))
   # mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
   # for i in range(numSamples):
        # markIndex = int(clusterAssment[i, 0])
    #    plt.plot(b[i][0], b[i][1], mark[km.labels_[i]])  # mark[markIndex])
   #  mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
   # a = clf.cluster_centers_
    #for i in range(k):
    #    plt.plot(a[i][0],a[i][1], mark[i], markersize=12)
        # print centroids[i, 0], centroids[i, 1]
   # plt.show()

    pca = PCA(n_components=3)  # 输出两维
    newData = pca.fit_transform(weight)  # 载入N维
   # s=pca.fit_transform(a)
    a=km.fit_predict(weight)
    print(a)
    print(newData[:,1])
    fig = plt.figure()
    fig.suptitle("三维散点图",fontproperties ='SimHei',fontsize = 20)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(newData[:,0], newData[:,1], newData[:,2],c=a)
    plt.show()
    print(metrics.calinski_harabaz_score(newData, a))
    clusters = km.labels_.tolist()
    cluster_dict={}
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]      
    doc=1
    for cluster in clusters:
        f_docs.write(str(str(doc))+','+str(cluster)+'\n')
        doc+=1
        if cluster not in cluster_dict:
            cluster_dict[cluster]=1
        else:
            cluster_dict[cluster]+=1
    print(f_docs)
    f_docs.close()
    cluster=1
    
    f_clusterwords = open(cluster_keywords,'w+')
    for ind in order_centroids: # 每个聚类选 50 个词
        words=[]
        for index in ind[:10]:
            words.append(word_dict[index])
        print( cluster,','.join(words))
        f_clusterwords.write(str(cluster)+'\t'+','.join(words)+'\n')
        cluster+=1
        print( '*****'*5)
    f_clusterwords.close()

  # select the best cluster num
def best_kmeans(tfidf_matrix,word_dict):
    import matplotlib.pyplot as plt 
    from matplotlib.font_manager import FontProperties 
    from sklearn.cluster import KMeans 
    from scipy.spatial.distance import cdist 
    import numpy as np
    K = range(1, 10) 
    meandistortions = [] 
    for k in K: 
        print (k,'****'*5)
        kmeans = KMeans(n_clusters=k) 
        kmeans.fit(tfidf_matrix)    
        meandistortions.append(sum(np.min(cdist(tfidf_matrix.toarray(), kmeans.cluster_centers_, 'euclidean'), axis=1)) / tfidf_matrix.shape[0]) 
    plt.plot(K, meandistortions, 'bx-')
    plt.grid(True) 
    #plt.xlabel('Number of clusters')
   # plt.ylabel('Average within-cluster sum of squares')
  #  plt.title('Elbow for Kmeans clustering')
    plt.show()

if __name__=='__main__':
    corpus_train = "./1.txt"
    cluster_docs = "./cluster_result_document.txt"
    cluster_keywords = "./cluster_result_keyword.txt"
    num_clusters = 2
    tfidf_train,word_dict,weight=tfidf_vector(corpus_train)
    best_kmeans(tfidf_train,word_dict)
    cluster_kmeans(tfidf_train,word_dict,weight,cluster_docs,cluster_keywords,num_clusters)
