#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings("ignore")


# In[2]:


data = pd.read_csv('Mall_Customers.csv')


# In[3]:


print('There are {} rows and {} columns in our dataset'.format(data.shape[0],data.shape[1]))


# In[4]:


data.head()


# In[5]:


data.info()


# In[6]:


data.describe()


# In[7]:


data.isnull().sum()


# In[8]:


males_age = data[data['Gender']=='Male']['Age']
females_age = data[data['Gender']=='Female']['Age']
age_bins = range(15,75,5)

#male histogram
fig2,(ax1,ax2) = plt.subplots(1,2,figsize=(12,5),sharey=True)
sns.distplot(males_age,bins=age_bins,kde=False,color='#0066ff', ax=ax1,hist_kws=dict(edgecolor="k",linewidth=2))
ax1.set_xticks(age_bins)
ax1.set_ylim(top=25)
ax1.set_title('Males')
ax1.set_ylabel('count')
ax1.text(45,23,"TOTAL count: {}".format(males_age.count()))
ax1.text(45,22,"Mean age: {:.1f}".format(males_age.mean()))
#female histogram
sns.distplot(females_age,bins=age_bins,kde=False,color='#cc66ff', ax=ax2,hist_kws=dict(edgecolor="k",linewidth=2))
ax2.set_xticks(age_bins)
ax2.set_ylim(top=25)
ax2.set_title('Females')
ax2.set_ylabel('count')
ax2.text(45,23,"TOTAL count: {}".format(females_age.count()))
ax2.text(45,22,"Mean age: {:.1f}".format(females_age.mean()))
plt.show()


# In[9]:


print('Kolgomoov-Smirnov test p-value: {:.2f}'.format(stats.ks_2samp(males_age,females_age)[1]))


# In[10]:


def labeler(pct,allvals):
    absolute=int(pct/100.*np.sum(allvals))
    return "{:.1f}%\n({:d})".format(pct,absolute)

sizes=[males_age.count(),females_age.count()]

fig0,ax1=plt.subplots(figsize=(6,6))
wedges, texts, autotexts = ax1.pie(sizes,autopct=lambda pct: labeler(pct,sizes),radius=1,colors=['#0066ff','#cc66ff'],startangle=90,textprops=dict(color="w"),wedgeprops=dict(width=0.7, edgecolor='w'))
                                                   
ax1.legend(wedges,['male','female'],
            loc='center right',
            bbox_to_anchor=(0.7,0,0.5,1))
                                                   
plt.text(0,0,'TOTAl\n{}'.format(data['Age'].count()),
         weight='bold',size=12,color='#52527a',
         ha='center',va='center')
                                                   
plt.setp(autotexts,size=12,weight='bold')
ax1.axis('equal')
plt.show


# In[11]:


males_income=data[data['Gender']=='Male']['Annual Income (k$)'] #subset with males incomed
females_income=data[data['Gender']=='Female']['Annual Income (k$)'] #subset with females income

my_bins = range(10,150,10)

#males histogram
fig, (ax1,ax2,ax3)=plt.subplots(1,3,figsize=(18,5))
sns.distplot(males_income,bins=my_bins,kde=False,color='#0066ff',ax=ax1,hist_kws=dict(edgecolor="k",linewidth=2))
ax1.set_xticks(my_bins)
ax1.set_yticks(range(0,24,2))
ax1.set_ylim(0,22)
ax1.set_title('Males')
ax1.set_ylabel('Count')
ax1.text(85,19,"Mean income:{:.1f}k$".format(males_income.mean()))
ax1.text(85,18,"Median income:{:.1f}k$".format(males_income.median()))
ax1.text(85,17,"Std. deviation:{:.1f}k$".format(males_income.std()))


#females histogram
sns.distplot(females_income,bins=my_bins,kde=False,color='#cc66ff',ax=ax2,hist_kws=dict(edgecolor="k",linewidth=2))
ax2.set_xticks(my_bins)
ax2.set_yticks(range(0,24,2))
ax2.set_ylim(0,22)
ax2.set_title('Females')
ax2.set_ylabel('Count')
ax2.text(85,19,"Mean income:{:.1f}k$".format(females_income.mean()))
ax2.text(85,18,"Median income:{:.1f}k$".format(females_income.median()))
ax2.text(85,17,"Std. deviation:{:.1f}k$".format(females_income.std()))

#boxplot
sns.boxplot(x='Gender',y='Annual Income (k$)', data=data, ax=ax3)
ax3.set_title('Boxplot of annual income')
plt.show()


# In[12]:


medians_by_age_group = data.groupby(["Gender",pd.cut(data['Age'],age_bins)]).median()
medians_by_age_group.index=medians_by_age_group.index.set_names(['Gender','Age_group'])
medians_by_age_group.reset_index(inplace=True)


# In[13]:


fig,ax=plt.subplots(figsize=(12,5))
sns.barplot(x='Age_group',y='Annual Income (k$)',hue='Gender',data=medians_by_age_group,
            palette=['#cc66ff','#0066ff'],
            alpha=0.7,edgecolor='k',
            ax=ax)
ax.set_title('Median annual income of male and female customers')
ax.set_xlabel('Age group')
plt.show()


# In[14]:


#correlations


# In[15]:


from scipy.stats import pearsonr
#cculating pearson's correlation
corr,_=pearsonr(data['Age'],data['Spending Score (1-100)'])
jp=(sns.jointplot('Age','Spending Score (1-100)',data=data,
                  kind='reg')).plot_joint(sns.kdeplot,zorder=0,n_levels=6)
plt.text(0,120,'Pearson: {:.2f}'.format(corr))
plt.show()


# In[16]:


corr1,_=pearsonr(males_age.values,males_income.values)
corr2,_=pearsonr(females_age.values,females_income.values)

sns.lmplot('Age','Annual Income (k$)',data=data,hue='Gender',aspect=1.5)
plt.text(15,87,'Pearson: {:.2f}'.format(corr1),color='blue')
plt.text(65,80,'Pearson: {:.2f}'.format(corr2),color='orange')

plt.show()


# In[17]:


from sklearn.cluster import KMeans


# In[18]:


X_numerics=data[['Age','Annual Income (k$)','Spending Score (1-100)']]


# In[19]:


from sklearn.metrics import silhouette_score
n_clusters=[2,3,4,5,6,7,8,9,10]
clusters_inertia=[]
s_scores=[]

for n in n_clusters:
    KM_est=KMeans(n_clusters=n,init='k-means++').fit(X_numerics)
    clusters_inertia.append(KM_est.inertia_)
    silhouette_avg=silhouette_score(X_numerics,KM_est.labels_)
    s_scores.append(silhouette_avg)


# In[20]:


fig,ax=plt.subplots(figsize=(12,5))
ax=sns.lineplot(n_clusters,clusters_inertia,marker='o',ax=ax)
ax.set_title("Elbow method")
ax.set_xlabel("number of clusters")
ax.set_ylabel("clusters inertia")
ax.axvline(5,ls='--',c="red")
ax.axvline(6,ls='--',c="red")
plt.grid()
plt.show()


# In[21]:


fig,ax=plt.subplots(figsize=(12,5))
ax=sns.lineplot(n_clusters,s_scores,marker='o',ax=ax)
ax.set_title("silhouette score method")
ax.set_xlabel("number of clusters")
ax.set_ylabel("silhouette score")
ax.axvline(6,ls='--',c="red")
plt.grid()
plt.show()


# In[22]:


KM_5_clusters=KMeans(n_clusters=5,init='k-means++').fit(X_numerics)
KM5_clustered=X_numerics.copy()
KM5_clustered.loc[:,'Cluster']=KM_5_clusters.labels_


# In[23]:


fig1,(axes)=plt.subplots(1,2,figsize=(12,5))

scat_1=sns.scatterplot('Annual Income (k$)','Spending Score (1-100)',data=KM5_clustered,
                hue='Cluster',ax=axes[0],palette='Set1',legend='full')
sns.scatterplot('Age','Spending Score (1-100)',data=KM5_clustered,
                hue='Cluster',palette='Set1',ax=axes[1],legend='full')
axes[0].scatter(KM_5_clusters.cluster_centers_[:,1],KM_5_clusters.cluster_centers_[:,2],marker='s',s=40,c="blue")
axes[1].scatter(KM_5_clusters.cluster_centers_[:,0],KM_5_clusters.cluster_centers_[:,2],marker='s',s=40,c="blue")
plt.show()


# In[24]:


KM5_clust_sizes=KM5_clustered.groupby('Cluster').size().to_frame()
KM5_clust_sizes.columns=["KM5_size"]
KM5_clust_sizes


# In[25]:


from mpl_toolkits.mplot3d import Axes3D

fig=plt.figure(figsize=(7,7))
ax=Axes3D(fig,rect=[0,0,.99,1],elev=20,azim=210)
ax.scatter(KM5_clustered['Age'],
           KM5_clustered['Annual Income (k$)'],
           KM5_clustered['Spending Score (1-100)'],
           c=KM5_clustered['Cluster'],s=35,edgecolor='k',cmap=plt.cm.Set1)

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel("Age")
ax.set_ylabel("Annual Income (k$)")
ax.set_zlabel("Spending Score (1-100")
ax.set_title("3D view of K-Means 5 clusters")
ax.dist=12
plt.show()


# In[26]:


KM_6_clusters=KMeans(n_clusters=6,init='k-means++').fit(X_numerics)
KM6_clustered=X_numerics.copy()
KM6_clustered.loc[:,'Cluster']=KM_6_clusters.labels_


# In[27]:


fig1,(axes)=plt.subplots(1,2,figsize=(12,5))

scat_1=sns.scatterplot('Annual Income (k$)','Spending Score (1-100)',data=KM6_clustered,
                hue='Cluster',ax=axes[0],palette='Set1',legend='full')
sns.scatterplot('Age','Spending Score (1-100)',data=KM6_clustered,
                hue='Cluster',palette='Set1',ax=axes[1],legend='full')
axes[0].scatter(KM_6_clusters.cluster_centers_[:,1],KM_6_clusters.cluster_centers_[:,2],marker='s',s=40,c="blue")
axes[1].scatter(KM_6_clusters.cluster_centers_[:,0],KM_6_clusters.cluster_centers_[:,2],marker='s',s=40,c="blue")
plt.show()


# In[28]:


KM6_clust_sizes=KM6_clustered.groupby('Cluster').size().to_frame()
KM6_clust_sizes.columns=["KM6_size"]
KM6_clust_sizes


# In[29]:


from mpl_toolkits.mplot3d import Axes3D

fig=plt.figure(figsize=(7,7))
ax=Axes3D(fig,rect=[0,0,.99,1],elev=20,azim=210)
ax.scatter(KM6_clustered['Age'],
           KM6_clustered['Annual Income (k$)'],
           KM6_clustered['Spending Score (1-100)'],
           c=KM6_clustered['Cluster'],s=35,edgecolor='k',cmap=plt.cm.Set1)

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel("Age")
ax.set_ylabel("Annual Income (k$)")
ax.set_zlabel("Spending Score (1-100")
ax.set_title("3D view of K-Means 6 clusters")
ax.dist=12
plt.show()


# In[30]:


#DBSCAN


# In[31]:


from sklearn.cluster import DBSCAN


# In[32]:


from itertools import product
eps_values=np.arange(8,12.75,0.25)
min_samples=np.arange(3,10)
DBSCAN_params=list(product(eps_values,min_samples))


# In[33]:


no_of_clusters=[]
sil_score=[]
for p in DBSCAN_params:
    DBS_clustering=DBSCAN(eps=p[0],min_samples=p[1]).fit(X_numerics)
    no_of_clusters.append(len(np.unique(DBS_clustering.labels_)))
    sil_score.append(silhouette_score(X_numerics,DBS_clustering.labels_))


# In[34]:


tmp=pd.DataFrame.from_records(DBSCAN_params,columns=['Eps','Min_samples'])
tmp['No_of_clusters']=no_of_clusters
pivot_1=pd.pivot_table(tmp,values='No_of_clusters',index='Min_samples',columns='Eps')

fig,ax=plt.subplots(figsize=(18,6))
sns.heatmap(pivot_1,annot=True,annot_kws={"size":16},cmap="YlGnBu",ax=ax)
ax.set_title('Number of clusters')
plt.show()


# In[35]:


tmp=pd.DataFrame.from_records(DBSCAN_params,columns=['Eps','Min_samples'])
tmp['sil_score']=sil_score
pivot_1=pd.pivot_table(tmp,values='sil_score',index='Min_samples',columns='Eps')

fig,ax=plt.subplots(figsize=(18,6))
sns.heatmap(pivot_1,annot=True,annot_kws={"size":10},cmap="YlGnBu",ax=ax)
plt.show()


# In[36]:


DBS_clustering=DBSCAN(eps=12.5,min_samples=4).fit(X_numerics)

DBSCAN_clustered=X_numerics.copy()
DBSCAN_clustered.loc[:,'Cluster']=DBS_clustering.labels_


# In[37]:


DBSCAN_clust_sizes=DBSCAN_clustered.groupby('Cluster').size().to_frame()
DBSCAN_clust_sizes.columns=["DBSCAN_size"]
DBSCAN_clust_sizes


# In[38]:


outliers=DBSCAN_clustered[DBSCAN_clustered['Cluster']==-1]
fig2,(axes)=plt.subplots(1,2,figsize=(12,5))

sns.scatterplot('Annual Income (k$)','Spending Score (1-100)',
                data=DBSCAN_clustered[DBSCAN_clustered['Cluster']!=-1],
                hue='Cluster',ax=axes[0],palette='Set1',legend='full',s=45)

sns.scatterplot('Age','Spending Score (1-100)',
                data=DBSCAN_clustered[DBSCAN_clustered['Cluster']!=-1],
                hue='Cluster',ax=axes[1],palette='Set1',legend='full',s=45)

axes[0].scatter(outliers['Annual Income (k$)'],outliers['Spending Score (1-100)'],s=5,label='outliers',c="k")
axes[1].scatter(outliers['Age'],outliers['Spending Score (1-100)'],s=5,label='outliers',c="k")
axes[0].legend()
axes[1].legend()

plt.setp(axes[0].get_legend().get_texts(),fontsize='10')
plt.setp(axes[1].get_legend().get_texts(),fontsize='10')

plt.show()


# In[39]:


#meanshift


# In[40]:


from sklearn.cluster import MeanShift,estimate_bandwidth

bandwidth=estimate_bandwidth(X_numerics,quantile=0.1)
ms=MeanShift(bandwidth).fit(X_numerics)

X_numerics['Labels']=ms.labels_
plt.figure(figsize=(12,8))
sns.scatterplot(X_numerics['Annual Income (k$)'],X_numerics['Spending Score (1-100)'],hue=X_numerics['Labels'],
                palette=sns.color_palette('hls',np.unique(ms.labels_).shape[0]))
plt.plot()
plt.title('MeanShift')
plt.show()


# In[41]:


MS_clustered=X_numerics.copy()
MS_clustered.loc[:,'Cluster']=ms.labels_


# In[42]:


MS_clust_sizes=MS_clustered.groupby('Cluster').size().to_frame()
MS_clust_sizes.columns=["MS_size"]
MS_clust_sizes


# In[43]:


#Agglomerative Clustering


# In[44]:


from sklearn.cluster import AgglomerativeClustering

agglom=AgglomerativeClustering(n_clusters=5,linkage='average').fit(X_numerics)

X_numerics['Labels']=agglom.labels_
plt.figure(figsize=(12,8))
sns.scatterplot(X_numerics['Annual Income (k$)'],X_numerics['Spending Score (1-100)'],hue=X_numerics['Labels'],
                palette=sns.color_palette('hls',5))
plt.title("Agglomerative with 5 Clusters")
plt.show()


# In[45]:


from scipy.cluster import hierarchy
from scipy.spatial import distance_matrix

dist=distance_matrix(X_numerics,X_numerics)
print(dist)


# In[46]:


Z=hierarchy.linkage(dist,'complete')
plt.figure(figsize=(18,50))
dendro=hierarchy.dendrogram(Z,leaf_rotation=0,leaf_font_size=12,orientation='right')


# In[47]:


Z=hierarchy.linkage(dist,'average')
plt.figure(figsize=(18,50))
dendro=hierarchy.dendrogram(Z,leaf_rotation=0,leaf_font_size=12,orientation='right')


# In[48]:


Agg_clustered=X_numerics.copy()
Agg_clustered.loc[:,'Cluster']=agglom.labels_


# In[49]:


Agg_clust_sizes=Agg_clustered.groupby('Cluster').size().to_frame()
Agg_clust_sizes.columns=["Agg_size"]
Agg_clust_sizes


# In[50]:


clusters=pd.concat([KM5_clust_sizes,KM6_clust_sizes,DBSCAN_clust_sizes,MS_clust_sizes,Agg_clust_sizes],axis=1,sort=False,copy=True)
clusters


# In[ ]:




