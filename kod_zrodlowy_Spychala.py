
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import scipy
import math
import seaborn as sns
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import warnings; warnings.simplefilter('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#wczytanie i zlaczenie plikow
allFiles = ['dane.csv', 'dane2.csv', 'dane3.csv']
data = []
for f in allFiles:
    df = pd.read_csv(f, sep=';')
    data.append(df)
frame = pd.merge(data[0], data[1], on='Data')
frame = pd.merge(frame, data[2], on='Data')


# In[3]:


frame[0:3]


# In[4]:


#zmiana nazw kolumn
columns = { 'Data' : 'date', 'poziom wody w piezometrze B1 npm [cm]': 'waterlv',
            'temperatura wody w piezometrze B1 [C]': 'watertemp',
            'poziom morza': 'sealv',
            'Opady' : 'precip', 'Temperatura powietrza [C]': 'temp',
            'Prędkość wiatru' : 'vwind', 'Kierunek wiatru' : 'dwind' }
frame = frame[list(columns.keys())].rename(columns=columns)


# In[5]:


frame[0:3]


# In[6]:


frame.shape


# In[7]:


#zamiana kolumny z datami na typ datetime
frame['date'] = pd.to_datetime(frame['date'])
print (frame['date'].dtype)
frame.set_index(frame["date"],inplace=True)
#suma poziomu wody w piezometrze dla miesiecy
df1 = frame['waterlv'].resample('M', how='sum')


# In[8]:


#zamiana na radiany
frame['wind'] = frame['dwind'].apply(math.radians)
#sin
frame['sin'] = frame['wind'].apply(math.sin)
#cos
frame['cos'] = frame['wind'].apply(math.cos)


# In[9]:


frame[0:3]


# In[10]:


#tworzenie kolumn z wartosciami opadow od 1 do 5 dni wstecz
def precip_before(date, days_count):
    key = date - timedelta(days=days_count)
    if key in frame.index:
        return frame.loc[key]['precip']
    else:
        return None
    
for i in range(1,6):
    frame['precip'+ str(i)] = frame['date'].apply(lambda x : precip_before(x, i))


# In[11]:


frame[10:13]


# In[12]:


#tworzenie kolumny z suma opadow 5 dni wstecz
try:
    frame['precipsum']=frame.iloc[:,11:16].sum(axis=1)
except:
    None
        


# In[13]:


frame[0:3]


# In[14]:


#tworzenie kolumny ze srednia temperatura dla 5 dni wstecz
def mean_temp5(date):
    tlist = []
    for i in range(1,6):
        key = date - timedelta(days=i)
        if key in frame.index:
            tlist.append(frame.loc[key]['temp'])
        else:
            return None
    return np.mean(tlist)

frame['meantemp'] = frame['date'].apply(mean_temp5)


# In[15]:


frame[0:3]


# In[16]:


# usuniecie wierszy z NaN
frame = frame.dropna()
frame[0:3]


# In[17]:


# zapis do pliku csv
frame.to_csv('result.csv')


# **Eksploracyjna analiza danych**

# In[18]:


#shape
frame.shape
type(frame)


# In[19]:


#data types
print(frame.dtypes)


# In[20]:


#head
frame.head(5)


# In[21]:


#descriptions
print(frame.describe())


# In[22]:


#usuniecie kolumn ze zbioru
df = frame[frame.columns.difference(['date', 'wind', 'dwind'])]
framey = df[df.columns.difference(['waterlv'])]
df[0:3]


# In[23]:


#correlation
corr = frame.corr(method = 'pearson')
print(corr)


# In[24]:


#macierz korelacji
from yellowbrick.features import Rank2D

visualizer = Rank2D(algorithm="pearson")
visualizer.fit_transform(df)
visualizer.poof()


# In[25]:


import matplotlib.pyplot as plt
#histograms
plt.rcParams['figure.figsize'] = [10, 8]
df.hist(sharex = False, sharey = False, xlabelsize = 1, ylabelsize = 1)
plt.show()


# In[26]:


import seaborn as sns
sns.set(style="ticks")

#sns.pairplot(data=frame, kind="reg")


# In[27]:


#skewness
skew = frame.skew()
print(skew)


# In[28]:


# funkcje wykonujaca transfromacje danych
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import QuantileTransformer


def getInverse(transformer, columns, x):
    inv = transformer.inverse_transform(x)
    inv = pd.DataFrame(inv)
    inv.columns = columns
    return inv

def transform(tr, x):
    tr = tr.fit(x)
    transformed = tr.transform(x)
    transformed = pd.DataFrame(transformed)
    transformed.columns = x.columns
    inverse = lambda inv: getInverse(tr, x.columns, inv)
    return transformed, inverse

def transformY(invTrans, y):
    _, cols = df.shape
    rows = y.shape[0]
    temp = np.zeros((rows, cols))
    temp[:, df.columns.get_loc("waterlv")] = y
    return np.array(invTrans(temp)['waterlv'])


pt, invPt = transform(PowerTransformer(), df)
qt, invQt = transform(QuantileTransformer(), df)


# In[29]:


pt[0:3]


# In[30]:


qt[0:3]


# In[31]:


df[0:3]


# In[32]:


fig, (ax1,ax2, ax3) = plt.subplots(1,3, figsize=(15,4))
df.plot(y='waterlv', ax=ax1)
pt.plot(y='waterlv', ax=ax2)
qt.plot(y='waterlv', ax=ax3)


# In[33]:


pcorr = pt.corr(method = 'pearson')
print(pcorr)


# In[34]:


waterlv_index = df.columns.get_loc("waterlv")
waterlv_index


# In[35]:


#split-out validation dataset
from sklearn.model_selection import train_test_split
array = df.values
X = array[:, (0,1,2,3,4,5,6,7,8,9,10,11,12,14)].astype(np.float)
Y = array[:, waterlv_index].astype(np.float)
validation_size = 0.20
seed = 7 
X_train, X_validation, Y_train, Y_validation = train_test_split(X,Y, test_size = validation_size, random_state = seed)


# In[36]:


#split-out after YeoJohnson transformation
tarray = pt.values
Xt = tarray[:, (0,1,2,3,4,5,6,7,8,9,10,11,12,14)].astype(np.float)
Yt = tarray[:, waterlv_index].astype(np.float)
validation_size = 0.20
seed = 7 
Xt_train, Xt_validation, Yt_train, Yt_validation = train_test_split(Xt,Yt, test_size = validation_size, random_state = seed)


# In[37]:


#split-out after Quantile Transformer
qarray = qt.values
Xq = qarray[:, (0,1,2,3,4,5,6,7,8,9,10,11,12,14)].astype(np.float)
Yq = qarray[:, waterlv_index].astype(np.float)
validation_size = 0.20
seed = 7 
Xq_train, Xq_validation, Yq_train, Yq_validation = train_test_split(Xq,Yq, test_size = validation_size, random_state = seed)


# In[38]:


def evaluation(clf, X_train, Y_train, X_test, Y_test, invTr = None):
    if invTr is None:
        tr = lambda y: y
    else:
        tr = lambda y: transformY(invTr, y)
    train = tr(Y_train)
    train_pred = tr(clf.predict(X_train))
    test = tr(Y_test)
    test_pred = tr(clf.predict(X_test))
    show_evaluation(train, train_pred, test, test_pred)

def show_evaluation(train, train_pred, test, test_pred):
    print ("RMSE training set:", np.sqrt(mean_squared_error(train, train_pred)))
    print ("RMSE testing set:", np.sqrt(mean_squared_error(test, test_pred)))
    print ("MAE training set:", mean_absolute_error(train, train_pred))
    print ("MAE testing set:", mean_absolute_error(test, test_pred))
    
    fig, ax = plt.subplots()
    trainplot = sns.scatterplot(train,train_pred, label='train')
    testplot = sns.scatterplot(test,test_pred, label='test')
    plt.xlabel('waterlv')
    plt.ylabel('Predicted Value')
    plt.legend(loc="upper left")
    plt.show()
    


# In[39]:


def evaluateClf(clf, params):
    orig_clf = clf(**params)
    pt_clf = clf(**params)
    qt_clf = clf(**params)
    print(orig_clf)
    
    orig_clf.fit(X_train, Y_train)
    pt_clf.fit(Xt_train, Yt_train)
    qt_clf.fit(Xq_train, Yq_train)
    
    evaluation(orig_clf, X_train, Y_train, X_validation, Y_validation)
    evaluation(pt_clf, Xt_train, Yt_train, Xt_validation, Yt_validation, invPt)
    evaluation(qt_clf, Xq_train, Yq_train, Xq_validation, Yq_validation, invQt)


# **Random Forest**

# In[40]:


from sklearn.ensemble import RandomForestRegressor
evaluateClf(RandomForestRegressor, { 'n_estimators':200, 'max_depth': 6, 'random_state': 0})


# **Ridge**

# In[41]:


from sklearn import linear_model
evaluateClf(linear_model.Ridge, { 'alpha':.5})


# **Lasso**

# In[42]:


evaluateClf(linear_model.Lasso, { 'alpha':0.5})


# **Elastic Net**

# In[43]:


from sklearn.linear_model import ElasticNet
evaluateClf(ElasticNet, { 'alpha':0.1,'random_state':0})


# **SVM**

# In[44]:


from sklearn import svm
evaluateClf(svm.SVR, {'C':10})


# **Stochastic Gradient**

# In[45]:


from sklearn.linear_model import SGDClassifier
evaluateClf(linear_model.SGDRegressor, {'max_iter':1000, 'tol':1e-3})


# **Nearest Neighbors**

# In[46]:


from sklearn.neighbors import KNeighborsRegressor
neigh = KNeighborsRegressor(n_neighbors=5)
neigh.fit(X_train, Y_train) 
tneigh = KNeighborsRegressor(n_neighbors=5)
tneigh.fit(Xt_train, Yt_train) 
qneigh = KNeighborsRegressor(n_neighbors=5)
qneigh.fit(Xq_train, Yq_train)
evaluateClf(KNeighborsRegressor, {'n_neighbors':5})


# **Robust linear model estimation using RANSAC**
# 

# In[47]:


ransac = linear_model.RANSACRegressor()
ransac.fit(X_train, Y_train)
transac = linear_model.RANSACRegressor()
transac.fit(Xt_train, Yt_train) 
qransac = linear_model.RANSACRegressor()
qransac.fit(Xq_train, Yq_train)
evaluateClf(linear_model.RANSACRegressor, {})


# **XGBoost**

# In[48]:


import xgboost as xgb

evaluateClf(xgb.XGBRegressor, { 'max_depth':2, 'n_estimators':256, 'learning_rate':0.10, 'nthread': 4} )


# **Analiza głównych składowych - PCA**

# In[49]:


from sklearn.decomposition import PCA 
from sklearn.preprocessing import scale
import scipy.spatial.distance as dist
from sklearn.manifold import MDS


# In[50]:


pt_scaled = scale(pt)
qt_scaled = scale(qt)

pt_distance = dist.pdist(pt_scaled)
pt_distance = dist.squareform(pt_distance)
pt_punkty = MDS(n_components=2, dissimilarity='precomputed', random_state=1,eps=1e-5,n_init=10).fit_transform(pt_distance)

pca_pt = PCA(8,whiten=True).fit(pt_scaled)
pca_pt.explained_variance_ratio_
pt_components = pca_pt.transform(pt_scaled)


# In[51]:


pca_pt.explained_variance_ratio_


# In[52]:


qt_distance = dist.pdist(qt_scaled)
qt_distance = dist.squareform(qt_distance)
qt_punkty = MDS(n_components=2, dissimilarity='precomputed', random_state=1,eps=1e-5,n_init=10).fit_transform(qt_distance)

pca_qt = PCA(4,whiten=True).fit(qt_scaled)
pca_qt.explained_variance_ratio_
qt_components = pca_qt.transform(qt_scaled)


# In[53]:


pca_qt.explained_variance_ratio_


# In[54]:


df_pca = pd.DataFrame({'var':pca_pt.explained_variance_ratio_,
             'PC':['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8']})
print(df_pca)
sns.barplot(x='PC',y="var", data=df_pca)


# **Grupowanie**

# In[55]:


pt_scaled = scale(pt)
qt_scaled = scale(qt)


# In[56]:


pts_array = pt.values
Xpts = pts_array[:, (0,1,2,3,4,5,6,7,8,9,10,11,12,14)].astype(np.float)
Ypts = pts_array[:, waterlv_index].astype(np.float)


# In[57]:


qts_array = qt.values
Xqts = qts_array[:, (0,1,2,3,4,5,6,7,8,9,10,11,12,14)].astype(np.float)
Yqts = qts_array[:, waterlv_index].astype(np.float)


# In[58]:


from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer


# In[59]:


# Instantiate the clustering model and visualizer
model = KMeans()
visualizer = KElbowVisualizer(model, k=(2,12),  metric='calinski_harabaz', locate_elbow=True)

visualizer.fit(Xpts)    # Fit the data to the visualizer
visualizer.poof()    # Draw/show/poof the data


# In[60]:


from sklearn.cluster import MiniBatchKMeans
from yellowbrick.cluster import SilhouetteVisualizer


# In[61]:


model = KMeans(5, random_state=42)
visualizer = SilhouetteVisualizer(model, colors='yellowbrick', palette="Set3")

visualizer.fit(Xpts)        # Fit the data to the visualizer
visualizer.poof()


# In[62]:


kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(Xpts)

y_kmeans = kmeans.predict(Xpts)
y_kmeans


# In[63]:


pd.set_option('display.max_columns', 500)
df_clust = df.copy()
df_clust['klasa'] = y_kmeans
grpb = df_clust.groupby('klasa').agg(['mean','std','count'])
grpb


# In[64]:


plt.hist(df_clust.klasa)


# In[65]:


current_palette1 = sns.color_palette('Set3')
first = current_palette1[4]
second = current_palette1[6]
third = current_palette1[3]
fourth = current_palette1[9]
fifth = current_palette1[11]
current_palette = [first, second, third, fourth, fifth]
sns.set_palette(current_palette)

sns.palplot(current_palette)
type(current_palette)


# In[66]:


# import seaborn as sns

fig, ax = plt.subplots()
plt.rcParams['figure.figsize'] = [10,8]
sns.scatterplot(df_clust.index, df_clust.waterlv, hue=df_clust.klasa, 
                edgecolor="none", palette=current_palette, s=40, legend="full",)

ax.legend(frameon=True)

ax.set(xlim=('2008-03-01','2015-12-31'))
ax.grid(True)
ax.set_ylabel('waterlv')
ax.set_xlabel('date')

plt.show()


# In[67]:


fig, ax = plt.subplots()

sns.scatterplot(df_clust.index, df_clust.waterlv, hue=df_clust.klasa, 
                edgecolor="none", palette=current_palette, s = 50 )
ax.legend(frameon=True)
ax.set(xlim=('2010-01-01','2010-12-31'))
ax.grid(True)
ax.set_ylabel('waterlv')
ax.set_xlabel('date')
plt.xticks(
    rotation=30, 
    horizontalalignment='right', 
    fontsize='small'
)

plt.show()


# **Grid Search**
# **Random Forest**

# In[68]:


from sklearn.model_selection import GridSearchCV
tuned_parameters = {'n_estimators': [10, 50, 100, 200], 'max_depth': [5, 10, 20, 30], 'min_samples_split': [0.1,0.3,0.6],
                   'min_samples_leaf': [1,2,4]}

model = RandomForestRegressor(random_state=1)

grids= GridSearchCV(model ,tuned_parameters,cv=5,scoring='neg_mean_absolute_error',n_jobs=12)
grids.fit(X_train,Y_train)
grids.best_params_


# In[69]:


from sklearn.ensemble import RandomForestRegressor
evaluateClf(RandomForestRegressor, { 'n_estimators':10, 'max_depth': 10, 'random_state': 0,
                                   'min_samples_leaf': 1, 'min_samples_split':0.1})


# In[70]:


from sklearn.model_selection import GridSearchCV
tuned_parameters = {'n_estimators': [8, 10,12], 'max_depth': [8, 10, 12], 'min_samples_split': [0.05, 0.1,0.15]}

model = RandomForestRegressor(random_state=1)

grids= GridSearchCV(model ,tuned_parameters,cv=5,scoring='neg_mean_absolute_error',n_jobs=12)
grids.fit(X_train,Y_train)
grids.best_params_


# In[71]:


from sklearn.ensemble import RandomForestRegressor
evaluateClf(RandomForestRegressor, { 'n_estimators':10, 'max_depth': 12, 'random_state': 1, 'min_samples_split': 0.05, 'min_samples_leaf': 1})


# In[72]:


from sklearn.model_selection import GridSearchCV
tuned_parameters = {'n_estimators': [10, 50, 100, 200], 'max_depth': [5, 10, 20, 30]}

model = RandomForestRegressor(random_state=1)

grids= GridSearchCV(model ,tuned_parameters,cv=5,scoring='neg_mean_absolute_error',n_jobs=12)
grids.fit(X_train,Y_train)
grids.best_params_


# In[73]:


from sklearn.model_selection import GridSearchCV
tuned_parameters = {'n_estimators': [180, 200, 220], 'max_depth': [18, 20, 22]}

model = RandomForestRegressor(random_state=1)

grids= GridSearchCV(model ,tuned_parameters,cv=5,scoring='neg_mean_absolute_error',n_jobs=12)
grids.fit(X_train,Y_train)
grids.best_params_


# In[74]:


from sklearn.ensemble import RandomForestRegressor
evaluateClf(RandomForestRegressor, { 'n_estimators':200, 'max_depth': 20, 'random_state': 1})


# **Grid Search**
# **XGboost**

# In[75]:


tuned_parameters = {'n_estimators': [10, 100, 200, 300], 'max_depth': [3, 5, 10], 'learning_rate': [0.01, 0.05, 0.07, 0.1], 'objective':['reg:linear'], 'min_child_weight': [1, 2]}

model = xgb.XGBRegressor()

grids= GridSearchCV(model ,tuned_parameters,cv=5,scoring='neg_mean_absolute_error',n_jobs=12)
grids.fit(X_train,Y_train)
grids.best_params_


# In[76]:


evaluateClf(xgb.XGBRegressor, {'n_estimators': 300, 'max_depth': 10, 'learning_rate': 0.05, 'min_child_weight': 2,
                              'objective': 'reg:linear'}  )


# In[77]:


tuned_parameters = {'n_estimators': [275,300, 325], 'max_depth': [8, 10, 12], 'learning_rate': [0.04, 0.05, 0.06], 'objective':['reg:linear'], 'min_child_weight': [2, 3]}

model = xgb.XGBRegressor()

grids= GridSearchCV(model ,tuned_parameters,cv=5,scoring='neg_mean_absolute_error',n_jobs=12)
grids.fit(X_train,Y_train)
grids.best_params_


# In[78]:


evaluateClf(xgb.XGBRegressor, {'n_estimators': 325, 'max_depth': 10, 'learning_rate': 0.04, 'min_child_weight': 3,
                              'objective': 'reg:linear'}  )


# **Residuals**

# In[79]:


from yellowbrick.regressor import ResidualsPlot

# Instantiate the linear model and visualizer
modelRF = RandomForestRegressor(n_estimators=200, max_depth= 6, random_state=1)
visualizer = ResidualsPlot(modelRF)

visualizer.fit(X_train, Y_train)  # Fit the training data to the visualizer
visualizer.score(X_validation, Y_validation)  # Evaluate the model on the test data
visualizer.poof() 


# In[80]:


modelXGB = xgb.XGBRegressor(n_estimators =256, max_depth= 2, learning_rate= 0.1, nthread = 4)
visualizer = ResidualsPlot(modelXGB)

visualizer.fit(X_train, Y_train)  # Fit the training data to the visualizer
visualizer.score(X_validation, Y_validation)  # Evaluate the model on the test data
visualizer.poof()    


# **Wizualizacja wyników predykcji dla modeli lasów losowych i XGBoost**

# In[81]:


modelXGB.fit(X_train, Y_train)
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15,15))
xgb_pred = modelXGB.predict(X)
xgb_err = xgb_pred - df.waterlv
ax1.plot(xgb_pred, df.index, color = '#e8702a', label = 'predycja')
ax2.plot(xgb_err, df.index, color='#0ea7b5', label = 'błąd')
df.reset_index().plot(x='waterlv', y='date', color = '#0c457d', ax=ax1, label = 'wartość rzeczywista')
ax1.legend()
ax2.legend()
ax1.set_title('A', fontsize=20)
ax2.set_title('B', fontsize=20)
plt.show()


# In[82]:


modelRF.fit(X_train, Y_train)
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15,15))
rf_pred = modelRF.predict(X)
rf_err = rf_pred - df.waterlv
ax1.plot(rf_pred, df.index, color = '#e8702a', label = 'predycja')
ax2.plot(rf_err, df.index, color='#0ea7b5', label = 'błąd')
df.reset_index().plot(x='waterlv', y='date', color = '#0c457d', ax=ax1, label = 'wartość rzeczywista')
ax1.legend()
ax2.legend()
ax1.set_title('A', fontsize=20)
ax2.set_title('B', fontsize=20)
plt.show()


# In[83]:


colnames = list(df.columns) 
colnames.remove('waterlv')
colnames


# In[84]:


rf_importances = modelRF.feature_importances_


# In[85]:


imp = list(zip(rf_importances, colnames))
imp.sort()


# In[86]:


xgb_importances = modelXGB.feature_importances_
xgb_importances


# In[87]:


impx = list(zip(xgb_importances, colnames))
impx.sort()


# In[88]:


fig, (ax1, ax2) = plt.subplots(2,1, figsize=(7,10))
ax1.barh([i[1] for i in imp], [i[0] for i in imp])
ax2.barh([i[1] for i in impx], [i[0] for i in impx], color = 'r')
ax1.set_title('A', fontsize=20)
ax2.set_title('B', fontsize=20)
plt.show()


# In[89]:


clust_error = df_clust.copy()
rf_pred = modelRF.predict(X)
rf_err = rf_pred - df.waterlv
xgb_pred = modelXGB.predict(X)
xgb_err = xgb_pred - df.waterlv
clust_error['rf_err'] = rf_err
clust_error['rf_err_abs'] = abs(rf_err)
clust_error['xgb_err'] = xgb_err
clust_error['xgb_err_abs'] = abs(xgb_err)
grpb_err = clust_error.groupby('klasa').agg(['mean','std'])
grpb_err


# In[90]:


train = Y_train
train_pred = modelRF.predict(X_train)
test = Y_validation
test_pred = modelRF.predict(X_validation)

lbls = ['train', 'test']
plt.scatter(train,train_pred, c='C0', edgecolors='w', label='train')
plt.scatter(test,test_pred, c='C1', edgecolors='w', label='test')

z_train = np.polyfit(train, train_pred, 1)
p_train = np.poly1d(z_train)
print()

z_test = np.polyfit(test, test_pred, 1)
p_test = np.poly1d(z_test)
print()

plt.xlim([0, 150])
plt.ylim([0, 140])
x = np.linspace(0, 150)
plt.plot(x,p_train(x), "-", label = ("y = %.3fx + %.3f"%(z_train[0],z_train[1])))
plt.plot(x,p_test(x), label = ("y = %.3fx + %.3f"%(z_test[0],z_test[1])))
plt.legend()
plt.show()


# In[91]:


train = Y_train
train_pred = modelXGB.predict(X_train)
test = Y_validation
test_pred = modelXGB.predict(X_validation)

lbls = ['train', 'test']

z_train = np.polyfit(train, train_pred, 1)
p_train = np.poly1d(z_train)
print()

z_test = np.polyfit(test, test_pred, 1)
p_test = np.poly1d(z_test)
print()

plt.xlim([0, 150])
plt.ylim([0, 140])
x = np.linspace(0, 150)
plt.plot(x,p_train(x), "-", label = ("y = %.3fx + %.3f"%(z_train[0],z_train[1])))
plt.plot(x,p_test(x), label = ("y = %.3fx + %.3f"%(z_test[0],z_test[1])))
plt.scatter(train,train_pred, c='C0', edgecolors='w', label='train')
plt.scatter(test,test_pred, c='C1', edgecolors='w', label='test')

plt.legend( ncol=2)
plt.show()


# In[92]:


#wygładzanie z wykorzystaniem średniej ruchomej
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15,15))
rf_pred = modelRF.predict(X)
rf_err = rf_pred - df.waterlv
df_rpred = pd.DataFrame(rf_pred)

rolling_mean = df_rpred.rolling(window=100).mean()
rolling_mean2 = df_rpred.rolling(window=50).mean()

xgb_pred = modelXGB.predict(X)
xgb_err = xgb_pred - df.waterlv
df_xpred = pd.DataFrame(xgb_pred)

rolling_mean3 = df_xpred.rolling(window=100).mean()

ax1.plot(rolling_mean, df.index, color = '#42b883', label = 'predycja RF SMA 100')
ax2.plot(rolling_mean3, df.index, color = '#ff00ff', label = 'predycja XGB SMA 100')
ax1.plot(df.waterlv.rolling(window=100).mean(), df.index, color = '#e8702a', label = 'wartość rzeczywista SMA 100')
ax2.plot(df.waterlv.rolling(window=100).mean(), df.index, color = '#e8702a', label = 'wartość rzeczywista SMA 100')

df.reset_index().plot(x='waterlv', y='date', color = '#0c457d', ax=ax1, label = 'wartość rzeczywista')
df.reset_index().plot(x='waterlv', y='date', color = '#0c457d', ax=ax2, label = 'wartość rzeczywista')

ax1.legend()
ax2.legend()
ax1.set_title('A', fontsize=20)
ax2.set_title('B', fontsize=20)
plt.show()

