
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'notebook')
get_ipython().run_line_magic('matplotlib', 'notebook')


# In[4]:


#wczytanie i zlaczenie plikow
allFiles = ['dane.csv', 'dane2.csv', 'dane3.csv']
data = []
for f in allFiles:
    df = pd.read_csv(f, sep=';')
    data.append(df)
frame = pd.merge(data[0], data[1], on='Data')
frame = pd.merge(frame, data[2], on='Data')


# In[5]:


#zmiana nazw kolumn
columns = { 'Data' : 'date', 'poziom wody w piezometrze B1 npm [cm]': 'waterlv',
            'temperatura wody w piezometrze B1 [C]': 'watertemp',
            'poziom morza': 'sealv',
            'Opady' : 'precip', 'Temperatura powietrza [C]': 'temp',
            'Prędkość wiatru' : 'vwind', 'Kierunek wiatru' : 'dwind' }
frame = frame[list(columns.keys())].rename(columns=columns)


# In[6]:


frame[0:3]


# In[7]:


#wykres zmiany poziomu morza w czasie
frame.plot(x='date', y='sealv')


# In[8]:


#wykres poziom wody w piezometrze w stosunku do poziomu wody w morzu
frame.plot(x='waterlv', y='sealv', style='ro', alpha=0.1)


# In[31]:


#zamiana kolumny z datami na typ datetime
frame['date'] = pd.to_datetime(frame['date'])
print (frame['date'].dtype)
frame.set_index(frame["date"],inplace=True)
#suma poziomu wody w piezometrze dla miesiecy
df1 = frame['waterlv'].resample('M', how='sum')


# In[10]:


#zamiana wartosci kierunku wiatru w stopniach na symbole
def wiatr(angle):
    if angle <= 22.5:
        return 'N'
    elif angle > 22.5 and angle <= 67.5:
        return 'NE'
    elif angle > 67.5 and angle <= 112.5:
        return 'E'
    elif angle > 112.5 and angle <= 157.5:
        return 'SE'
    elif angle > 157.5 and angle <= 202.5:
        return 'S'
    elif angle > 202.5 and angle <= 247.5:
        return 'SW'
    elif angle > 247.5 and angle <= 292.5:
        return 'W'
    elif angle > 292.5 and angle <= 337.5:
        return 'NW'
    else:
        return 'N'
    
frame['wind'] = frame['dwind'].apply(wiatr)


# In[27]:


frame[0:10]


# In[13]:


#tworzenie kolumn z wartosciami opadow od 1 do 5 dni wstecz
def precip_before(date, days_count):
    key = date - timedelta(days=days_count)
    if key in frame.index:
        return frame.loc[key]['precip']
    else:
        return None
    
for i in range(1,6):
    frame['precip'+ str(i)] = frame['date'].apply(lambda x : precip_before(x, i))


# In[14]:


#tworzenie kolumny z suma opadow 5 dni wstecz
try:
    frame['precipsum']=frame.iloc[:,9:14].sum(axis=1)
except:
    None
        


# In[26]:


frame[0:10]


# In[19]:


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


# In[23]:


frame[0:10]


# In[29]:


# usuniecie wierszy z NaN
frame = frame.dropna()
frame[0:10]


# In[30]:


# zapis do pliku csv
frame.to_csv('result.csv')

