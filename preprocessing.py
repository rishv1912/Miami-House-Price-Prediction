import pandas as pd
from sklearn.model_selection import train_test_split


path = 'data/filtered_data.csv'

# at first we'll split it into only two sets  Train, Test 
# at second time we'll split it into three sets , Train,Valid and Test 

data = pd.read_csv(path)

data.head()

X = data.drop('SALE_PRC',axis=1)

y = data['SALE_PRC']



X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0,test_size=0.2)


print("X train",X_train)
print('X Test',X_test)
print('Y Train',y_train)
print('Y Test',y_test)