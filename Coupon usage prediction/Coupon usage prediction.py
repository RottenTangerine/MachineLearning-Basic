import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

data = pd.read_csv('../data/ccf_offline_stage1_train.csv')
data = data.dropna(how='any')
data['Discount_rate'] = data['Discount_rate'].apply(lambda x: 1 if ':' in x else 0)
data['target'] = pd.to_datetime(data['Date'], format='%Y%m%d') - pd.to_datetime(data['Date_received'], format='%Y%m%d')

y_data = data['target'].apply(lambda x: 1 if x <=  pd.Timedelta(15, 'D') else 0)
x_data = data.drop(columns=['target'])

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)
print(x_test.shape, x_train.shape)
model = DecisionTreeClassifier(max_depth=5)
model.fit(x_train, y_train)

print(classification_report(y_test, model.predict(x_test)))
