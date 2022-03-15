import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

data = pd.read_csv('../../data/Heart Disease Dataset.csv')

# one-hot coding
data = pd.get_dummies(data, columns=['sex'])

y_data = data['target']
x_data = data.drop(columns=['target'])


# standardization
standardizer = StandardScaler()
standardizer.fit(x_data)
x_data = standardizer.transform(x_data)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3)

classifier = LogisticRegression()
classifier.fit(x_train, y_train)

report = classification_report(y_test, classifier.predict(x_test))
print(report)
