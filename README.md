# Model-Selection
# GridSearchCM for model selection
# import library pandas
import pandas as pd

# Load Dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/haberman.csv'
list_cols = ['Age', "Patient's Years", "N_positive_ax", "survival_status"]
df2 = pd.read_csv(url, names=list_cols)
df2['survival_status'].value_counts()

# make variabels train and test
X = df2.drop('survival_status', axis = 1)
y = df2['survival_status']

X_train, X_test, y_train, y_test = train_test_split(Xs,y, test_size=0.25, random_state=42, stratify=y)

# import library KNN and GridSearchCv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

# tuning hyperparameter
model_knn = KNeighborsClassifier()
param_grid = {'n_neighbors' : np.arange(3,51), 'weights' : ['uniform','distance']}
gscv = GridSearchCV(model_knn, param_grid, scoring='roc_auc', cv = 10)
gscv.fit(X_train, y_train)

# best parameter
gscv.best_params_

# best score validation
gscv.best_score_

# make predict with probability in data test
y_predict = gscv.predict_proba(X_test)
y_predict

