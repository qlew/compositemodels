import statsmodels.api as sm
from censoredregression import CensoredRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

data = sm.datasets.fair.load()
X = data.exog
y = data.endog

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=23)

cens = CensoredRegression(RandomForestClassifier(
    random_state=12), RandomForestRegressor(random_state=12))

cens.fit(X_train, y_train)


cens.score(X_test, y_test)
cens.score(X_train, y_train)

cens2 = CensoredRegression(LogisticRegression(), RandomForestRegressor())

cens2.fit(X_train, y_train)
cens2.score(X_test, y_test)
cens2.score(X_train, y_train)
