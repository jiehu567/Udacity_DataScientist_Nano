# Training and Tuning

## Underfitting vs. Overfitting

- Underfitting: bias, not do well in training set
- Overfitting: memorize, do well in training set but testing set, error due to high variance

## Detect Error

## Cross Balidation

### Model Complexity Graph

underfit -> right -> overfit

high bias error -> high variance

low degree -> high degree (more complex)

## Don't use testing data to train model

3 sets:

- Training
- Validation: make decision
- Testing

### K-Fold Validation

Goal: Not necessary lose training data for validation

- Break data into K buckets
- Train K times: choose one bucket as testing set and the remaining K-1 as traing data
- Avg the results to get final model

```{python}
from sklearn.model_selection import KFold
kf = KFold(12, 3， shuffle = True)

for train_indices, test_indices in kf:
    ......

```

Parameters:
- n_splits
- size of testing set
- shuffle = True, randomly

## Learning Curve

when # of points increases, training error and cross validation error converge

Overfitting model / more complex models tend to converge slower.

# Grid Search

- Train the model and get parameters
- Grid Search: Use validation to pick the hyperparameters with highest score
- use testing data to make sure if model is good

```{python}
from sklearn.model_selection import 
parameters = {
    'kernel': ['poly', 'rbf'],
    'C': [0.1, 1, 10]
}

# scorer
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
scorer = make_scoreer(f1_score)

# creat GridSearch obj
grid_obj = GridSearchCV(clf, parameters, scoring=scorer)
grid_fit = grid_obj.fit(X, y)

best_clf = grid_fit.best_estimator_
best_clf.predict(X_test)
```


### Full code:

```{python}
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

clf = DecisionTreeClassifier(random_state=42)

# TODO: Create the parameters list you wish to tune.
parameters = {
    'max_depth': list(range(1, 11)),
    'min_samples_leaf': list(range(3, 11)),
    'min_samples_split': list(range(3, 11))
}

# TODO: Make an fbeta_score scoring object.
scorer = make_scorer(f1_score)

# TODO: Perform grid search on the classifier using 'scorer' as the scoring method.
grid_obj = GridSearchCV(clf, parameters, scoring=scorer)

# TODO: Fit the grid search object to the training data and find the optimal parameters.
grid_fit = grid_obj.fit(X_train, y_train)

# TODO: Get the estimator.
best_clf = grid_fit.best_estimator

# Fit the new model.
best_clf.fit(X_train, y_train)

# Make predictions using the new model.
best_train_predictions = best_clf.predict(X_train)
best_test_predictions = best_clf.predict(X_test)

# Calculate the f1_score of the new model.
print('The training F1 Score is', f1_score(best_train_predictions, y_train))
print('The testing F1 Score is', f1_score(best_test_predictions, y_test))

# Plot the new model.
plot_model(X, y, best_clf)

# Let's also explore what parameters ended up being used in the new model.
best_clf

```

### Plot model Code

```{python}
def plot_model(X, y, clf):
    plt.scatter(X[np.argwhere(y==0).flatten(),0],X[np.argwhere(y==0).flatten(),1],s = 50, color = 'blue', edgecolor = 'k')
    plt.scatter(X[np.argwhere(y==1).flatten(),0],X[np.argwhere(y==1).flatten(),1],s = 50, color = 'red', edgecolor = 'k')

    plt.xlim(-2.05,2.05)
    plt.ylim(-2.05,2.05)
    plt.grid(False)
    plt.tick_params(
        axis='x',
        which='both',
        bottom='off',
        top='off')

    r = np.linspace(-2.1,2.1,300)
    s,t = np.meshgrid(r,r)
    s = np.reshape(s,(np.size(s),1))
    t = np.reshape(t,(np.size(t),1))
    h = np.concatenate((s,t),1)

    z = clf.predict(h)

    s = s.reshape((np.size(r),np.size(r)))
    t = t.reshape((np.size(r),np.size(r)))
    z = z.reshape((np.size(r),np.size(r)))

    plt.contourf(s,t,z,colors = ['blue','red'],alpha = 0.2,levels = range(-1,2))
    if len(np.unique(z)) > 1:
        plt.contour(s,t,z,colors = 'k', linewidths = 2)
    plt.show()

```