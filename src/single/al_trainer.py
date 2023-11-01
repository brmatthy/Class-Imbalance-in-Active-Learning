'''
Module containing a trainig function for any given classifier and query strategy.
'''

'''
Use a model and a query strategy to perform active learning for a number of cycles.
The model will be fitted during this function.
@param clf -- The model to fit with. This classifier will not be altered.
@param qs -- The query strategy to pick the samples.
@param X -- The unlabled data. This list will not be altered.
@param y -- Lables already known to the learner. This list will not be altered.
@param y_true -- The correct lables for the data X. This is used as oracle.
This list will not be altered.
'''
def al_single_step(clf, qs, X, y, y_true):
    # get the index of the selected samples
    q_idx = qs.query(X=X, y=y, clf=clf)
    # ask oracle to lable the samples
    y[q_idx] = y_true[q_idx]
    # refit the model
    clf.fit(X,y)

