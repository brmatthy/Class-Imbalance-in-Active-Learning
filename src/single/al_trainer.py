'''
Module containing a trainig function for any given classifier and query strategy.
'''


'''
Use a model and a query strategy to perform active learning for a number of cycles.
The model will be fitted during this function.
@param clf -- The model to fit with. This classifier will be altered.
@param qs -- The query strategy to pick the samples.
@param X -- The unlabled data. This list will not be altered.
@param y -- Lables already known to the learner. This list will not be altered.
@param y_true -- The correct lables for the data X. This is used as oracle.
This list will not be altered.
@param cycles (default: 10) -- Number of cycles to train the learner.
@param output_cycles (deflaut: None) -- List of iteration numbers. Any iteration that
is listed in this list will add the y of it's iteration to the output. If this 
value is None, all iterations will be included.

@returns Dict of y's.
Returns a dict of label lists. Any iteration contained in the output_cycle will add
it's lable list (y) to this dict.
'''
def al_single(clf, qs, X, y, y_true, cycles=10, output_cycles=None):
    # copy y
    lables = y[:]
    # fit the model
    clf.fit(X, lables)

    # update cycles to output
    if output_cycles == None:
        output_cycles = range(cycles)

    output = {}
    for i in range(cycles):
        # get the indices of the selected samples
        q_idx = qs.query(X=X, y=lables, clf=clf)
        # ask oracle to lable the samples
        lables[q_idx] = y_true[q_idx]
        # refit the model
        clf.fit(X,lables)

        # add to output
        if i in output_cycles:
            output[i] = lables

    return output

