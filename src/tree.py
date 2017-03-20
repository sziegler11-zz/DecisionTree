class DecisionNode:

	def __init__(self,col=-1,value=None,results=None,ln=None,rn=None):
		self.col = col
		self.value = value
		self.results = results
		self.ln = ln
		self.rn = rn
		


def divideset(X, y, column, value):
	"""
	Given the feature set, labels, column, and splitting value, return "left" and "right" feature set-label
	pairs 
	"""
    if isinstance(value, int) or isinstance(value, float):
        X_left, y_left = X.copy()[X[column] < value], y.copy()[X[column] < value]
        X_right, y_right = X.copy()[X[column] >= value], y.copy()[X[column] >= value]
    else:
        X_left, y_left = X.copy()[X[column] != value], y.copy()[X[column] != value]
        X_right, y_right = X.copy()[X[column] == value], y.copy()[X[column] == value]
    
    return (X_left,y_left,X_right,y_right)

 def gini(y):
 	"""
 	Computes the gini impurity of a list of labels.

 	Arguments:
 		y [pandas Series] - labels of samples (in the terminal node of a decision tree)

 	Returns:
 		g [float] - gini impurity 
 	"""
    counts = y.value_counts()
    normalized = counts.apply(lambda x: 1.0*x/np.sum(counts))
    return np.sum([p*(1-p) for p in normalized])   


def fit(X, y, score=gini):
    
    bestGain = 0
    bestCol = None
    bestValue = None
    bestSplit = None
    
    rootScore = score(y)
    
    # iterate through rows and columns, searching for the split with the largest gain
    for row in X.iterrows():
        for col in row[1].index:
            value = row[1][col]
            X_left,y_left,X_right,y_right = divideset(X,y,col,value)
            leftScore = score(y_left)
            rightScore = score(y_right)
            p_left = 1.0*len(y_left)/len(y)
            p_right = 1.0 - p_left
            gain = rootScore - p_left*leftScore - p_right*rightScore
            
            if gain > max(0,bestGain):
                bestGain = gain
                bestCol = col
                bestValue = value
                bestSplit = (X_left,y_left,X_right,y_right)
    
    # recursively call fit on the left and right subtrees
    if bestGain > 0:
        leftNode = fit(bestSplit[0],bestSplit[1],score=score)
        rightNode = fit(bestSplit[2],bestSplit[3],score=score)
        return DecisionNode(col=bestCol, value=bestValue, results=None, ln=leftNode, \
                           rn=rightNode)
    else:
        counts = y.value_counts()
        return DecisionNode(results=dict([(key,counts[key]) for key in counts.index]))

def classify(observation, tree):
	"""
	Given an observation, which contains all the requisite features
	"""
    if tree.results is not None:
        return max(tree.results.keys(), key=lambda x: tree.results[x])
    else:
        v = observation[tree.col]
        branch = None
        if isinstance (v,int) or  isinstance(v,float):
            if v >= tree.value:
                branch = tree.rn
            else:
                branch = tree.ln
        else:
            if v == tree.value:
                branch = tree.rn
            else:
                branch = tree.ln
        return classify(observation, branch)

def printtree(tree,indent=''):
    # Is this a leaf node?
    if tree.results!=None:
       print str(tree.results)
    else:
       # Print the criteria
       print str(tree.col)+':'+str(tree.value)+'? '
       # Print the branches
       print indent+'T->',
       printtree(tree.tb,indent+'  ')
       print indent+'F->',
       printtree(tree.fb,indent+'  ')



                
    
