import numpy as np

def preprocess(filepath):
    data_X = []
    data_Y = []
    with open(filepath) as fp:
        for cnt, line in enumerate(fp):
            y, x = line.split("\t")[0], line.split("\t")[1]      
            data_X.append(x)
            data_Y.append(y)
            
        m = len(data_X)
        data_X = np.reshape(np.asarray(data_X),(m,1))
        data_Y = np.reshape(np.asarray(data_Y),(m,1))

        #Shuffle the data
        data = np.concatenate((data_X, data_Y), axis=1)
        np.random.shuffle(data)

        data_X = data[:,0]
        data_Y = data[:,1]
        
        return (data_X, data_Y, m)
    

def split_dataset(data_X, data_Y):
    
    #TO-DO: harcoded values should be removed
    train_X, train_Y = data_X[:4251], data_Y[:4251]
    dev_X, dev_Y = data_X[4251:5668], data_Y[4251:5668]
    test_X, test_Y = data_X[5668:], data_Y[5668:]
    
    return(train_X, train_Y, dev_X, dev_Y, test_X, test_Y)
    