# Train binary predictors for each target word using scikit-learn library (logistic regression).


import lib.utility as u
import config as c
import nltk
import numpy
import math
from sklearn.linear_model import LogisticRegression


if __name__ == "__main__":
    # Load feature space and target words
    features_index = u.read_features_file(c.output_dir + "feat-space.txt")
    feature_size = len(features_index)
    target_words = u.get_target_words()
    
    # Train predictor for each target word
    for target in target_words:
        print "Training for", target
        
        # Path to train data
        path = c.train_data_dir + target + ".txt"
        
        # Open train data file and build binary feature vectors vectors
        # Arrays for vectors and labels
        data = []
        labels = []
        with open(path) as file:
            for line in file:
                # create binary feature vector
                vect = numpy.zeros(feature_size)
                l = line.strip().split()
                for word in l[1:]:
                    # don't include target word in feature vector
                    if word != target:
                        try:
                            vect[features_index[word]] = 1
                        except:
                            #print "Not found in features:", word
                            pass
                data.append(vect)
                
                # add label to array
                if l[0] == "+1":
                    labels.append(1)
                else:
                    labels.append(0)
        
        # Train predictor using scikit-learn logistic regression
        logreg = LogisticRegression(fit_intercept=True)
        logreg.fit(data, labels)
        
        # Get weights and intercept/bias from model
        # NOTE: if we set intercept to false, it will fall back and use 0.0 as the intercept value
        # we still write it to file to make further programs simpler and consistent
        w = logreg.coef_
        try:
            b = logreg.intercept_[0]
        except:
            b = 0.0
        
        # Write learnt weight vector to file
        with open(c.weight_vector_dir + target + "-wv.txt", "w") as file:
            # write word
            file.write(target + " ")
            # then the intercept/bias
            file.write(str(b) + " ")
            # then the weights
            for i in w[0]:
                file.write(str(i) + " ")
            file.write("\n")
        
        #print "Done"
        print ""
    