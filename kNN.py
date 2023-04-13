import os
import sys
import csv
import math
from collections import Counter

#   Establishes main variables needed before executing kNN() method
def main(training_path, test_path, k_neighbour):
    training_data = read_data(training_path)
    test_data = read_data(test_path)
    training_row_num = len(training_data)
    test_row_num = len(test_data)
    features_num = len(training_data[0])-1
    feature_ranges = calc_range(training_data, training_row_num, features_num)
    kNN(training_data, test_data, k_neighbour, training_row_num, test_row_num, features_num, feature_ranges)

# Reads data from the files provided via Command-Line Arguments
def read_data(filepath):
    list = []
    with open(filepath, newline='') as file_data:
        filereader=csv.reader(file_data, delimiter=' ')
        labels = next(filereader)
        for row in filereader:
            list.append(row)
    return list

#   Calculating Range for Min-Max Normalisation
def calc_range(training_data, rows, features):
    range_list = []
    for feat in range(features):
        max_instance = float(training_data[0][feat])
        min_instance = float(training_data[0][feat])
        for row in range(rows):
            if min_instance > float(training_data[row][feat]):
                min_instance = float(training_data[row][feat])
            if max_instance < float(training_data[row][feat]):
                max_instance = float(training_data[row][feat])
        range_list.append((max_instance-min_instance)**2)
    return range_list    


#   Executes the kNN method - calculating Euclidean dist, finding nearest neighbours and determine majority's class
#       then making a prediction and if prediction was correct correct_predictions counts increases, and remains if incorrect
#       This kNN method also prints each prediction and Accuracy to console as well as to sampleoutput.txt
def kNN(training_data, test_data, k_value, training_rows, test_rows, features_num, ranges):
    correct_predictions = 0
    text_output = []
    for i in range(test_rows):
        eucl_distances = calc_distance(training_data, test_data, training_rows, features_num, ranges, i) # returns list of eucl_distances      
        class_prediction = find_nearest_neighbours(k_value, eucl_distances, training_data, features_num)
        correct_predictions += make_prediction(test_data, features_num, i, class_prediction, text_output)
    with open (os.getcwd()+'/sampleoutput.txt', 'w') as file:
        file.write('K-value = '+str(k_value)+'\n')
        for i in range(len(text_output)):
            file.write(text_output[i]+'\n')
            print(text_output[i])
        file.write('Correct predictions: '+str(correct_predictions)+' out of '+str(test_rows)+'\n')
        file.write('Accuracy of predictions: '+str((correct_predictions/test_rows)*100)+'\n')
    print('Correct predictions: '+str(correct_predictions)+' out of '+str(test_rows))
    print('Accuracy of predictions: '+str((correct_predictions/test_rows)*100))

#   Calculates the Euclidean Distances for each instance to the Test instance
def calc_distance(trianing_data, test_data, rows, feat_num, ranges, current_test_row):
    distances = []
    for current_train_row in range(rows):
        dist = 0
        for current_feat in range(feat_num):
            dist += ((float(test_data[current_test_row][current_feat]) - float(trianing_data[current_train_row][current_feat]))**2)/ranges[current_feat]
        distances.append(math.sqrt(dist))
    return distances    

#   Goes through the distances and stores the classes of the nearest neighbours to test instance
def find_nearest_neighbours(k_value, eucl_distances, training_data, features_num):
    temp_dist = eucl_distances
    nearest_dist = []
    row_index = []
    predictions = []
    for k in range(k_value):
        min_dist = min(temp_dist)
        nearest_dist.append(min_dist)
        min_dist_index = temp_dist.index(min_dist)
        row_index.append(min_dist_index)
        predictions.append(training_data[min_dist_index][features_num])
        temp_dist[min_dist_index] = 100
    return predictions

#   Counts the majority of classes amongst nearest neighbours, and if matches returns 1 to add to total of correct predictions
#       Otherwise return zero for incorrect predictions
def make_prediction(test_data, feat_num, current_test_row, current_predictions, text_output):
    prediction = Counter(current_predictions).most_common()[0][0]
    if test_data[current_test_row][feat_num] == prediction:
        message = 'Class prediction for instance '+str(current_test_row+1)+' is: '+str(prediction)+'. Correct!'
        text_output.append(message)
        return 1
    else:
        message = 'Class prediction for instance '+str(current_test_row+1)+' is: '+str(prediction)+'. Incorrect, actual Class: '+str(test_data[current_test_row][feat_num])
        text_output.append(message)
        return 0
    
#   Triggered when kNN.py executed, checks if arguments are valid and passes them to main()
#   Checks if ther's the right number of args, if first two args (dataset files) are actually files, and if final args for k-value is an integer
if __name__ == '__main__':

    try:
        arg_training = sys.argv[1]
        arg_test = sys.argv[2]
        arg_k_value = int(sys.argv[3])
    except IndexError:
        print('Incorrect arguments, please refer to ReadMe.txt.')
        sys.exit(1)
    except ValueError:
        print('Last argument for k-neighbour must be an int. Please refer to ReadMe.txt')
        sys.exit(1)

    dir_path = os.getcwd()
    training_path = dir_path+'/'+arg_training
    test_path = dir_path+'/'+arg_test  

    if os.path.isfile(training_path) and os.path.isfile(test_path):
        main(training_path, test_path, arg_k_value)
    else:
        print('Argument for either training or test data is not a file. Please refer to the ReadMe.txt.')
