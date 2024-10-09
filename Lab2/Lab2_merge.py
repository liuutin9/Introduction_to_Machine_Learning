import numpy as np
import pandas as pd
import math
import random
from numpy import sqrt
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
input_data = pd.read_csv('lab2_basic_input.csv')
input_data
max_depth = 2
depth = 0
min_samples_split = 2
n_features = input_data.shape[1] - 1
def entropy(data):
  """
  This function measures the amount of uncertainty in a probability distribution
  args:
  * data(type: DataFrame): the data you're calculating for the entropy
  return:
  * entropy_value(type: float): the data's entropy
  """
  p = 0 # to count the number of cases that survived
  n = 0 # to count the number of cases that passed away

  ### START CODE HERE ###
  # Hint 1: what is the equation for calculating entropy?
  # Hint 2: consider the case when p == 0 or n == 0, what should entropy be?
  resultList = list(data.loc[:, 'hospital_death'])
  p = sum(resultList)
  length = len(resultList)
  n = length - p
  if (n == 0 or p == 0):
    entropy_value = 0
  else:
    entropy_value = (-p / length) * np.log2(p / length) + (-n / length) * np.log2(n / length)
  ### END CODE HERE ###

  return entropy_value

# [Note] You have to save the value of "ans_entropy" into the output file
# Please round your answer to 4 decimal place
ans_entropy = entropy(input_data).round(4)
print("ans_entropy = ", ans_entropy)
def information_gain(data, mask):
  """
  This function will calculate the information gain
  args:
  * data(type: DataFrame): the data you're calculating for the information gain
  * mask(type: Series): partition information(left/right) of current input data,
    - boolean 1(True) represents split to left subtree
    - boolean 0(False) represents split to right subtree
  return:
  * ig(type: float): the information gain you can obtain by classifying the data with this given mask
  """
  ### START CODE HERE ###
  # Hint: you should use mask to split the data into two, then recall what is the equation for calculating information gain
  left = mask[mask == True].index
  right = mask[mask == False].index
  
  left_len = len(left)
  right_len = len(right)
  total_len = left_len + right_len
  
  left_subtree = data.loc[left]
  right_subtree = data.loc[right]
  
  left_entropy = entropy(left_subtree)
  right_entropy = entropy(right_subtree)
  
  before_entropy = entropy(data)
  after_entropy = left_len / total_len * left_entropy + right_len / total_len * right_entropy

  ig = before_entropy - after_entropy
  ### END CODE HERE ###

  return ig

# [Note] You have to save the value of "ans_informationGain" into your output file
# Here, let's assume that we split the input_data with 2/3 of the data in the left subtree and 1/3 in the right subtree
# Please round your answer to 4 decimal place
temp1 = np.zeros((int(input_data.shape[0]/3), 1), dtype=bool)
temp2 = np.ones(((input_data.shape[0]-int(input_data.shape[0]/3), 1)), dtype=bool)
temp_mask = np.concatenate((temp1, temp2))
df_mask = pd.DataFrame(temp_mask, columns=['mask'])
ans_informationGain = information_gain(input_data, df_mask['mask']).round(4)
print("ans_informationGain = ", ans_informationGain)
def find_best_split(data, impl_part):
  """
  This function will find the best split combination of data
  args:
  * data(type: DataFrame): the input data
  * impl_part(type: string): 'basic' or 'advanced' to specify which implementation to use
  return
  * best_ig(type: float): the best information gain you obtain
  * best_threshold(type: float): the value that splits data into 2 branches
  * best_feature(type: string): the feature that splits data into 2 branches
  """
  best_ig = -1e9
  best_threshold = 0
  best_feature = ''

  if(impl_part == 'basic'):
    # Implement this part of the function using the method we provided
    ### START CODE HERE ###
    features = list(data.columns)
    for feature in features:
      if feature == 'hospital_death':
        break
      data_sorted = data.sort_values(by = feature)
      mask = np.zeros((int(data.shape[0]), 1), dtype=bool)
      for i in range(data.shape[0] - 1):
        # mask[data_sorted.index[i]] = True
        mask[i] = True
        if (data.loc[data_sorted.index[i], feature] == data.loc[data_sorted.index[i + 1], feature]):
          continue
        df_mask = pd.DataFrame(mask, columns = ['mask'], index = data_sorted.index)
        ig = information_gain(data, df_mask['mask'])
        # print(df_mask)
        if ig > best_ig:
          best_ig = ig
          best_threshold = (data.loc[data_sorted.index[i], feature] + data.loc[data_sorted.index[i + 1], feature]) / 2
          best_feature = feature
          # print(best_ig, best_threshold, best_feature)
    ### END CODE HERE ###
  else:
    # You can implement another method here for the advanced part
    ### START CODE HERE ###
    advance = False
    ### END CODE HERE ###


  return float(best_ig), float(best_threshold), best_feature


# [Note] You have to save the value of "ans_ig", "ans_value", and "ans_name" into the output file
# Here, let's try to find the best split for the input_data
# Please round your answer to 4 decimal place
ans_ig, ans_value, ans_name = find_best_split(input_data, 'basic')
print("ans_ig = ", ans_ig)
print("ans_value = ", ans_value)
print("ans_name = ", ans_name)
def make_partition(data, feature, threshold):
  """
  This function will split the data into 2 branches
  args:
  * data(type: DataFrame): the input data
  * feature(type: string): the attribute(column name)
  * threshold(type: float): the threshold for splitting the data
  return:
  * left(type: DataFrame): the divided data that matches(less than or equal to) the assigned feature's threshold
  * right(type: DataFrame): the divided data that doesn't match the assigned feature's threshold
  """
  ### START CODE HERE ###
  left = data[data[feature] <= threshold]
  right = data[data[feature] > threshold]
  ### END CODE HERE ###

  return left, right


# [Note] You have to save the value of "ans_left" into the output file
# Here, let's assume the best split is when we choose bmi as the feature and threshold as 21.0
left, right = make_partition(input_data, 'bmi', 21.0)
ans_left = left.shape[0]
print("ans_left = ", ans_left)
def build_tree(data, max_depth, min_samples_split, depth):
  """
  This function will build the decision tree
  args:
  * data(type: DataFrame): the data you want to apply to the decision tree
  * max_depth: the maximum depth of a decision tree
  * min_samples_split: the minimum number of instances required to do partition
  * depth: the height of the current decision tree
  return:
  * subtree: the decision tree structure including root, branch, and leaf (with the attributes and thresholds)
  """
  ### START CODE HERE ###
  # check the condition of current depth and the remaining number of samples
  if depth < max_depth and data.shape[0] > min_samples_split:
    # call find_best_split() to find the best combination
    ig, threshold, feature = find_best_split(data, 'basic')
    # check the value of information gain is greater than 0 or not
    if ig > 0 :
      # update the depth
      depth += 1
      # call make_partition() to split the data into two parts
      left, right = make_partition(data, feature, threshold)
      # If there is no data split to the left tree OR no data split to the right tree
      if (left.empty or right.empty):
        # return the label of the majority
        # vote for the final result
        label = int(data['hospital_death'].mode().iloc[0])
        return label
      else:
        question = "{} {} {}".format(feature, "<=", threshold)
        subtree = {question: []}

        # call function build_tree() to recursively build the left subtree and right subtree
        left_subtree = build_tree(left, max_depth, min_samples_split, depth)
        right_subtree = build_tree(right, max_depth, min_samples_split, depth)
        
        if left_subtree == right_subtree:
          subtree = left_subtree
        else:
          subtree[question].append(left_subtree)
          subtree[question].append(right_subtree)
    else:
      # return the label of the majority
      # vote for the final result
      label = int(data['hospital_death'].mode().iloc[0])
      return label
  else:
    # return the label of the majority
    # vote for the final result
    label = int(data['hospital_death'].mode().iloc[0])
    return label
  ### END CODE HERE ###

  return subtree
# Here, let's build a decision tree using the input_data

decisionTree = build_tree(input_data, max_depth, min_samples_split, depth)
decisionTree
# [Note] You have to save the features in the "decisionTree" structure into the output file
def get_features_and_thresholds(tree, features, thresholds):
    children = list(tree.values())[0]
    if isinstance(children[0], dict):
        get_features_and_thresholds(children[0], features, thresholds)
    if isinstance(children[1], dict):
        get_features_and_thresholds(children[1], features, thresholds)
    feature = list(tree.keys())[0].split(' ')[0]
    threshold = float(list(tree.keys())[0].split(' ')[2])
    features.append(feature)
    thresholds.append(threshold)
    return

ans_features = []
ans_thresholds = []
get_features_and_thresholds(decisionTree, ans_features, ans_thresholds)
ans_features
# [Note] You have to save the corresponding thresholds for the features in the "ans_features" list into the output file
ans_thresholds
basic = []
basic.append(ans_entropy)
basic.append(ans_informationGain)
basic.append([ans_ig, ans_value, ans_name])
basic.append(ans_left)
basic.append(ans_features + ans_thresholds)
num_train = 30
num_validation = 10

training_data = input_data.iloc[:num_train]
validation_data = input_data.iloc[-num_validation:]

y_train = training_data[['hospital_death']]
x_train = training_data.drop(['hospital_death'], axis=1)

y_validation = validation_data[['hospital_death']]
x_validation = validation_data.drop(['hospital_death'], axis=1)
y_validation = y_validation.values.flatten()

print(input_data.shape)
print(training_data.shape)
print(validation_data.shape)
max_depth = 2
depth = 0
min_samples_split = 2
n_features = x_train.shape[1]
def classify_data(instance, tree):
  """
  This function will predict/classify the input instance
  args:
  * instance: a instance(case) to be predicted
  return:
  * answer: the prediction result (the classification result)
  """
  equation = list(tree.keys())[0]
  if equation.split()[1] == '<=':
    temp_feature = equation.split()[0]
    temp_threshold = equation.split()[2]
    if instance[temp_feature] > float(temp_threshold):
      answer = tree[equation][1]
    else:
      answer = tree[equation][0]
  else:
    if instance[equation.split()[0]] in (equation.split()[2]):
      answer = tree[equation][0]
    else:
      answer = tree[equation][1]

  if not isinstance(answer, dict):
    return answer
  else:
    return classify_data(instance, answer)


def make_prediction(tree, data):
  """
  This function will use your pre-trained decision tree to predict the labels of all instances in data
  args:
  * tree: the decision tree
  * data: the data to predict
  return:
  * y_prediction: the predictions
  """
  ### START CODE HERE ###
  # [Note] You can call the function classify_data() to predict the label of each instance
  y_prediction = []
  for i in range(data.shape[0]):
    y_prediction.append(classify_data(data.iloc[i], tree))
  ### END CODE HERE ###

  return y_prediction


def calculate_score(y_true, y_pred):
  """
  This function will calculate the f1-score of the predictions
  args:
  * y_true: the ground truth
  * y_pred: the predictions
  return:
  * score: the f1-score
  """
  score = f1_score(y_true, y_pred)

  return score
decision_tree = build_tree(training_data, max_depth, min_samples_split, depth)

y_pred = make_prediction(decision_tree, x_validation)

# [Note] You have to save the value of "ans_f1score" into your output file
# Please round your answer to 4 decimal place
ans_f1score = calculate_score(y_validation, y_pred)
ans_f1score = round(ans_f1score, 4)
print("ans_f1score = ", ans_f1score)
# This is just for you to check your predictions
y_pred
basic.append(ans_f1score)
basic_path = 'lab2_basic.csv'

basic_df = pd.DataFrame({'Id': range(len(basic)), 'Ans': basic})
basic_df.set_index('Id', inplace=True)
basic_df
basic_df.to_csv(basic_path, header = True, index = True)
advanced_training_data = pd.read_csv('lab2_advanced_training.csv')
advanced_training_data
advanced_testing_data = pd.read_csv('lab2_advanced_testing.csv')
advanced_testing_data
### START CODE HERE ###
training_data = advanced_training_data
validation_data = ...
### END CODE HERE ###
### START CODE HERE ###
# Define the attributes
max_depth = 10
depth = 0
min_samples_split = 5

# total number of trees in a random forest
n_trees = 500

# number of features to train a decision tree
n_features = 25

# the ratio to select the number of instances
sample_size = 0.7
n_samples = int(training_data.shape[0] * sample_size)
### END CODE HERE ###
def build_forest(data, n_trees, n_features, n_samples):
  """
  This function will build a random forest.
  args:
  * data: all data that can be used to train a random forest
  * n_trees: total number of tree
  * n_features: number of features
  * n_samples: number of instances
  return:
  * forest: a random forest with 'n_trees' of decision tree
  """
  ### START CODE HERE ###
  data_len = data.shape[0]
  feature_list = data.columns.tolist()[:-1]
  forest = []
  ### END CODE HERE ###

  # Create 'n_trees' number of trees and store each into the 'forest' list
  for i in range(n_trees):
    
    print(f"Building tree {i+1} out of {n_trees}")

    ### START CODE HERE ###
    # Select 'n_samples' number of samples and 'n_features' number of features
    # (you can select randomly or use any other techniques)

    selected_datas = random.sample(range(data_len), n_samples)
    selected_features = random.sample(feature_list, n_features)

    ### END CODE HERE ###

    # print(f"selected_datas = {selected_datas}")
    # print(f"selected_features = {selected_features}")

    ### START CODE HERE ###
    # Store the rows in 'selected_datas' from 'data' into a new DataFrame
    tree_data = pd.DataFrame()
    tree_data = data.loc[selected_datas]

    # Filter the DataFrame for specific 'selected_features' (columns)
    tree_data = data[selected_features + ['hospital_death']]

    ### END CODE HERE ###

    # Then use the new data and 'build_tree' function to build a tree
    tree = build_tree(tree_data, max_depth, min_samples_split, depth)
    # print(tree)
    print(f"Tree {i+1} is built")

    # Save your tree
    forest.append(tree)

  return forest
forest = build_forest(training_data, n_trees, n_features, n_samples)
def make_prediction_forest(forest, data):
  """
  This function will use the pre-trained random forest to make the predictions
  args:
  * forest: the random forest
  * data: the data used to predict
  return:
  * y_prediction: the predicted results
  """
  y_prediction = []
  predictions = []

  ### START CODE HERE ###
  # Loop through each tree in the forest
  for tree in forest:
    # Call 'make_prediction'
    pred = make_prediction(tree, data)
    predictions.append(pred)

  # Here, each tree has made its predictions.
  # We can use majority vote in which the final prediction is determined by the mode (most frequent prediction) across all the trees.
  # Feel free to use any other method to determine the final prediction

  # Loop through each column of 'predictions'
  for col in range(len(predictions[0])):
    # For a specific column, find out each tree's prediction
    column_predictions = np.array(predictions)[:, col]
    # Then, use a method to determine the final prediction for this column
    # append the final prediction to y_prediction
    if sum(column_predictions) / len(column_predictions) >= 0.5:
      y_prediction.append(1)
    else:
      y_prediction.append(0)
  ### END CODE HERE ###



  return y_prediction
### START CODE HERE ###
# pred_validation = make_prediction_forest(forest, x_validation)
# score = calculate_score(y_validation, pred_validation)
# print(score)
### END CODE HERE ###
y_pred_test = make_prediction_forest(forest, advanced_testing_data)
advanced = []
for i in range(len(y_pred_test)):
  advanced.append(y_pred_test[i])
advanced_path = 'lab2_advanced.csv'

advanced_df = pd.DataFrame({'Id': range(len(advanced)), 'hospital_death': advanced})
advanced_df.set_index('Id', inplace=True)
advanced_df
advanced_df.to_csv(advanced_path, header = True, index = True)