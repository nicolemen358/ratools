#!/usr/bin/env python
# coding: utf-8


# Importing all packages
import json
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from sklearn import preprocessing
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


####################################################DATA AND CONFIG LOADING#########################################################
# Setting streamlit page configurations
st.set_page_config(layout="wide",page_icon='📊')

 # Loading the config file 
config = json.load(open('config.json'))

# Define functions to load data
#@st.cache
def load_data():
    df = pd.read_csv(config['data_file'])
    return df.sort_index()

# Calling the load data function
df = load_data()

# Setting the title for the page
st.title("Regression Analysis on Recidivism Indicators")

# Writing the initial page descriptions
st.write(
    """
    We will be examining how certain indicators predict recidivism by using the COMPAS data set.
    We will be using linear regression and ridge regression to see the indicators and how strongly they predict recidivism.
    This is important as it helps us plan for future and evaluate the current strategies to mitigate bias.
    """)


###################################################FEATURE ENGINEERING AND TRAINING#################################################
# Method to train a Logistic Regression model and get corresponding predictions.
def train(df):
    # Initializing a Logistic Regression Model
    model_without_transform = LogisticRegression()

    model_without_transform.fit(X_train_features, y_train)
    y_pred = model_without_transform.predict(X_test_features)
    return model_without_transform, y_pred

# Setting 3 columns of the application
left_column, middle_column, right_column = st.columns(3)

# Getting the baseline feature columns from the config file
all_feature_columns_raw = config['baseline_model_configs']['column_groups_baseline'].values()

# Flattening the list
all_feature_columns = []
for l in all_feature_columns_raw:
    for x in l:
        all_feature_columns.append(x)

print(all_feature_columns)

# Getting the output column name from the config file
output_columns = config['baseline_model_configs']['output_column']

# Getting the categorical columns from the config file in order to convert them to numeric using LabelEncoding
for cat_column in config['baseline_model_configs']['categorical_columns']:
    if cat_column in df.columns:
        le_cat = preprocessing.LabelEncoder()
        df[cat_column] = le_cat.fit_transform(df[cat_column])

# Dropping NAs
df = df.dropna()

print(df['race'])

#################################################TRAINING AND DISPLAYING THE BASELINE MODEL DETAILS#################################
# Putting in widgets in the left columns
with left_column:
    # Setting the header for the left column
    st.header("Overall model results")    

    # Creating the correlation dataframe
    corr_df = df.corr()

    print(df.columns)
    baseline_features_df = df[list(all_feature_columns + output_columns)]

    # Dropping rows with null values.
    #baseline_features_df = baseline_data.dropna()

    # Selecting the output column mentioned in the config file.
    y = baseline_features_df[config['baseline_model_configs']['output_column']]


    # Splitting the data into 80:20 for train adn test respectively.
    X_train, X_test, y_train, y_test  = train_test_split(df.drop(config['baseline_model_configs']['output_column'],axis=1),y, test_size=0.2,shuffle=False)

    # Selecting the feature columns in training and testing data.
    X_train_features = X_train[all_feature_columns]
    X_test_features = X_test[all_feature_columns]

    # print(X_train_features)

    # Training the model using the train() function.
    baseline_model,baseline_predictions = train(df)


###################################################GETTING THE SUBSET OF DATA BASED ON USER SELECTIONS###########################################
# Setting up a sidebar to put control widgets
with st.sidebar:

    # Initializing lists for selecting columns
    all_selected_columns = []
    selection_dict = {}

    #Looping through the column groups to get the selected columns (number of dropdowns would be equal to the number of column group keys 
    # mentioned in the config file)
    for column_group, cols in config['column_groups'].items():
        # Asking the user for the selection
        selection = st.selectbox("Select features from below to be used in Regression:",options = cols)

        # Storing user's choice of the column for the column group.
        selection_dict[column_group] = selection
        # Storing the columns in a list.
        all_selected_columns.append(selection)

    # print(all_selected_columns)
    # print(output_columns)

    # Selecting the columns based on user selections + the output column mentioned in the config file.
    selected_df = df[all_selected_columns + output_columns]

    # Dropping NAs
    selected_df = selected_df.dropna()
    # test size
    test_size = st.sidebar.number_input(
                    'Please choose your train-test split ratio: (range: 0.2-0.4):',
                    min_value=0.2,
                    max_value=0.4,
                    value=0.4,
                    step=0.05,
                        )


###############################################SPLITTING AND TRAINING A MODEL ON THE SELECTED DATA#######################################
# Selecting the output column
y = selected_df[output_columns]

# Defining the selected features.
X = selected_df

# Splitting the selected dataframe training and test features.
X_train_selection = X_train[all_selected_columns]
X_test_selection = X_test[all_selected_columns]

# Splitting the output column for training and test.
y_train_selection = y_train
y_test_selection = y_test

# print(X_train_selection)
# print(X_test_selection)
# print(y_train_selection)
# print(y_test_selection)

# Defining the model to train on the selected dataframe.
model_without_transform = LogisticRegression()

# Training the model defined above on the train set of the selected dataframe.
model_without_transform.fit(X_train_selection, y_train_selection)

# Getting the predictions on the test set of the selected dataframe.
y_pred = model_without_transform.predict(X_test_selection)
    

###################################################CALCULATING AND STORING THE VARIOUS STATISTICS FOR THE MODEL####################################

# Function to get outcome flips.
def get_outcome_flips(y_pred):
    #Initializing the change output lists.
    difference_in_predictions =[]
    high_to_low_preds = []
    low_to_high_preds = []

    # Iterating through the predictions to log the change in baseline outputs vs new model outputs.
    for i,pred in enumerate(y_pred):

        # If the baseline predicition is not equal to the new prediction, log as 1
        if baseline_predictions[i] == pred:
            difference_in_predictions.append(0)
        else:
            difference_in_predictions.append(1)

        # If the baseline prediction is high risk and new model prediction is low risk, log as 1
        if baseline_predictions[i]==1:
            if pred==0:
                high_to_low_preds.append(1)
            else:
                high_to_low_preds.append(0)
        
        # If the baseline prediction is low risk and new model prediction is high risk, log as 1
        if baseline_predictions[i]==0:
            if pred==1:
                low_to_high_preds.append(1)
            else:
                low_to_high_preds.append(0)
        


    # Calculating the percentage change in overall output
    percent_change_in_output = np.sum(difference_in_predictions)/len(difference_in_predictions)
    #st.write("Difference in Predictions: ", percent_change_in_output)


    # Calculating the percentage change from low to high and high to low.
    percent_change_from_high_to_low = np.sum(high_to_low_preds)/len(high_to_low_preds)
    percent_change_from_low_to_high = np.sum(low_to_high_preds)/len(low_to_high_preds)

    return percent_change_in_output, percent_change_from_high_to_low, percent_change_from_low_to_high



# Writing the various model statistics in the left column.
with left_column:

    # Getting the probabilities of the predictions.
    y_pred_prob = model_without_transform.predict_proba(X_test_selection)[:,1]

    # Calculating the Area under the curve for the model.
    auc_score = roc_auc_score(y_test_selection, y_pred_prob)

    # Calculating the confusion matrix for the given model and its predictions.
    cm = confusion_matrix(y_test_selection, model_without_transform.predict(X_test_selection))

    # Calculating and writing the accuracy for the model on the app.
    st.write("**Accuracy**")
    st.markdown('<p class="big-font">' + str(accuracy_score(y_test_selection,y_pred))+'</p>', unsafe_allow_html=True)

    # Calculating and writing the area under the curve for the model on the app.
    st.write("**Area Under the Curve**")
    st.markdown('<p class="big-font">' + str(auc_score)+'</p>', unsafe_allow_html=True)

    # Calculating and writing the false positive rate for the model on the app.
    st.write("**False Positive Rate**")
    st.markdown('<p class="big-font">' + str(cm.ravel()[1]/(cm.ravel()[1] + cm.ravel()[0]))+'</p>', unsafe_allow_html=True)

    # Calculating and writing the false negative rate for the model on the app.
    st.write("**False Negative Rate**")
    st.markdown('<p class="big-font">' + str(cm.ravel()[2]/(cm.ravel()[2] + cm.ravel()[3]))+'</p>', unsafe_allow_html=True)

    # Calculating and witing the positive predictive value of the model on the app.
    st.write("**Positive Predictive Value(PPV)**")
    st.markdown('<p class="big-font">' + str(cm.ravel()[3]/(cm.ravel()[3] + cm.ravel()[1]))+'</p>', unsafe_allow_html=True)


    # alt.Chart(cm).mark_rect().encode(
    # x='x:O',
    # y='y:O',
    # color='z:Q'
    # )

    # fpr, tpr, thresholds = roc_curve(y_test_selection, y_pred_prob)
    # roc_df = pd.DataFrame()
    # roc_df['fpr'] = fpr
    # roc_df['tpr'] = tpr
    # roc_df['thresholds'] = thresholds
    # roc_df.head()


    # alt.Chart(roc_df).mark_line(color = 'red').encode(
    #                                                 alt.X('fpr', title="false positive rate"),
    #                                                 alt.Y('tpr', title="true positive rate"))

    #result_with_races_positive = result_with_races[result_with_races['y_predict']==1]

    # Calculating the outcome flip results using the function defined above.
    percent_change_in_output, percent_change_from_high_to_low, percent_change_from_low_to_high = get_outcome_flips(y_pred)

    # Writing the outcome flip results.
    st.write("Difference in Predictions: ", percent_change_in_output)
    st.write("Percentage change from high to low: ", percent_change_from_high_to_low)
    st.write("Percentage change from low to high: ", percent_change_from_low_to_high)

############################################CALCULATING AND DISPLAYING RACE-WISE STATISTICS###########################################


# st.write(roc_df)
df = load_data()

# Getting the indices with the test selections.
result_with_races = df.loc[list(X_test_selection.index)]

# Storing the actual and predicted outputs.
result_with_races['y_test'] = y_test_selection
result_with_races['y_predict'] = y_pred
result_with_races['y_predict_prob'] = y_pred_prob

# Storing the baseline predictions.
result_with_races['baseline_prediction'] = baseline_predictions


# Printing the baseline predictions.
print("Predictions")
print(len(baseline_predictions))
print(len(y_pred))

# Initializing the middle column to display the statistics and results for the African-American Race.
with middle_column:

    # Defining the header for the section.
    st.header("Model Results for African-American race")

    # Getting the indices with the "Black/African American" race.
    result_with_races_black = result_with_races[result_with_races['race'] == 'Black/African American']

    # Calculating the area under the curve for the african-american race.
    auc_score = roc_auc_score(result_with_races_black['y_test'], result_with_races_black['y_predict_prob'])

    # Calculating the confusion matrix for the african-american race.
    cm = confusion_matrix(result_with_races_black['y_test'], result_with_races_black['y_predict'])

    # Calculating and displaying the accuracy for he african-american race.
    st.write("**Accuracy**")
    st.markdown('<p class="big-font">' + str(accuracy_score(result_with_races_black['y_test'],result_with_races_black['y_predict']))+'</p>', unsafe_allow_html=True)

    # Calculating and displaying the area under the curve for the african-american race.
    st.write("**Area Under the Curve**")
    st.markdown('<p class="big-font">' + str(auc_score)+'</p>', unsafe_allow_html=True)

    # Calulating and displaying the false positive rate for the african-american race.
    st.write("**False Positive Rate**")
    st.markdown('<p class="big-font">' + str(cm.ravel()[1]/(cm.ravel()[1] + cm.ravel()[0]))+'</p>', unsafe_allow_html=True)

    # Calculating and displaying the false negative rate for the african-american race.
    st.write("**False Negative Rate**")
    st.markdown('<p class="big-font">' + str(cm.ravel()[2]/(cm.ravel()[2] + cm.ravel()[3]))+'</p>', unsafe_allow_html=True)

    # Calculating and displaying the positive predictive value for the african-american race.
    st.write("**Positive Predictive Value(PPV)**")
    st.markdown('<p class="big-font">' + str(cm.ravel()[3]/(cm.ravel()[3] + cm.ravel()[1]))+'</p>', unsafe_allow_html=True)


    # Gettting the positive predictions for the african-american race.
    result_with_races_positive = result_with_races[result_with_races['y_predict']==1]

    # Getting the outcome flips for the african-american race.
    percent_change_in_output_black, percent_change_from_high_to_low_black, percent_change_from_low_to_high_black = get_outcome_flips(result_with_races_black['y_predict'])

    # Writing the results for the african-american race.
    st.write("Difference in Predictions: ", percent_change_in_output_black)
    st.write("Percentage change from high to low: ", percent_change_from_high_to_low_black)
    st.write("Percentage change from low to high: ", percent_change_from_low_to_high_black)


# Initializing the right column to display the statistics for the caucasian race.  
with right_column:

    # Defining the header for the section.
    st.header("Model Results for Caucasian race")

    # Getting the indices for the white/caucasian race.
    result_with_races_white = result_with_races[result_with_races['race'] == 'White']

    # Calculating and displaying the area under the curve for the caucasian race.
    auc_score = roc_auc_score(result_with_races_white['y_test'], result_with_races_white['y_predict_prob'])

    # Calculating the confusion matrix for the caucasian race.
    cm = confusion_matrix(result_with_races_white['y_test'], result_with_races_white['y_predict'])

    # Calculating the accuracy for the caucasian race.
    st.write("**Accuracy**")
    st.markdown('<p class="big-font">' + str(accuracy_score(result_with_races_white['y_test'],result_with_races_white['y_predict']))+'</p>', unsafe_allow_html=True)

    # Calculating and displaying the area under the curve for the caucasian race.
    st.write("**Area Under the Curve**")
    st.markdown('<p class="big-font">' + str(auc_score)+'</p>', unsafe_allow_html=True)

    # Calulating and displaying the false positive rate for the caucasian race.
    st.write("**False Positive Rate**")
    st.markdown('<p class="big-font">' + str(cm.ravel()[1]/(cm.ravel()[1] + cm.ravel()[0]))+'</p>', unsafe_allow_html=True)

    # Calculating and displaying the false negative rate for the caucasian race.
    st.write("**False Negative Rate**")
    st.markdown('<p class="big-font">' + str(cm.ravel()[2]/(cm.ravel()[2] + cm.ravel()[3]))+'</p>', unsafe_allow_html=True)

    # Calculating and displaying the positive predictive value for the caucasian race.
    st.write("**Positive Predictive Value(PPV)**")
    st.markdown('<p class="big-font">' + str(cm.ravel()[3]/(cm.ravel()[3] + cm.ravel()[1]))+'</p>', unsafe_allow_html=True)

    # Getting the positive predictions for the caucasian race.
    result_with_races_positive = result_with_races[result_with_races['y_predict']==1]

    # Getting the outcome flips for the caucasian race.
    percent_change_in_output_white, percent_change_from_high_to_low_white, percent_change_from_low_to_high_white = get_outcome_flips(result_with_races_white['y_predict'])

    # Writing the results of the outcome flips.``
    st.write("Difference in Predictions: ", percent_change_in_output_white)
    st.write("Percentage change from high to low: ", percent_change_from_high_to_low_white)
    st.write("Percentage change from low to high: ", percent_change_from_low_to_high_white)




# Calculating the feature flips.
st.write("Feature Flips: ")

# Looping through all column groups in the baseline model as defined in the config file.
for col_group in config['baseline_model_configs']['column_groups_baseline']:

    # Checking if the column group is 'others' or not.
    if col_group!='others':

        # Getting the baseline column group selection for the given column group.
        baseline_group = config['baseline_model_configs']['column_groups_baseline'][col_group][0]

        # Getting the group selection for the column group.
        group_selection = selection_dict[col_group]

        # Checking the value of the baseline column group selection and the user-selected column.
        comparison_column = np.where(df[baseline_group] == df[group_selection], 1, 0)

        # Calculating and displaying percentage change of the column group.
        st.write("Percent change for ", col_group)
        st.write(np.sum(comparison_column)/len(df))

        # Initializing variable to store outcome flips.
        outcome_flips = []

        # Filling out the NAs with 0s for the outcome column.
        df['nca_flag'] = df['nca_flag'].fillna(0)

        # Looping through the comparison result column
        for i,row in enumerate(comparison_column):
            # If the column value for a particular row is equal, then AND the result with the outcome column and add to the outcome_flips list.
            if row == 0:
                outcome_flips.append(row & int(df['nca_flag'].iloc[i]))
        
        # Calculating and displaying the outcome flip.
        st.write("Outcome flips:")
        st.write(np.sum(outcome_flips)/np.sum(comparison_column))
