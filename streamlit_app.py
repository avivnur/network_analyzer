import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from utils import load_train_data, load_test_data, load_weights, encode_categorical_columns, load_scaler
from models import load_SVM, AttackPredictor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import OneClassSVM

# configuration of the page
st.set_page_config(
    page_title="Network Logs Analysis",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "# Network Logs Analysis - WPI",
    }
)

# load dataframes
df_train = load_train_data()

# sidebar
st.sidebar.header('Network Logs Analysis 👨‍💻')

# List files
files_test = [
    'KDDTest+.txt',
    'test_known.csv',
    'test_normal.csv',
    'test_unknown.csv',
    ]

# make dropdown menu on the left side using 'attack' column
selected_file = st.sidebar.selectbox('Select a file', files_test)
test = load_test_data(selected_file)

# load weights
train_columns, unique_class_dict, hidden_size, output_size, num_layers = load_weights()

### Create a copy for comparision and displaying
test_unscaled = test

### Process test data
test = encode_categorical_columns(test, unique_class_dict, train_columns)

# scale test data
test = load_scaler(test)

st.markdown("<h1 style='text-align: center; color: blue;'>Analysis of Network Logs to Identify Malicious or Suspicious Behaviors</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: black;'>KDD Cup 1999 Data Set is a collection of network logs that are used to identify malicious or suspicious behaviors.</p>", unsafe_allow_html=True)
st.divider()
# make bar chart using 'attack' column from train and test data
# split the page into two columns
col1, col2 = st.columns(2)
with col1:
    attack = df_train['attack'].value_counts()[:5]
    fig = plt.figure(figsize=(10, 10))
    sns.barplot(x=attack.values, y=attack.index, palette='Set2')
    plt.title('Attack Types')
    plt.xlabel('Number of Attacks in train data')
    plt.ylabel('Attack Types')
    st.pyplot(fig)
with col2:
    attack = test['attack'].value_counts()[:5]
    fig = plt.figure(figsize=(10, 10))
    sns.barplot(x=attack.values, y=attack.index, palette='Set2')
    plt.title('Attack Types')
    plt.xlabel('Number of Attacks in test data')
    plt.ylabel('Attack Types')
    st.pyplot(fig)

st.divider()

# Load SVM
svm = OneClassSVM(nu=0.05)
svm = load_SVM()

analyze = st.sidebar.button('Analyze')
if analyze:
    ### Apply the anomaly detection model to the test data
    test_predictions = svm.predict(test.drop(columns=['attack', 'target']).values)
    ### Label the anomalous instances as class 2
    test.loc[test_predictions == -1, 'target'] = 2
    ### Remove the anomalous instances (class 2) from the test data.
    test_non_anomalous = test[test['target'] != 2]

    st.header(':pushpin: Detecting and alerting any Anomalous traffic (SVM)')
    ### Taking only the 'target' column
    predicted_target_df = test['target'] 
    ### Droping the 'target' column from the original test dataframe
    connect_test_df = test_unscaled.drop('target', axis=1)
    ### join the dataframes horizontally
    anomalous_df = pd.concat([predicted_target_df, connect_test_df], axis=1)
    ### check if there are any rows where target is 2
    if (anomalous_df['target'] == 2).any():
        st.warning('ATTENTION!!! ATTENTION!!! ATTENTION!!!', icon="⚠️")
        st.warning('Suspicious Traffic Detected. Quarantined for further investigation!', icon="⚠️")

        ### create a new dataframe that contains only the rows where target is 2
        target_2_df = anomalous_df[anomalous_df['target'] == 2]
        st.info(f"Total number of anomalous traffic detected: {len(target_2_df)}")
        ### sort df by the 'column1' column in ascending order
        target_2_df = target_2_df.sort_values(by='duration', ascending=False)

        with st.expander("Click to view the Anomalous Traffic"):
            st.dataframe(target_2_df[['duration', 
                        'src_bytes',
                        'dst_bytes',
                        'protocol_type', 
                        'service',
                        'flag']].head(30))
        ### save the new dataframe to a CSV file
        target_2_df.to_csv("dataset/anomalous_traffic.csv", index=False)
        print()
        st.info("Successfully save anomalous_traffic.csv.", icon="📁")
    else:
        st.write("You're clean.")
        st.success('No suspicious traffic detected.', icon="👍")

    ### Load the trained model
    ### Initializing the weights (similar to the training)
    input_size = test.shape[1] - 2 # Exclude 'attack' and 'target' columns
    hidden_size = hidden_size # Same values from the weights.pkl
    output_size = output_size # Same values from the weights.pkl
    num_layers = num_layers   # Same values from the weights.pkl

    # Load LSTM
    model = AttackPredictor(input_size, hidden_size, output_size, num_layers)
    model.load_state_dict(torch.load("model/lstm_attack_predictor.pth", map_location='cpu'))

    ### Make predictions on the test set
    model.eval()
    with torch.no_grad():
        X_test = torch.tensor(test_non_anomalous.drop(columns=['attack', 'target']).values, dtype=torch.float32)
        y_test = torch.tensor(test_non_anomalous['target'].values, dtype=torch.long)
        test_outputs = model(X_test.unsqueeze(1))
        test_predictions_lstm = torch.argmax(test_outputs, dim=1)

    ### Create a DataFrame with predicted Y and actual Y
    results_dict = {'predicted_y': test_predictions_lstm.cpu().numpy(),
                    'actual_y': y_test.cpu().numpy()}
    results_df = pd.DataFrame(results_dict)

    accuracy = accuracy_score(results_df['actual_y'], results_df['predicted_y'])
    with st.expander("Click to view the accuracy of the LSTM model"):
        st.write(f"Accuracy: {accuracy*100:.3f}%")
        st.write(results_df.head(1))

    st.header(':pushpin: Detecting and alerting any Anomalous traffic (LSTM)')
    
    ### Taking only the 'target' column
    predicted_target_df_lstm = results_df['predicted_y'] 
    ### Droping the 'target' column from the original test dataframe
    connect_test_df_lstm = test_unscaled.drop('target', axis=1)
    ### join the dataframes horizontally
    lstm_predicted_df = pd.concat([predicted_target_df_lstm, connect_test_df_lstm], axis=1)
    
    ### check if there are any rows where target is 2
    if (lstm_predicted_df['predicted_y'] == 1).any():
        st.warning('ATTENTION!!! ATTENTION!!! ATTENTION!!!', icon="⚠️")
        st.warning('Malicious Traffic Detected. Quarantined for further investigation!', icon="⚠️")
        ### create a new dataframe that contains only the rows where target is 2
        target_1_df = lstm_predicted_df[lstm_predicted_df['predicted_y'] == 1]
        st.write(f"Total number of Malicious traffic detected: {len(target_1_df)}")
        ## sort df by the 'column1' column in ascending order
        target_2_df = target_2_df.sort_values(by='duration', ascending = False)
        with st.expander("Click to view the Malicious Traffic"):
            st.dataframe(target_2_df[['duration', 
                            'src_bytes',
                            'dst_bytes',
                            'protocol_type', 
                            'service',
                            'flag']].head(30))
        ### save the new dataframe to a CSV file
        target_2_df.to_csv("dataset/lstm_anomalous_traffic.csv", index=False)
        st.info("Successfully save lstm_anomalous_traffic.csv.", icon="📁")
    else:
        st.write("There are no rows with target = 1 in the lstm_predicted_df dataframe.")
        st.success('No Malicious traffic detected.', icon="👍")

