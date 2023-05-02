import pandas as pd
import pickle
import streamlit as st
from sklearn.preprocessing import StandardScaler

# Rename columns
Columns = (['duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent','hot',
            'num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root','num_file_creations',
            'num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count','srv_count',
            'serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate',
            'dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate',
            'dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate',
            'dst_host_srv_rerror_rate','attack','level'])

# Load data
@st.cache_data
def load_train_data():
    df_train = pd.read_csv('dataset/KDDTrain+.txt', sep = "," , encoding = 'utf-8', header = None)
    df_train.columns = Columns
    return df_train

@st.cache_data
def load_test_data(file_name):
    if file_name.endswith('.csv'):
        df_test = pd.read_csv('dataset/' + file_name, sep = "," , encoding = 'utf-8', header = None)
    elif file_name.endswith('.txt'):
        df_test = pd.read_csv('dataset/' + file_name, sep = "," , encoding = 'utf-8', header = None)
    else:
        st.warning('Please choose a .csv or .txt file')
    df_test.columns = Columns
    ### Record 'attack' feature into binary categorical values on 'target' column 
    df_test['target'] = df_test['attack'].apply(lambda x: 0 if x == 'normal' else 1)
    return df_test

@st.cache_data
def load_weights():
    train_columns, unique_class_dict, hidden_size, output_size, num_layers = pickle.load(open('model/weights.pkl','rb'))
    return train_columns, unique_class_dict, hidden_size, output_size, num_layers

@st.cache_data
def load_scaler(df):
    scaler = StandardScaler()
    columns_to_scale = df.columns.difference(['attack', 'target']) # Every column except these
#     loaded_scaler = pickle.load(open('model/scaler.pkl', 'rb'))
#     df[columns_to_scale] = loaded_scaler.transform(df[columns_to_scale])
    df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
    return df

### Creates equal number of columns between train and test for categorical features
### Prepares train and test data for unseen catagories
def prepare_columns(df, col_names):
    unique_class_dict = {}
    for col_name in col_names:
        ### Get the unique classes from the column and save them as a list
        unique_classes = df[col_name].unique().tolist()
        unique_class_dict[col_name] = unique_classes
        ### Create dummy variables for each categorical class and concatenate with the dataframe
        dummies = pd.get_dummies(df[col_name], prefix=col_name)
        df = pd.concat([df, dummies], axis=1)
        ### Add an additional column for unseen categories (initialize with zeros)
        df[col_name + '_unseen'] = 0
        ### Drop the original column
        df = df.drop(columns=[col_name])
    return df, unique_class_dict

### Applies the new encoded catogies to the test data and matches the train data
def encode_categorical_columns(test_df, unique_class_dict, train_columns):
    for col_name, unique_classes in unique_class_dict.items():
        ### Create dummy variables for each categorical class and concatenate with the dataframe
        dummies = pd.get_dummies(test_df[col_name], prefix=col_name)
        unseen_classes = set(test_df[col_name].unique()) - set(unique_classes)
        for uc in unique_classes:
            if uc not in dummies.columns:
                dummies[f'{col_name}_{uc}'] = 0
        test_df[f'{col_name}_unseen'] = 0
        for unseen_class in unseen_classes:
            if f'{col_name}_{unseen_class}' in dummies.columns:
                test_df.loc[test_df[col_name] == unseen_class, f'{col_name}_unseen'] = 1
                dummies = dummies.drop(columns=[f'{col_name}_{unseen_class}'])
        test_df = pd.concat([test_df, dummies], axis=1)
        test_df = test_df.drop(columns=[col_name])
    ### Reorder columns in the test dataframe to match the order of the train dataframe
    test_df = test_df.reindex(columns=train_columns, fill_value=0)
    return test_df


