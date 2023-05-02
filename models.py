from sklearn.svm import OneClassSVM
import pickle
import os
from joblib import dump, load
import streamlit as st
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')

@st.cache_data
def load_SVM():
    model_name = 'svm_model.pkl'
    path = 'model'

    #### Initialize one-class SVM model
    svm = OneClassSVM(nu=0.05)  # Set nu value based on the desired contamination rate
    svm = pickle.load(open('model/svm_model.pkl', 'rb'))
    return svm

### Define LSTM Class 
class AttackPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(AttackPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out