import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score

# Streamlit setup
st.title("Anomaly Detection in Solar Power Generation")
st.write("Upload the dataset to detect anomalies in solar power generation data.")

# File upload
uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")
if uploaded_file is not None:
    with st.spinner("Loading data..."):
        inverter1 = pd.read_excel(uploaded_file, sheet_name=1, skiprows=lambda x: x == 1)
        inverter2 = pd.read_excel(uploaded_file, sheet_name=2, skiprows=lambda x: x == 1)
        inverter3 = pd.read_excel(uploaded_file, sheet_name=3, skiprows=lambda x: x == 1)
        weather = pd.read_excel(uploaded_file, sheet_name=5, parse_dates=['DATE & TIME'])

    # Preprocess data as per original code
    inverter1['DATE & TIME'] = pd.to_datetime(inverter1['DATE & TIME'], format='mixed').dt.floor('s')
    inverter2['DATE & TIME'] = pd.to_datetime(inverter2['DATE & TIME'], format='mixed').dt.floor('s')
    inverter3['DATE & TIME'] = pd.to_datetime(inverter3['DATE & TIME'], format='mixed').dt.floor('s')
    inverter1.set_index('DATE & TIME', inplace=True)
    inverter2.set_index('DATE & TIME', inplace=True)
    inverter3.set_index('DATE & TIME', inplace=True)
    inverter1_resampled = inverter1.resample('5min').mean()
    inverter2_resampled = inverter2.resample('5min').mean()
    inverter3_resampled = inverter3.resample('5min').mean()
    
    inverter1_resampled['Inverter'] = 'INVERTER1'
    inverter2_resampled['Inverter'] = 'INVERTER2'
    inverter3_resampled['Inverter'] = 'INVERTER3'
    inverters = pd.concat([inverter1_resampled, inverter2_resampled, inverter3_resampled])
    
    weather['DATE & TIME'] = pd.to_datetime(weather['DATE & TIME'], format='mixed').dt.floor('s')
    weather.set_index('DATE & TIME', inplace=True)
    weather_resampled = weather.resample('5min').mean()
    data = pd.merge(inverters, weather_resampled, on='DATE & TIME', how='outer')

    data = data.drop(columns=["RAIN", "WIND DIRECTION", "WIND SPEED"])
    columns_to_remove = ['GHI', 'GII', 'DC CURRENT', 'MODULE TEMP.1', 'MODULE TEMP.2']
    data = data.drop(columns=columns_to_remove)
    
    numerical_columns = data.columns.difference(['Inverter'])
    data[numerical_columns] = data[numerical_columns].fillna(0)

    data['Inverter'] = data['Inverter'].astype('category')
    data['DC VOLTAGE'] = data['DC VOLTAGE'].astype('float32')
    data['DC POWER'] = data['DC POWER'].astype('float32')
    data['TEMPERATURE'] = data['TEMPERATURE'].astype('float32')
    data['HUMIDITY'] = data['HUMIDITY'].astype('float32')
    data['AMBIENT TEMPERATURE'] = data['AMBIENT TEMPERATURE'].astype('float32')
    
    data['Hour'] = data.index.hour.astype('int8')
    data['Day'] = data.index.day.astype('int8')
    data['Month'] = data.index.month.astype('int8')
    bins = [-1, 4, 9, 14, 18]
    labels = ['Night', 'Morning', 'Afternoon', 'Evening']
    data['TimeOfDay'] = pd.cut(data['Hour'], bins=bins, labels=labels, right=True, include_lowest=True)
    data.loc[data['Hour'].between(19, 23), 'TimeOfDay'] = 'Night'
    data = data[data['TimeOfDay'] != 'Night']
    
    scaler = StandardScaler()
    data_scaled = data[["DC VOLTAGE", "DC POWER", "TEMPERATURE", "HUMIDITY", "AMBIENT TEMPERATURE"]].copy()
    data_scaled = scaler.fit_transform(data_scaled.astype('float32'))

    iso_forest = IsolationForest(n_estimators=100, contamination=0.01, random_state=42, n_jobs=-1)
    oc_svm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.01)
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.01, novelty=True)
    
    iso_forest.fit(data_scaled)
    data['Anomaly_IF'] = iso_forest.fit_predict(data_scaled)
    data['Anomaly_SVM'] = oc_svm.fit_predict(data_scaled)
    lof.fit(data_scaled)
    data['Anomaly_LOF'] = lof.predict(data_scaled)
    
    anomalies = data[data['Anomaly_IF'] == -1]
    anomalies_svm = data[data['Anomaly_SVM'] == -1]
    anomalies_lof = data[data['Anomaly_LOF'] == -1]

    # Display evaluation metrics for models
    def evaluation_metrics(model_name, anomaly_labels):
        tn, fp, fn, tp = confusion_matrix(data['Anomaly_IF'], anomaly_labels, labels=[1, -1]).ravel()
        accuracy = accuracy_score(data['Anomaly_IF'], anomaly_labels)
        precision = precision_score(data['Anomaly_IF'], anomaly_labels, pos_label=-1)
        recall = recall_score(data['Anomaly_IF'], anomaly_labels, pos_label=-1)
        f1 = f1_score(data['Anomaly_IF'], anomaly_labels, pos_label=-1)
        
        st.write(f"{model_name} Metrics:")
        st.write(f"Accuracy: {accuracy:.4f}")
        st.write(f"Precision: {precision:.4f}")
        st.write(f"Recall: {recall:.4f}")
        st.write(f"F1-Score: {f1:.4f}")
        st.write(f"Confusion Matrix: {tn, fp, fn, tp}")
        st.write("="*50)

    # Display metrics
    evaluation_metrics("One-Class SVM", data['Anomaly_SVM'])
    evaluation_metrics("Local Outlier Factor (LOF)", data['Anomaly_LOF'])

    # Plot anomalies
    st.write("### Anomaly Visualization")

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.scatter(anomalies.index, anomalies['DC POWER'], color='red', label='IF Anomalies', s=10)
    ax.scatter(anomalies_svm.index, anomalies_svm['DC POWER'], color='blue', label='SVM Anomalies', s=10)
    ax.scatter(anomalies_lof.index, anomalies_lof['DC POWER'], color='green', label='LOF Anomalies', s=10)
    ax.legend()
    ax.set_title('DC POWER with Anomalies Highlighted (IF, SVM, LOF)')
    ax.set_xlabel('Time')
    ax.set_ylabel('DC POWER (scaled)')
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.scatter(anomalies['AMBIENT TEMPERATURE'], anomalies['DC POWER'], color='red', label='IF Anomalies', s=10)
    ax.scatter(anomalies_svm['AMBIENT TEMPERATURE'], anomalies_svm['DC POWER'], color='blue', label='SVM Anomalies', s=10)
    ax.scatter(anomalies_lof['AMBIENT TEMPERATURE'], anomalies_lof['DC POWER'], color='green', label='LOF Anomalies', s=10)
    ax.legend()
    ax.set_title('DC POWER vs AMBIENT TEMPERATURE with Anomalies')
    ax.set_xlabel('AMBIENT TEMPERATURE')
    ax.set_ylabel('DC POWER (scaled)')
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.scatter(anomalies.HUMIDITY, anomalies['DC POWER'], color='red', label='IF Anomalies', s=10)
    ax.scatter(anomalies_svm.HUMIDITY, anomalies_svm['DC POWER'], color='blue', label='SVM Anomalies', s=10)
    ax.scatter(anomalies_lof.HUMIDITY, anomalies_lof['DC POWER'], color='green', label='LOF Anomalies', s=10)
    ax.legend()
    ax.set_title('DC POWER vs HUMIDITY with Anomalies')
    ax.set_xlabel('HUMIDITY')
    ax.set_ylabel('DC POWER (scaled)')
    st.pyplot(fig)

    # Display the final data with anomalies
    st.write("### Data with Anomalies")
    st.dataframe(data)
else:
    st.write("Please upload a file to start anomaly detection.")
