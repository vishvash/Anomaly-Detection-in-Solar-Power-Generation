# Solar Power Anomaly Detection System

This project is a **Streamlit-based application** for detecting anomalies in solar power generation data using machine learning models such as **Isolation Forest**, **One-Class SVM**, and **Local Outlier Factor (LOF)**. The system processes solar inverter and weather data to identify deviations that may affect production capacity and efficiency. 

---

## 📑 Features
- **Data Preprocessing**:
  - Handles duplicates, missing values, and irrelevant features.
  - Resampling data to 5-minute intervals.
  - Feature engineering and scaling.

- **Anomaly Detection Models**:
  - Isolation Forest
  - One-Class SVM
  - Local Outlier Factor (LOF)

- **Performance Metrics**:
  - Accuracy, Precision, Recall, F1-Score
  - Confusion matrix for detailed evaluation.

- **Interactive Visualization**:
  - Plotting anomalies over time and across key metrics (e.g., `DC POWER`, `HUMIDITY`, `AMBIENT TEMPERATURE`).

---

## 🚀 How to Run the Project
### Prerequisites
1. Python 3.8+
2. Install the required libraries:
   ```bash
   pip install pandas numpy scikit-learn matplotlib streamlit
## 🚀 Steps to Run the Project

1. **Clone this repository**:  
   ```bash
   git clone https://github.com/your-repo-name/solar-anomaly-detection.git
   cd solar-anomaly-detection

2. **Run the Streamlit application:**
   ```bash
   streamlit run app.py

3. **Upload your dataset:**
   Upload your Excel file containing inverter data and weather data to start anomaly detection.


# Anomaly Detection in Solar Inverter Data

## 📊 Input Data

### Inverter Data:
- **DC CURRENT**
- **DC VOLTAGE**
- **DC POWER**
- **TEMPERATURE**
- Timestamp for each reading.

### Weather Data:
- **GHI** (Global Horizontal Irradiance)
- **GII** (Global Inclined Irradiance)
- **HUMIDITY**
- **AMBIENT TEMPERATURE**
- Other relevant features.

---

## 🔍 Models Used
1. **Isolation Forest**: Detects anomalies based on decision trees.
2. **One-Class SVM**: Identifies outliers using support vector machines.
3. **Local Outlier Factor (LOF)**: Measures the local density deviation of a given data point.

### Performance Metrics:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**

---

## 🖼️ Example Visualizations
1. **DC POWER with Anomalies Highlighted**  
   Shows deviations detected by models over time.

2. **DC POWER vs AMBIENT TEMPERATURE**  
   Helps analyze the effect of temperature on power generation.

3. **DC POWER vs HUMIDITY**  
   Highlights anomalies under varying humidity levels.

---

## 🛠️ Code Highlights
### Preprocessing:
- Resampling, handling missing data, and feature selection.

### Scaling:
- Using `StandardScaler` for normalizing numerical features.

### Model Training:
- Implementation of **Isolation Forest**, **One-Class SVM**, and **Local Outlier Factor (LOF)**.

---

## 📚 Dependencies
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `streamlit`

---

## 📝 Future Improvements
- Add support for **real-time anomaly detection**.
- Incorporate additional models for **comparative analysis**.
- Enhance visualizations with more **interactive plots**.
