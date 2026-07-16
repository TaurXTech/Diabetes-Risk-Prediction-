
# 🩺 Diabetes Risk Prediction using Machine Learning

A web-based machine learning application built with **Python** and **Streamlit** that predicts whether a person is at risk of diabetes based on health-related input features.

## 📌 Project Overview

This application uses a trained Machine Learning model to predict the likelihood of diabetes from user-provided medical information. The model is integrated into an interactive Streamlit web application, allowing users to enter their health information and receive instant predictions.

---

## 🚀 Features

- Interactive web interface built with Streamlit
- Predicts diabetes risk in real time
- User-friendly input widgets
- Displays prediction results instantly
- Visualizes prediction probabilities using charts
- Easy to deploy on Streamlit Community Cloud

---

## 🛠 Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Pickle (.pkl model)

---

## 📂 Project Structure

```
Diabetes-Risk-Prediction/
│
├── app.py                  # Main Streamlit application
├── diabetes_model.pkl      # Trained Machine Learning model
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
└── dataset.csv             # Dataset (optional)
```

---

## ⚙ Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/Diabetes-Risk-Prediction.git
```

Move into the project folder:

```bash
cd Diabetes-Risk-Prediction
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## ▶ Running the Application

Start the Streamlit application:

```bash
python -m streamlit run app.py
```

The application will open in your default web browser.

---

## 📊 Input Features

The application accepts health-related information such as:

- Glucose Level
- BMI
- Age
- Blood Pressure
- Insulin Level
- Skin Thickness
- Pregnancies
- Diabetes Pedigree Function

*(The available features depend on the model used.)*

---

## 🧠 Machine Learning Model

The prediction model was trained using Scikit-learn and saved as a `.pkl` file using Python's Pickle module.

The Streamlit application loads the trained model and performs predictions without retraining.

---

## 📈 Output

The application displays:

- Diabetes Risk Prediction
- Probability Chart
- User-friendly success/warning messages

---

## 🌐 Deployment

This application can be deployed easily using:

- Streamlit Community Cloud
- GitHub

---

## 📷 Screenshots

You can add screenshots of your application here.

Example:

```
screenshots/homepage.png
screenshots/prediction.png
```

---

## 📚 Future Improvements

- Support additional Machine Learning algorithms
- Improve prediction accuracy
- User authentication
- Prediction history
- Dark mode
- Download prediction reports

---

## 👨‍💻 Author

**Taurai Mvundusi**

B.Tech Computer Science & Engineering

---

## 📄 License

This project is intended for educational and learning purposes.
