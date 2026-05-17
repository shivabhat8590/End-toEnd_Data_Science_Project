Student Performance Prediction System

A complete end-to-end Machine Learning project that predicts student exam performance using various academic and lifestyle factors. The project includes data preprocessing, model training, FastAPI backend development, and a web-based prediction interface.

Features
Student exam score prediction
Machine Learning model using Random Forest Regressor
Data preprocessing and feature encoding
FastAPI backend integration
Interactive HTML frontend
Real-time predictions
REST API support
Deployment ready
Tech Stack
Technology	Usage
Python	Programming Language
Pandas	Data Processing
NumPy	Numerical Operations
Scikit-learn	Machine Learning
FastAPI	Backend Framework
HTML/CSS	Frontend
Joblib	Model Saving
Uvicorn	ASGI Server
Dataset

Dataset used:

Student Performance Factors Dataset

Features include:

Hours Studied
Attendance
Sleep Hours
Previous Scores
Motivation Level
Internet Access
Family Income
Teacher Quality
Physical Activity
And more...

Project Structure
student-performance-project/
│
├── app/
│   └── main.py
│
├── templates/
│   └── index.html
│
├── models/
│   └── student_model.pkl
│
├── data/
│   └── StudentPerformanceFactors.csv
│
├── train_model.py
├── requirements.txt
└── README.md

Install Dependencies
pip install -r requirements.txt

Train Model
Run:
python train_model.py
This will:
preprocess the dataset
train the model
save the trained model
Run FastAPI Server
uvicorn app.main:app --reload

<img width="1357" height="339" alt="Image" src="https://github.com/user-attachments/assets/8e776bde-53bd-4e59-a8c1-168ecaa03dca" />


Open browser:

http://127.0.0.1:8000
API Endpoint
Predict Student Score
POST /predict

<img width="1884" height="1011" alt="Image" src="https://github.com/user-attachments/assets/d9657ff3-f5d0-44a1-a79d-6d483707d159" />

Example Input:

{
  "Hours_Studied": 5,
  "Attendance": 85,
  "Parental_Involvement": 2,
  "Access_to_Resources": 2,
  "Extracurricular_Activities": 1,
  "Sleep_Hours": 7,
  "Previous_Scores": 75,
  "Motivation_Level": 2,
  "Internet_Access": 1,
  "Tutoring_Sessions": 2,
  "Family_Income": 1,
  "Teacher_Quality": 2,
  "School_Type": 1,
  "Peer_Influence": 2,
  "Physical_Activity": 4,
  "Learning_Disabilities": 0,
  "Parental_Education_Level": 2,
  "Distance_from_Home": 1,
  "Gender": 1
}

Example Output:

{
  "Predicted_Exam_Score": 78.52
}

Machine Learning Workflow
Data Collection
       ↓
Data Cleaning
       ↓
Feature Encoding
       ↓
Train-Test Split
       ↓
Model Training
       ↓
Prediction
       ↓
Web Deployment
