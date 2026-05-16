from fastapi import FastAPI, Form, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import joblib
import numpy as np

# Create FastAPI app
app = FastAPI()

# Templates folder
templates = Jinja2Templates(directory="templates")

# Load trained model
model = joblib.load("models/student_model.pkl")


# Home Route
@app.get("/", response_class=HTMLResponse)
def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )


# Prediction Route
@app.post("/predict", response_class=HTMLResponse)
def predict(
    request: Request,

    Hours_Studied: float = Form(...),
    Attendance: float = Form(...),
    Parental_Involvement: int = Form(...),
    Access_to_Resources: int = Form(...),
    Extracurricular_Activities: int = Form(...),
    Sleep_Hours: float = Form(...),
    Previous_Scores: float = Form(...),
    Motivation_Level: int = Form(...),
    Internet_Access: int = Form(...),
    Tutoring_Sessions: int = Form(...),
    Family_Income: int = Form(...),
    Teacher_Quality: int = Form(...),
    School_Type: int = Form(...),
    Peer_Influence: int = Form(...),
    Physical_Activity: float = Form(...),
    Learning_Disabilities: int = Form(...),
    Parental_Education_Level: int = Form(...),
    Distance_from_Home: int = Form(...),
    Gender: int = Form(...)
):

    # Create feature array
    features = np.array([[
        Hours_Studied,
        Attendance,
        Parental_Involvement,
        Access_to_Resources,
        Extracurricular_Activities,
        Sleep_Hours,
        Previous_Scores,
        Motivation_Level,
        Internet_Access,
        Tutoring_Sessions,
        Family_Income,
        Teacher_Quality,
        School_Type,
        Peer_Influence,
        Physical_Activity,
        Learning_Disabilities,
        Parental_Education_Level,
        Distance_from_Home,
        Gender
    ]])

    # Predict
    prediction = model.predict(features)

    predicted_score = round(float(prediction[0]), 2)

    # Return HTML page with prediction
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "prediction": predicted_score
        }
    )