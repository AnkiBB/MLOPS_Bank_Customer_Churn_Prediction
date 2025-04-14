from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import HTMLResponse, RedirectResponse
from uvicorn import run as app_run

from typing import Optional

# Importing constants and pipeline modules from the project
from src.constants import APP_HOST, APP_PORT
from src.pipline.prediction_pipeline import BankData, BankDataClassifier
from src.pipline.training_pipeline import TrainPipeline

# Initialize FastAPI application
app = FastAPI()

# Mount the 'static' directory for serving static files (like CSS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up Jinja2 template engine for rendering HTML templates
templates = Jinja2Templates(directory='templates')

# Allow all origins for Cross-Origin Resource Sharing (CORS)
origins = ["*"]

# Configure middleware to handle CORS, allowing requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DataForm:
    """
    DataForm class to handle and process incoming form data.
    This class defines the Bank-related attributes expected from the form.
    """
    def __init__(self, request: Request):
        self.request: Request = request
        self.credit_score: Optional[int] = None
        self.France: Optional[int] = None
        self.Germany: Optional[int] = None
        self.Spain: Optional[float] = None
        self.gender: Optional[int] = None
        self.age: Optional[float] = None
        self.tenure: Optional[int] = None
        self.balance: Optional[int] = None
        self.products_number: Optional[int] = None
        self.credit_card: Optional[int] = None
        self.active_member: Optional[int] = None
        self.estimated_salary: Optional[int] = None
                

    async def get_bank_data(self):
        """
        Method to retrieve and assign form data to class attributes.
        This method is asynchronous to handle form data fetching without blocking.
        """
        form = await self.request.form()
        self.credit_score = form.get("credit_score")
        self.France = form.get("France")
        self.Germany = form.get("Germany")
        self.Spain = form.get("Spain")
        self.gender = form.get("gender")
        self.age = form.get("age")
        self.tenure = form.get("tenure")
        self.balane = form.get("balance")
        self.products_number = form.get("products_number")
        self.credit_card = form.get("credit_card")
        self.active_member = form.get("active_member")
        self.estimated_salary = form.get("estimated_salary")

# Route to render the main page with the form
@app.get("/", tags=["authentication"])
async def index(request: Request):
    """
    Renders the main HTML form page for bank data input.
    """
    return templates.TemplateResponse(
            "bankdata.html",{"request": request, "context": "Rendering"})

# Route to trigger the model training process
@app.get("/train")
async def trainRouteClient():
    """
    Endpoint to initiate the model training pipeline.
    """
    try:
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()
        return Response("Training successful!!!")

    except Exception as e:
        return Response(f"Error Occurred! {e}")

# Route to handle form submission and make predictions
@app.post("/")
async def predictRouteClient(request: Request):
    """
    Endpoint to receive form data, process it, and make a prediction.
    """
    try:
        form = DataForm(request)
        await form.get_bank_data()
        
        bank_data = BankData(
                                credit_score = form.credit_score,
                                France = form.France,
                                Germany = form.Germany,
                                Spain = form.Spain,
                                gender = form.gender,
                                age = form.age,
                                tenure = form.tenure,
                                balance = form.balance,
                                products_number = form.products_number,
                                credit_card = form.credit_card,
                                active_member= form.active_member,
                                estimated_salary=form.estimated_salary
                                                                )

        # Convert form data into a DataFrame for the model
        bank_df = bank_data.get_bank_input_data_frame()

        # Initialize the prediction pipeline
        model_predictor = BankDataClassifier()

        # Make a prediction and retrieve the result
        value = model_predictor.predict(dataframe=bank_df)[0]

        # Interpret the prediction result as 'Churn-Yes' or 'Churn-No'
        status = "Churn-Yes" if value == 1 else "Churn-No"

        # Render the same HTML page with the prediction result
        return templates.TemplateResponse(
            "bankdata.html",
            {"request": request, "context": status},
        )
        
    except Exception as e:
        return {"status": False, "error": f"{e}"}

# Main entry point to start the FastAPI server
if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)