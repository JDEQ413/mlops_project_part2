

from api.models.models import HousePricing
from fastapi import FastAPI
from predictor.api_predict import ModelAPIPredictor
from starlette.responses import JSONResponse
from utilities.custom_loging import CustomLogging

# Initializes logger
logger = CustomLogging()
logger = logger.Create_Logger(logger_name="main_fast_api.log", flag_streamer=True)

model_path = "./models/"

app = FastAPI()

"""
PARAMETER VALUES
Values are required after de endpoint.
"""


@app.get('/', status_code=200)
async def healthcheck():
    logger.info("Healthcheck requested.")
    return 'HousePricing Regressor is ready to go!'


@app.post('/predict')
def predictor(housepricing_features: HousePricing):
    logger.info("Prediction requested.")
    predictor = ModelAPIPredictor(model_path + "random_forest_output.pkl")
    X = [
        housepricing_features.crim,
        housepricing_features.zn,
        housepricing_features.indus,
        housepricing_features.chas,
        housepricing_features.nox,
        housepricing_features.rm,
        housepricing_features.age,
        housepricing_features.dis,
        housepricing_features.rad,
        housepricing_features.tax,
        housepricing_features.ptratio,
        housepricing_features.b,
        housepricing_features.lstat
    ]
    prediction = predictor.predict([X])
    logger.debug("Prediction was made: " + str(prediction))
    return JSONResponse(f"Resultado predicción: {prediction}")
