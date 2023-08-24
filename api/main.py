import requests
from fastapi import Body, FastAPI
from utilities.custom_loging import CustomLogging

# Initializes logger
logger = CustomLogging
logger = logger.Create_Logger(logger_name='api.log', flag_streamer=True)

app = FastAPI()


# ML model prediction function using the prediction API request
def predict():

    url3 = "http://server.docker:8000/"

    logger.debug("Front-end prediction requested.")
    response = requests.request("GET", url3)
    response = response.text
    logger.debug("Front-end prediction obtained.")

    return response


def predict_housepricing(input):
    url3 = "http://server.docker:8000/predict_houseprice"

    logger.debug("Front-end houseprice prediction requested.")
    response = requests.post(url3, json=input)
    response = response.text
    logger.debug("Front-end houseprice prediction obtained.")

    return response


@app.get("/")
def read_root():
    logger.info("House-Pricing Model Front-end is ready to go!")
    return "House-Pricing Model Front-end is ready to go!"


@app.get("/healthcheck")
async def v1_healhcheck():
    url3 = "http://server.docker:8000/"

    response = requests.request("GET", url3)
    response = response.text
    logger.info(f"Healthcheck in progress... : {response}")

    return response


@app.post('/predict')
def predictor(payload: dict = Body(...)):
    logger.info("Prediction requested.")
    response = predict_housepricing(payload)
    return {"response": response}
