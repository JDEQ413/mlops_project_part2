# Machine Learning Operations : MLOps

The first part of the project can be found at the following repository: [mlops_project](https://github.com/JDEQ413/mlops_project).

Tehs second, and final part, of this project can be found at the following repository: [mlops_project_part2](https://github.com/JDEQ413/mlops_project_part2).

As we advance on this readme.md file, particular deliverables will be specified.
 
# Final Project - Part 1

 - Project advance 1 for the subject ```Problems resolution with technology application```.

## Module 1 - Key concepts of ML systems

### Base-line definition and project scope

**Base-line**

This the baseline of the project.

* A Random Forest model is implemented, which loads a dataset and applies StandardScaler transformation to all numerical fields, also applies 70-30 partition for train and test sets respectively, achieving a final accuracy around 85%, and cross validation accuracy around 82%.

**Scope**

* The expected scope of this project is implementation of techniques and good practices to achieve deployment of the full functionality of this code through REST API.

**Situation**

* This is an excersice taken from kaggle to work with, in which the objective is to try to determine the median value of owner-occupied homes (MEDV, dependent variable), given a serie of independent variables like structural, neighborhood, accessibility and air pollution data in Boston around 70's.

* To know more about the dataset you can see directly kaggle [link](https://www.kaggle.com/datasets/fedesoriano/the-boston-houseprice-data).

### Notebook

* **Base-line** code may be found at [base_line.ipynb](https://github.com/JDEQ413/mlops_project/blob/main/docs/base_line.ipynb).
  * It was tacken from several notebooks developed by other users in kaggle platform.
    * [MAGANTI IT](https://www.kaggle.com/code/magantiit/linearregression)
    * [SADIK AL JARIF](https://www.kaggle.com/code/sadikaljarif/boston-housing-price-prediction)
    * [MARCIN RUTECKI](https://www.kaggle.com/code/marcinrutecki/regression-models-evaluation-metrics)
    * [UNMOVED](https://www.kaggle.com/code/unmoved/regress-boston-house-prices)
 
* **Sectioned notebook** contains the code that was identified in every recommended section.
  * It may be found at [house-price.ipynb](https://github.com/JDEQ413/mlops_project/blob/main/docs/house-price.ipynb)
 
* **.py file** is the result of first steps to acomplish refactorization, and is the base to separate sections in folders.
  * It may be found at [house-price.py](https://github.com/JDEQ413/mlops_project/blob/main/docs/house-price.py)
 

## Module 2 - Basic concepts and tools for software development

### Virtual environments

* Change the directory to "mlops_project" folder.
* Create a virtual environment with Python 3+:
    ```bash
    python3 -m venv venv
    ```
    * Windows cmd:
      ```bash
      py -3.10 -m venv venv310
      ```
 
* Activate the virtual environment
    ```bash
    source venv/bin/activate
    ```
    * Windows cmd:
      ```bash
      venv310\scripts\activate.bat
      ```

* Install the other libraries. Run the following command to install the libraries/packages.
    ```bash
    pip install -r requirements-310.txt
    ```
    * Requirements file may be consulted [here](https://github.com/JDEQ413/mlops_project/blob/main/requirements-310.txt).

* Don´t forget to select the new environment at kernel when using VSCode.

### Continuous use of GitHub

* GitHub was used continuosly during the development of this project, increasing graduately the content of the repository.
  * [mlops_project/commits/main](https://github.com/JDEQ413/mlops_project/commits/main)

### Unit tests

* Unit tests are developed to acomplish basic validations on specific functionalities of the project.
  * Library: pytest
    * Install, move to project directory: ```pip install pytest``` or ```pip install -r requirements-310.txt```
  * Code folder: [tests](https://github.com/JDEQ413/mlops_project/tree/main/tests)
  * Run tests, terminal or console:
    * Individual test: ```pytest tests/unit_tests.py::test_csv_file_existence -v```
    * Multiple tests: ```pytest tests/unit_tests.py -v```

### Pre-Commits

* Pre-commits were implemented on the project from [house-price.py](https://github.com/JDEQ413/mlops_project/blob/main/docs/house-price.py) file onward, since this was the first element pre-commits are able to evaluate.
* This are the repos implemented for linting and formatting:
  * isort
  * autoflake
  * autopep8
  * flake8
    * _Every one has its own hooks to represent specific checks on the code._
    * _The corresponding libraries are contained inside requirements-310.txt file. They may be installed but nothing will happen if .yaml file does not exist or is empty, or pre-commit has not been initialized on the project for the first time._
* Configuration file can be found at [.pre-commit-config.yaml](https://github.com/JDEQ413/mlops_project/blob/main/.pre-commit-config.yaml)

**Setup pre-commits**

* Open your terminal or command prompt, navigate to the root directory of your project
* Pre-commit needs to be installed, at this time it already was by que _requirements_ file.
  * If you are installing it for the firs time use:
    ```bash
    pip install pre-commit
    ```
* After creating the .pre-commit-config.yaml file, initialize pre-commit for the project:
  ```bash
  pre-commit install
  ```
* With the pre-commit hooks installed, you can now make changes to your Python code. When you're ready to commit your changes, run the following command to trigger the pre-commit checks:
  ```bash
  git commit -m "add pre-commit file"
  ```
* If every check "passed", then you are ready to upload your changes to the repository at GitHub.
  ```bash
  git push
  ```

## Module 3 - Development of ML models

### Refactorization

* Folders with refactorized code is found in the following directory structure of this project ([repository](https://github.com/JDEQ413/mlops_project)).
  * api
  * docs
  * [mlops_project](https://github.com/JDEQ413/mlops_project/tree/main/mlops_project)
    * [data](https://github.com/JDEQ413/mlops_project/tree/main/mlops_project/data)
    * [load](https://github.com/JDEQ413/mlops_project/tree/main/mlops_project/load)
    * [models](https://github.com/JDEQ413/mlops_project/tree/main/mlops_project/models)
    * [preprocess](https://github.com/JDEQ413/mlops_project/tree/main/mlops_project/preprocess)
    * [predictor](https://github.com/JDEQ413/mlops_project/tree/main/mlops_project/predictor)
    * [train](https://github.com/JDEQ413/mlops_project/tree/main/mlops_project/train)
  * tests 
 * In every one of the folders with a link attached there is code and files extracted from the [house-price.py](https://github.com/JDEQ413/mlops_project/blob/main/docs/house-price.py) file, and functionalities, with sections identified and linted and formatted with pre-commits hooks.
 * All the code separated in modules and classes can be executed in the terminal
   * Change the directory to "mlops_project" folder
   * If not active, activate virtual environment
     ```bash
     source venv/bin/activate
     ```
     * Windows cmd:
       ```bash
       venv310\scripts\activate.bat
       ```
   * Run the following:
     ```bash
     python mlops_project\mlops_project.py
     ```

### REST API

* The implementation of REST API was through the application of fastapi, pydantic and uvicorn libraries and the corresponding code can be found in the [api](https://github.com/JDEQ413/mlops_project/tree/main/api) folder of this project ([repository](https://github.com/JDEQ413/mlops_project)).
  * All libraries are included in _requirements_ file, and are already installed by this point.
* The endpoints generated to run the project as an API are:
  * healthcheck
  * train_new_model
  * predictor
* Run the following command to start House-pricing API locally.
  ```bash
  uvicorn api.main:app --reload
  ```
* You can check the endpoints as follows:
  * Access ```http://127.0.0.1:8000/```, you will see a message like this ```"HousePricing Regressor is ready to go!"```.
  * Access ```http://127.0.0.1:8000/docs```, the browser will display something like this:

**Test predictions**

* **1.** To test predictions using the ```predict``` endpoint, give the folloing values.
  ```
  {
    "crim": 0.06905,
    "zn": 0.0,
    "indus": 2.18,
    "chas": 0,
    "nox": 0.458,
    "rm": 7.147,
    "age": 54.2,
    "dis": 6.0622,
    "rad": 3,
    "tax": 222.0,
    "ptratio": 18.7,
    "b": 396.90,
    "lstat": 5.33
  }
  ```

* **2.** For a second test predictions use the folloing values.
  ```
  {
    "crim": 0.02729,
    "zn": 0.0,
    "indus": 7.07,
    "chas": 0,
    "nox": 0.469,
    "rm": 7.185,
    "age": 61.1,
    "dis": 4.9671,
    "rad": 2,
    "tax": 242.0,
    "ptratio": 17.8,
    "b": 392.83,
    "lstat": 4.03
  }
  ```


# Final Project - Part 2

 - Project advance 2 for the subject ```Problems resolution with technology application```.

## Module 4 - Deploying ML Models

### Logging

**Configuration**

* To implement logging in every module of the project, lets begin with the actual structure of this ginal module.
  * api (Fron-end Fast API)
  * server (Back-end Fast API)
* Since each folder will be a REST API with specific independent role, a 'utilities' module will be placed in each to implement logging, where ```custom_logging.py``` contains que class ```CustomLogging()```.
  
  ![image](https://github.com/JDEQ413/mlops_project_part2/assets/139833546/99de7f3b-ea0f-4c5a-8706-6595992cdf25)  ![image](https://github.com/JDEQ413/mlops_project_part2/assets/139833546/19452574-82ff-4f1c-b4a9-0ffd58ba2458)

* This class has a method to create a logger, wher it configures 'Log file name', 'Format', 'File_Handler', and/or 'Stream_Handler'. Returning the configured logger itself.

**Usage**

* In every module of the project it is necessary to implement logging. Using 'utilities' module we can make it easy.
* First, import CustomLogging() class, instantiate it, and then create it specifying the name for the module log file where you are calling it, and define if its necessary to make it streamer to be aware of whats going on while executing the project.
  * ![image](https://github.com/JDEQ413/mlops_project_part2/assets/139833546/f446625a-cea8-454d-bcd9-4465cc9a6d8b)
* As a result you will get a log file for the module where it was applyed, and the amount of entries you get depends on the nature of the module calling the log.
  * ![image](https://github.com/JDEQ413/mlops_project_part2/assets/139833546/70b08fde-e9e4-4130-913e-c38f95a073fe)


### Docker

**Build image**

* Ensure you are in the root folder.
* Run the following code to build the image:


### Docker-compose

**Start**

* Before giving command lines, you need to know that a basic structure to fullfill the implementation of docker-compose, as it was said before, separate roles are needed to be played by REST API projects.
* ![image](https://github.com/JDEQ413/mlops_project_part2/assets/139833546/a5776530-0566-4d55-9798-8af49bdab749)
  * api plays the role of front-end Fast API
  * server plays the role of back-end Fast API
* The files you need to have for each role are:
  * **Dockerfile**. Specifications for docker-compose to create image of the role.
  * **requirements.txt**. Each role have different minimum needs of libraries to run, here is where you specify them.
  * **main.py**. Each role should have endpoints to fulfill what is required of it to achieve, and its behavior is defined here, and so the interaction between roles.
* Also, a ```docker-compose.yml``` file is required to orchestrate images and container defined through the three files listed before.

**Network**

* Begining with instructions, the first thing needed is to create a network, which will be the chanel of communication between roles. So go to the project folder, activate virtual envorinment, and run the following command.
  ```
  docker network create AIservice
  ```
**Run**

* Ensure you are in the directory where the docker-compose.yml file is located. In this case is root, so you can run the following to start the Server and Frontend APIs
  ```
  docker-compose -f itesm_mlops_project/docker-compose.yml up --build
  ```
* If everything worked fine:
  * Images were created
  * Container was created
  * Images were assigned to container
  * Container is running with images running
  * You will see something like this.
    ![image](https://github.com/JDEQ413/mlops_project_part2/assets/139833546/e922a419-1b53-42cb-a74b-ba7b3c0eb2f6)
  * And docker will have this.
    ![image](https://github.com/JDEQ413/mlops_project_part2/assets/139833546/2119a8f2-dd3a-4e2d-a50e-88c4bca5a9d4)


**Localhost**

* If you check both, server and frontend, localhost at the ports defined to expose, you will have this.
  * Server (back-end)
    ![image](https://github.com/JDEQ413/mlops_project_part2/assets/139833546/15b04e6d-466d-46ba-b13d-50b9c99eb550)
  * Api (front-end)
    ![image](https://github.com/JDEQ413/mlops_project_part2/assets/139833546/1a14db5c-a446-433a-852c-4e6aa0585d0c)

**Endpoints**

* To check endpoints just add ```/docs``` to address bar after port, you will have this.
  * Server (back-end)
    
    ![image](https://github.com/JDEQ413/mlops_project_part2/assets/139833546/7037c93c-7a51-4a93-a35a-8f441139cf9d)

  * Api (front-end)
    
    ![image](https://github.com/JDEQ413/mlops_project_part2/assets/139833546/b115592c-1716-413d-874f-ee5511f26e3d)

**Prediction**

* You can try it out every endpoint, one of them is ```predict```, and by giving the folloing values you will get a prediction.
  ```
  {
    "crim": 0.06905,
    "zn": 0.0,
    "indus": 2.18,
    "chas": 0,
    "nox": 0.458,
    "rm": 7.147,
    "age": 54.2,
    "dis": 6.0622,
    "rad": 3,
    "tax": 222.0,
    "ptratio": 18.7,
    "b": 396.90,
    "lstat": 5.33
  }
  ```
* The result may be something like this.
  
  ![image](https://github.com/JDEQ413/mlops_project_part2/assets/139833546/d1c5fa74-f91f-4bec-80de-e8c9b6ee5ecd)

**Logs**

* If logging was correctly configured and implemented, you will see something like this in the command console thanks to streaming handler.
  
  ![image](https://github.com/JDEQ413/mlops_project_part2/assets/139833546/20c18b5d-c020-4b53-bc76-49b011b62869)

  * It means front-end (api) made a request of prediction
  * which is attended by the server (back-end)
  * server loads the model and it receives the request
  * makes the prediction, shows the result
  * and responds to front end with it
  * finally, front-end receives the result and says everything is ok (```200 OK```).
    
* If file handler was correctly configured, you can get log files from docker containers.
  * To know the names of the containers-images you can run ```docker ps -a```, and get something like this.
    ![image](https://github.com/JDEQ413/mlops_project_part2/assets/139833546/4ccfe6f5-346c-428e-bcd7-9ff13862eed3)
  * Now that you know the names, you can run the following code: ```docker cp mlops_project_part2-server-1:/server_main.log .``` and get a result like this.
    ![image](https://github.com/JDEQ413/mlops_project_part2/assets/139833546/65de926e-70a7-45dc-8272-ffdf5af89070)
  * And repeat it to obtain logs from api (front-end)
    ![image](https://github.com/JDEQ413/mlops_project_part2/assets/139833546/6d16d4b5-ef2d-43a1-98fb-8c1178941ffb)

 * When opening log files you will se something like this
   * Server (back-end)
     ![image](https://github.com/JDEQ413/mlops_project_part2/assets/139833546/92c83ff9-aa85-43dd-ba20-204fb8831b53)
   * Api (front-end)
     ![image](https://github.com/JDEQ413/mlops_project_part2/assets/139833546/4905393f-b1f2-46c6-8c3e-7a77a5f02e79)

**Delete**

* To give an end to this project, we stop container and delete it, running the following commands in that order.
  * ```docker-compose -f docker-compose.yml stop```
  * ``` docker-compose -f docker-compose.yml rm```
  ![image](https://github.com/JDEQ413/mlops_project_part2/assets/139833546/31715ef5-29e3-4afc-8d37-9bfbb0c47c97)
  ![image](https://github.com/JDEQ413/mlops_project_part2/assets/139833546/953ba3a2-5e30-4e9c-bfe1-d86863ca2d0d)
