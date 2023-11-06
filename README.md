# Sentiment-Analysis-From-Text-Feedback
## Online deployment
We have deployed our project on cloud using heroku. It can be accessed using the following link:

https://sentiment-analyzer-web-app.herokuapp.com/

## Running the project locally
#### Steps to be followed:
1. Download the zip file
2. Setup your virtual environment
3. Extract the zipped contents to the environment
4. Install the necessary libraries specified in ```requirements.txt```

      1. They can be installed using the ```pip install -r requirements.txt``` command in the terminal

5. Finally run the project using ```streamlit run predictor.py``` or ```python -m streamlit run D:\Sentiment-Analysis-From-Text-Feedback-main(Updated)\predictor.py``` in the terminal
6. A local web app will be deployed.
7. Enter your review in the text box and press ```CTRL+ENTER``` to get the sentiment prediction.

## Dataset
Link to the dataset used in the project:

https://www.kaggle.com/arnabpml/zomato-reviews-sentiment-analysis/data

## Tech stack
1. Python3
2. Streamlit
3. Heroku

## Files in the repository
1. ```main.py``` - This is the python file where we have trained our ML model.
1. ```predictor.py``` - This is the python file where we have made our predictor and necessary codes to deploy our model.
1. ```reviews.csv``` - The dataset we have used to train our model
1. ```requirements.txt``` - It contains all the necessary libraries required for the project
2. ```analysis.pkl``` - Stores a dictionary of words used in our training data
4. ```classifier model``` - Used to store our trained ML model to later use in prediction
5. ```background.jpg``` - Background image of our web app
6. ```setup.sh``` - Shell file that sets up the cloud environment on heroku
7. ```Procfile``` - Gives the necessary commandss to heroku which are required to deploy our app
