# Spam-Text-Detection

The aim of this project was to build a machine learning model that can detect whether a given message or spam or not.<br /><br />
In this repository i have included the data set which i have taken from kaggle<br /><br />

## Models 

I build three differnet machine learning models and checked which model gave better results. For XGboost i gave good f1 score for both classes and it also have very good accuracy.According to me in classification models its better to ignore accuracy and focus more on f1 socre that to on minority class as usually there would be issue of imblaced data.

## Results 

#### Accuracy - 
![alt text](https://github.com/NikhilK-crypto/Spam-Text-Detection/blob/main/Images/accuracy.png)

#### Confusion Matrix
Logistic Regression<br /><br />
![alt text](https://github.com/NikhilK-crypto/Spam-Text-Detection/blob/main/Images/logistic_matrx.png)

Random Forest <br /><br />
![alt text](https://github.com/NikhilK-crypto/Spam-Text-Detection/blob/main/Images/Random_matrix.png)

XGBoost <br /><br />
![alt text](https://github.com/NikhilK-crypto/Spam-Text-Detection/blob/main/Images/XGBoost_matrix.png)

#### Classification Report

Logistic Regression<br /><br />
![alt text](https://github.com/NikhilK-crypto/Spam-Text-Detection/blob/main/Images/Logistic%20Regression.png)

Random Forest <br /><br />
![alt text](https://github.com/NikhilK-crypto/Spam-Text-Detection/blob/main/Images/randomforest.png)

XGBoost <br /><br />
![alt text](https://github.com/NikhilK-crypto/Spam-Text-Detection/blob/main/Images/XGboost.png)

## Front-end 
Devloped a UI using streamlit api, where user can interact with the model. <br /><br />
![alt text](https://github.com/NikhilK-crypto/Spam-Text-Detection/blob/main/Images/Screenshot%20(37).png)
<br /><br />just download app.py, models and transformation file and run it. Make sure you have necessary packages installed on your pc <br /><br />


## necessary packages to run the app.py 
streamlit<br />
joblib<br />
nltk<br />
re <br />


