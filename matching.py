# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.utils import np_utils
import json



# Importing the dataset
dataset = pd.read_csv('/Users/aouni/Desktop/SoTuTech/MatchingSkills/matching.csv')
X = dataset.iloc[:, 0:13].values
y = dataset.iloc[:, 13].values


"""
# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]
"""

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
onehotencoder = OneHotEncoder(categorical_features = [0])
#y = onehotencoder.fit_transform(y).toarray()
y = np_utils.to_categorical(y)



# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

"""
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
"""






import keras
from keras.models import Sequential
from keras.layers import Dense
from flask import Flask
from flask import jsonify
from flask import request
from flask import Response


app = Flask(__name__)




@app.route('/noob', methods=['GET'])
def hello_world():
    return jsonify({'message' :'sdfqsd'})




classifier = Sequential()

classifier.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu', input_dim = 13))

classifier.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu'))


classifier.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu'))



classifier.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu'))




classifier.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'softmax'))

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

classifier.fit(X_train, y_train, batch_size = 3, epochs = 100)

#------------



y_pred = classifier.predict(X_test)

#Mobile Developer
new_prediction = classifier.predict(np.array([[70, 60, 70, 80, 70, 45, 30, 23, 43, 20, 24, 34, 22]]))

#Web Developer
new_prediction = classifier.predict(np.array([[28, 60, 40, 30, 45, 73, 80, 65, 0, 0, 30, 40, 66]]))




@app.route('/test/<uuid>', methods=['GET', 'POST'])
def add_message(uuid):
    content = request.get_json()
    print("test ")
    datafromjson =  []
    
    try: 
        datafromjson.append(content["Java"])
    except Exception as err: 
        datafromjson.append(0)
    try: 
        datafromjson.append(content["Python"])
    except Exception as err: 
        datafromjson.append(0)
    try: 
        datafromjson.append(content["Swift"])
    except Exception as err: 
        datafromjson.append(0)
    try: 
        datafromjson.append(content["IOS"])
    except Exception as err: 
        datafromjson.append(0)
    try: 
        datafromjson.append(content["Android"])
    except Exception as err: 
        datafromjson.append(0)
    try: 
        datafromjson.append(content["AngularJS"])
    except Exception as err: 
        datafromjson.append(0)
    try: 
        datafromjson.append(content["HTML"])
    except Exception as err: 
        datafromjson.append(0)
    try: 
        datafromjson.append(content["CSS"])
    except Exception as err: 
        datafromjson.append(0)
    try: 
        datafromjson.append(content["deep learning"])
    except Exception as err: 
        datafromjson.append(0)
    try: 
        datafromjson.append(content["Machine Learning"])
    except Exception as err: 
        datafromjson.append(0)
    try: 
        datafromjson.append(content["Spark"])
    except Exception as err: 
        datafromjson.append(0)
    try: 
        datafromjson.append(content["Hadoop"])
    except Exception as err: 
        datafromjson.append(0)
    try: 
        datafromjson.append(content["Php"])
    except Exception as err: 
        datafromjson.append(0)
    
    
    
    
    
    
    
    
    
    
    
    
    print(np.reshape(datafromjson ,(-1,13)))
    
   
    
    
    new_prediction = classifier.predict(np.reshape(datafromjson ,(-1,13)))
    
    new_prediction = new_prediction*100
    new_prediction_dict = [{"Data Analyst":item[0], "Data Scientist":item[1], "Machine Learning":item[2], "Mobile Developer":item[3], "Web Developer":item[4]} for item in new_prediction]
    
    print(type(new_prediction.tolist()))
    print(type(new_prediction_dict[0]))
    print(type(new_prediction_dict))
    
    #dictionary = sorted( ((v,k) for k,v in new_prediction_dict[0].items()), reverse=True)

    
    #return Response(json.dumps(new_prediction_dict[0]),  mimetype='application/json')
    return jsonify({'preds' : new_prediction.tolist()})

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
if __name__ == "__main__":
    app.run(debug=True)
#cm = confusion_matrix(y_test, y_pred)
    
    
    
    
new_prediction_dict = [{"Data Analyst":item[0], "Data Scientist":item[1], "Machine Learning":item[2], "Mobile Developer":item[3], "Web Developer":item[4]} for item in new_prediction]
type(new_prediction.tolist())
type(new_prediction_dict)





