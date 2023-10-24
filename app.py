import json
import pickle
import sklearn
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)
## Load the model
clfmodel=pickle.load(open('./artifacts/clfmodel.pkl','rb'))
scalar=pickle.load(open('./artifacts/scaler.pkl','rb'))


@app.route('/') #home page
def home():
    return render_template('home.html')#must create tempalte folder




# @app.route('/predict_api',methods=['POST']) #send request to app to get output
# def predict_api():
#     data=request.json['data']
#     print(data)
#     print(np.array(list(data.values())).reshape(1,-1)) #single data point 
#     new_data=scalar.transform(np.array(list(data.values())).reshape(1,-1)) #pass through scaling
#     output=regmodel.predict(new_data)
#     print(output[0])
#     return jsonify(output[0])



@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=scalar.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=clfmodel.predict(final_input)[0]
    return render_template("home.html",prediction_text="The Maternal Health Risk is {}".format(output))



if __name__=="__main__":
    app.run(debug=True)
   
     
