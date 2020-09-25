from flask import Flask,request

import pandas as pd 
import numpy as np
import pickle
from flasgger import Swagger

app=Flask(__name__)
Swagger(app)
pickle_out=open("model.pk1","rb")
model=pickle.load(pickle_out)

@app.route("/")
def hello():
    return "hello all"

@app.route('/predict',methods=['Get'])
def note_predict():
    '''Let's Authenticate the Banks Note 
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: variance
        in: query
        type: number
        required: true
      - name: skewness
        in: query
        type: number
        required: true
      - name: curtosis
        in: query
        type: number
        required: true
      - name: entropy
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values
      '''  
    
    variance=request.args.get("variance")
    skewness=request.args.get("skewness")
    curtosis=request.args.get("curtosis")
    entropy=request.args.get("entropy")
    answer=model.predict([[variance,skewness,curtosis,entropy]])
    return "the answer is "+str(answer)
    
@app.route('/predict_file',methods=['Post'])
def file_predict():
    '''Let's Authenticate the Banks Note 
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: file
        in: formData
        type: file
        required: true
    responses:
        200:
            description:The output values
    '''
    
    dt=pd.read_csv(request.files.get("file"))
    ans=model.predict(dt)
    
    return "the output is" + str(list(ans))
    
    
   





if __name__ == '__main__':
    app.run(port='8000')           