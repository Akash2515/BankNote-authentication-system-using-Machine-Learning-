from flask import Flask,request,render_template

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
    return render_template("index.html", action="Add")

@app.route('/predict',methods=['Get'])
def index():
    
    if 'submit' in request.form:  
        Variance=int(request.form["variance"])
        Skewness=int(request.form["skewness"])
        Curtosis=int(request.form["curtosis"])
        Entropy=int(request.form["entropy"])
        answer=model.predict([[Variance,Skewness,Curtosis,Entropy]])
        ans="the answer is " +str(answer)
        print(ans)
        return render_template("index.html", action="Add" ,result=ans)
        
@app.route('/predict_file',methods=['POST'])
def file_predict():
    """Let's Authenticate the Banks Note 
    This is using docstrings for specifications.
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
      
    responses:
        200:
            description: The output values
        
    """
    
    dt=pd.read_csv(request.files.get("file"))
    ans=model.predict(dt)
    
    return "the output is" + str(list(ans))
    
    
   





if __name__ == '__main__':
    app.run(port='8000')           