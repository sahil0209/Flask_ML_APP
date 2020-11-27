from flask import Flask,render_template,url_for, redirect
import flask_wtf.csrf 
import pickle
from forms import IrisForm
import numpy as np
import os

app = Flask('__name__')
app.config['SECRET_KEY'] = os.urandom(16)

model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return redirect(url_for('predict'))



@app.route('/predict',methods = ['GET','POST'])
def predict():
    form = IrisForm()
    if form.validate_on_submit():
        tempArray = [form.sepal_length.data,form.sepal_width.data,form.petal_length.data,form.sepal_length.data]
        final_features = [np.array(tempArray)]
        predict = model.predict(final_features)
        if predict[0]==0:
            a = "setosa"
        elif predict[0]==1:
            a = "versicolor"
        elif predict[0]==2:
            a =  "virginica"
        return render_template('species_display.html',a = a)
    return render_template('pred_iris.html',form = form)



if __name__ == "__main__":
    app.run(debug=True)