from urllib import request
from flask import Flask, render_template, session, url_for, redirect
from flask_wtf import FlaskForm
from wtforms import SubmitField, StringField
import joblib
import pandas as pd

app = Flask(__name__)
app.config['SECRET_KEY'] = 'asecretkey'

def return_prediction(model, text):   
    probs = pd.DataFrame(model.predict_proba([text]), columns=model.classes_)
    probs = probs.sort_values(by=0, axis=1, ascending=False)
    probs = probs.iloc[: , :5]
    probs = probs.T
    pr = (list(probs.index)) 

    for i in range(1,len(pr)):
        for j in range(len(pr[i]) - 3):
            if pr[i][j] == "h" and pr[i][j+1] == "t" and pr[i][j+2] == "t":
                pr[i] = pr[i][:j]
                break

    for i in range(len(pr[0])-3):
        if pr[0][i] == "h" and pr[0][i+1] == "t" and pr[0][i+2] == "t":
            pr.append(pr[0][i:])
            pr[0] = pr[0][:i]
            break

    return pr

model = joblib.load('moviefinder.joblib')

class PredictForm(FlaskForm):
    print('predictform')
    text = StringField("Plot Info")
    submit = SubmitField("Predict")
    home = SubmitField("Try again")

@app.route("/", methods=["GET", "POST"])
def index():
    form = PredictForm()

    if form.validate_on_submit():
        if form.submit.data:
            session['Plot'] = form.text.data
            return redirect(url_for("prediction"))

    return render_template('home.html', form=form)

@app.route('/prediction')
def prediction():
    if str(session['Plot']) == "":
        return redirect(url_for("index"))
    content = {}
    content['text'] = str(session['Plot'])
    results = return_prediction(model, content['text'])
    
    return render_template('prediction.html', r1=results[0], r2=results[1], r3=results[2], r4=results[3], r5=results[4], p_link=results[5])

if __name__ == '__main__':
    app.run()

