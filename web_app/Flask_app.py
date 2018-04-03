from flask import Flask, render_template, request
from werkzeug import secure_filename
from wtforms import Form, TextAreaField, validators
import pickle
import sqlite3
import os
import numpy as np
from sklearn.externals import joblib
from sqlalchemy import create_engine
import pandas as pd
app = Flask(__name__)

######## Preparing the Classifier
cur_dir = os.path.dirname(os.path.abspath(__file__))
clf = joblib.load(os.path.join(cur_dir,'pkl_objects','titanic_ranf.pkl'))

#db = os.path.join(cur_dir, 'train.sqlite')

def classify(document):   
    X = pd.read_csv(document)[["age","fare","Second","Third","male","ischild","family_size"]].values  # document is read back as dataframe
    y = clf.predict_proba(X)[0][1]
    
    return "Probability of Survival is ", round(100*y,0),"%"

#def sqlite_entry(path, document):
#    conn=sqlite3.connect(path)    
#    df = pd.read_csv(document)
#    df.to_sql("table_sql", conn, if_exists='append', index=False)
#    conn.commit()
#    conn.close()
    
                  
@app.route('/')
def index():
    return render_template('reviewform.html')
    
@app.route('/upload',methods=['POST'])
def results():
    if request.method == 'POST':
        review=request.files['inputFile']
        pred_y=classify(review)
        #sqlite_entry(db,review)
        return render_template('results.html',content=review.filename,prediction=pred_y)


# @app.route('/uploader', methods = ['GET', 'POST'])
# def uploade_file():
#     if request.method == 'POST':
#         f = request.files['file']
#         f.save(secure_filename(f.filename))
#         return 'file uploaded successfully'

if __name__ == '__main__':
    app.run(debug = True)