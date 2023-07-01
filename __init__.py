import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from individual_company_stock import getHistoryData
import math
from flask import Flask, request, render_template, send_file
from datetime import date,timedelta
import matplotlib
matplotlib.use('agg')
app = Flask(__name__)

parent_dir="D:\\programs\\ads_flask\\logres_op\\"
model_dir = os.path.join(parent_dir, "model\\")
app.config['UPLOAD_FOLDER'] = os.path.join(parent_dir,"temp_img")

data_files = os.listdir(model_dir)
cdate=date.today()
stock_list=[]
for data_file in data_files:
    stock_list.append(os.path.splitext(data_file)[0])
val=0
def strdate(cdate,daydiff=0):
    '''returns datetime in string with the option to get a different date'''
    cdate=cdate-timedelta(days=daydiff)
    return(cdate.strftime("%d-%m-%Y"))
def get_his(stock,startdate,enddate):
    '''retrives history for a stock for a given time period'''
    temp=getHistoryData(stock,from_date=startdate,to_date=enddate)
    temp=temp.replace({',':''},regex=True)
    # print(temp) 
    return temp
def load_stock_model(stock_name):
    '''loads a particular model'''
    model_path = os.path.join(model_dir, stock_name + ".h5")
    model = load_model(model_path)
    return model
def make_prediction(stock_name):
    '''loads model,scrapes data, predicts 90 day future for a stock based on current day and returns the predictions'''
    model=load_stock_model(stock_name)
    # print(stock_name)
    input_data=get_his(stock_name,strdate(cdate,daydiff=180),strdate(cdate))
    # print(input_data)
    input_data=pd.DataFrame(pd.to_numeric(pd.Series(input_data["close "].tail(91))),columns=["close "])
    input_data=input_data.reset_index(drop=True)
    input_data["logret"]=np.log(input_data["close "]) - np.log(input_data["close "].shift(1))
    input_data=input_data.drop(0,axis=0)
    input_data.reset_index(drop=True,inplace=True)
    val=input_data["close "][89]
    predictions=model.predict(np.array(input_data["logret"][input_data["logret"].size-90:]).reshape(1,90,1),verbose=0)    
    t=[]
    for j in range(predictions.shape[0]):
        if(j==0):
            t.append(input_data["close "][0]*(math.e**predictions[j]))
        else:
            t.append((math.e**predictions[0][j])*t[j-1])
    predictions=pd.DataFrame(t).T
    return predictions#future 90 day predictions from today for stock [stock_name]
def create_plot(predictions, i):
    '''generates a plot which can be used in a webpage based on predictions.
    Use in conjunction with outputs from make_prediction'''
    fig = plt.figure(figsize=(15, 5))
    plt.scatter(np.arange(90), predictions, c="g", figure=fig)
    plt.plot(np.arange(90), predictions, figure=fig)
    plt.xticks(ticks=np.arange(0, 90, 5), labels=np.arange(1, 91, 5), figure=fig)
    plt.xlabel("created on " + str(date.today()), figure=fig)
    plt.ylabel("predicted value", figure=fig)
    plt.title("future 90-Day predictions for " + str(i), figure=fig)
    return fig  # Return the figure object

@app.route('/')
def index():
    return render_template('index1.html')
@app.route('/predict/')
def predict():
    stock_name=request.args.get('stock_name')  # Get the value of the "stock_name" query parameter
    if stock_name in stock_list:
        try:
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'],"image.png"))
        except:
            print("no file")
        p=make_prediction(stock_name)
        fig = create_plot(p, stock_name)
        # Save the figure to a file
        image_path = os.path.join(parent_dir, "temp_img", "image.png")
        fig.savefig(os.path.join(app.config['UPLOAD_FOLDER'],"image.png"))
        plt.close(fig)  # Close the figure to release memory
        return render_template("index.html", stock_name=stock_name,value=val,data=pd.DataFrame(np.array(p).reshape((10,9))).to_html())
    else:
        return "Model not found for the specified stock."
if __name__ == '__main__':
    app.run(debug=True)
   