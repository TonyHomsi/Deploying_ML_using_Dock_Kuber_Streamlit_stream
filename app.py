import pandas as pd
import numpy as np
#import sklearn
import joblib
#from flask import Flask,render_template,request
#app=Flask(__name__)
import streamlit


"""
define a function to make predictions using the
trained model. We pass the five input parameters in the function and do
a bit of reshaping and data casting to ensure consistency for predictions.
"""

model=open("linear_regression_model.pkl","rb")
lr_model=joblib.load(model)


def lr_prediction(var_1,var_2,var_3,var_4,var_5):
	pred_arr=np.array([var_1,var_2,var_3,var_4,var_5])
	preds=pred_arr.reshape(1,-1)
	preds=preds.astype(int)
	model_prediction=lr_model.predict(preds)
	return model_prediction

def run():
	streamlit.title("Linear Regression Model")
	html_temp="""
	"""
	streamlit.markdown(html_temp)
	var_2=streamlit.text_input("Variable 2")
	var_3=streamlit.text_input("Variable 3")
	var_1=streamlit.text_input("Variable 1")
	var_4=streamlit.text_input("Variable 4")
	var_5=streamlit.text_input("Variable 5")
	prediction=""
	if streamlit.button("Predict"):
		prediction=lr_prediction(var_1,var_2,var_3,var_4,var_5)
		streamlit.success("The prediction by Model is {}".format(prediction))
    
if __name__=='__main__':
    run()


# @app.route('/')
# def home():
# 	return render_template('home.html')

# @app.route('/predict',methods=['GET','POST'])

# def predict():
# 	if request.method =='POST':
# 		print(request.form.get('var_1'))
# 		print(request.form.get('var_2'))
# 		print(request.form.get('var_3'))
# 		print(request.form.get('var_4'))
# 		print(request.form.get('var_5'))
# 		try:
# 			var_1=float(request.form['var_1'])
# 			var_2=float(request.form['var_2'])
# 			var_3=float(request.form['var_3'])
# 			var_4=float(request.form['var_4'])
# 			var_5=float(request.form['var_5'])
# 			pred_args=[var_1,var_2,var_3,var_4,var_5]
# 			pred_arr=np.array(pred_args)
# 			print(pred_arr)
# 			preds=pred_arr.reshape(1,-1)
# 			model=open("linear_regression_model.pkl","rb")
# 			lr_model=joblib.load(model)
# 			model_prediction=lr_model.predict(preds)			
# 			model_prediction=round(float(model_prediction),2)
# 		except ValueError:
# 			return "Please Enter valid values"
# 	return render_template('predict.html',prediction=model_prediction)
# if __name__=='__main__':
# 	app.run(host='0.0.0.0')


