# This is basically the heart of my flask 
from flask import Flask, render_template, request, redirect, url_for
from scipy import sparse
import pandas as pd
import numpy as np
import pickle
import operator
import warnings
import nltk
nltk.download('punkt', download_dir='/app/nltk_data/')
warnings.filterwarnings("ignore")


app = Flask(__name__)

with open('pickles/UU_Recommendation_model.pkl','rb') as fp:
	model = pickle.load(fp)
# Store unique Username in list
users = list(model.index)

@app.route('/')
def home():
	return render_template('index.html', all_user=users)
	#return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
	userName = request.form.get('userName')
	Input = str(userName)
	print("User Name (Input): ", Input)
	pred_Recommendation = model.loc[Input].sort_values(ascending=False)[0:20]
	pred_Recommendation = pred_Recommendation.to_frame().reset_index()
	pred_Recommendation.columns = ['Product_Name', 'Recommendation_Score']
	# Top 20 product name based on User-User Recommendation model
	Product_Name = list(pred_Recommendation['Product_Name'])

	print("Product_Name: ", list(pred_Recommendation['Product_Name']))
	print("Recommendation_Score: ", list(pred_Recommendation['Recommendation_Score']))

	#Read Pickle file for clear text and retrieve reviews based on recommended product
	with open('pickles/df_clean_reviews.pkl','rb') as fp1:
		data = pickle.load(fp1)

	#Read tfidf_word_vectorizer
	with open('pickles/tfidf_word_vectorizer.pkl','rb') as fp2:
		tf_idf = pickle.load(fp2)
	
	# Load LR Model
	with open('pickles/LR_model.pkl','rb') as fp3:
		model_lr = pickle.load(fp3)
	
	# Store unique Username in list
	#users = list(data.reviews_username.unique())

	setiment = {}

	for p_name in Product_Name:
		print("p_name :",p_name)
		#filtered_df = data[data.name.isin(p_name.split( "#" ))]
		#Filter reviews clean dataframe based on recommended Product Name
		filtered_df = data.loc[data['name'] == p_name] 
		x_filtered_df=filtered_df['reviews_text']
		#Create TF-IDF word vectorizer
		x_transformed = tf_idf.transform(x_filtered_df.values.astype('U'))
		#Pass this input value to Logistic Regression Model to generate sentiment (1=Poitive, 0=Negative sentiments)
		y_pred = model_lr.predict(x_transformed)
		#Calculate Positive Sentiment Percentage
		percent_positive_sent = round((sum(y_pred)/len(y_pred))*100,2)
		#Store Positive Sentiment Percentage against product name in dictionary
		setiment[p_name] = percent_positive_sent

	print("Setiment :",setiment)
	# Retrieve Top 5 Products with highest Positive Sentiment Percentage
	top5_sorted_sentiment = dict(sorted(setiment.items(), key=operator.itemgetter(1), reverse=True)[:5])
	print("### Top5 Setiments for User ", Input, " is: ",top5_sorted_sentiment)

	return render_template('index.html', name=top5_sorted_sentiment, user = Input, all_user=users)
	#return render_template('index.html', name=top5_sorted_sentiment, user = Input)

if __name__ == "__main__":
    app.run(debug=True)