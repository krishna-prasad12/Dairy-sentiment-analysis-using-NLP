import streamlit as kp
import nltk
nltk.downloader.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer
import glob
import  plotly.express as px

analyzer=SentimentIntensityAnalyzer()
# kp.title('Dairy sentiment analysis')
filepaths=glob.glob('diary/*.txt')
# print(filepaths)
positivity=[]
negativity=[]
for filepath in filepaths:
    with open(filepath,'r') as file:
        dairy=file.read()
        # print(dairy)
    score=analyzer.polarity_scores(dairy)
    # print(score)
    positivity.append(score['neg'])
    negativity.append(score['pos'])

dates=[name.strip('.txt').strip('dairy/') for name in filepaths]

kp.title('Dairy Analysis')
kp.subheader('Positivity Graph')
figure=px.line(x=dates,y=positivity,labels={'x':'Dates','y':'Positive'})
kp.plotly_chart(figure)

kp.subheader('Negativity Graph')
figure=px.line(x=dates,y=negativity,labels={'x':'Dates','y':'Negative'})
kp.plotly_chart(figure)





