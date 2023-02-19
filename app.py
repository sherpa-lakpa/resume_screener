from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import docx2txt
from nltk.tokenize import WhitespaceTokenizer
import streamlit as st
import pickle

import plotly.graph_objects as go
import pandas as pd
import numpy as np
import plotly.express as px


class JobPredictor:
    def __init__(self) -> None:
        save_label_encoder = open("pickles/le.pickle","rb")
        self.le = pickle.load(save_label_encoder)
        save_label_encoder.close()

        save_word_vectorizer = open("pickles/word_vectorizer.pickle","rb")
        self.word_vectorizer = pickle.load(save_word_vectorizer)
        save_word_vectorizer.close()

        save_classifier = open("pickles/clf.pickle","rb")
        self.clf = pickle.load(save_classifier)
        save_classifier.close()

    def predict(self, resume):
        feature = self.word_vectorizer.transform([resume])
        predicted = self.clf.predict(feature)
        resume_position = self.le.inverse_transform(predicted)[0]
        return resume_position

    def predict_proba(self, resume):
        feature = self.word_vectorizer.transform([resume])
        predicted_prob = self.clf.predict_proba(feature)
        return predicted_prob[0]


text_tokenizer= WhitespaceTokenizer()
# reference : https://stackoverflow.com/questions/47301795/how-can-i-remove-special-characters-from-a-list-of-elements-in-python
remove_characters= str.maketrans("", "", "±§!@#$%^&*()-_=+[]}{;'\:,./<>?|")
#takes the job description word document in a text format
# job_description= docx2txt.process("temp_jd.docx")
cv = CountVectorizer()

st.title("Resume Screener")

jd_docx=st.file_uploader("Upload your Job Description",type=["docx"])
#converts pdf format to text
if st.button("Submit",key='1'):
    if jd_docx is not None:
        print(jd_docx)
        with open("temp_jd.docx","wb") as f:
            f.write(jd_docx.getbuffer())
        job_description = docx2txt.process(jd_docx)
        resume_position = JobPredictor().predict(job_description)
        
        st.write(f'JD uploaded! Position: {resume_position}')
    else:
        pass
    

file_docx=st.file_uploader("Upload you CV\Resume",type=["docx"])
if st.button("Submit"):
    if file_docx is not None:
        #converts doc format to text
        resume_docx = docx2txt.process(file_docx)

        #takes the job description word document in a text format
        job_description= docx2txt.process("temp_jd.docx")

        #takes the texts in a list
        text_docx= [resume_docx, job_description]
        #creating the list of words from the word document
        words_docx_list = text_tokenizer.tokenize(resume_docx)
        #removing speacial charcters from the tokenized words 
        words_docx_list=[s.translate(remove_characters) for s in words_docx_list]
        #giving vectors to the words
        count_docx = cv.fit_transform(text_docx)
        #using the alogorithm, finding the match between the resume/cv and job description
        similarity_score_docx = cosine_similarity(count_docx)
        match_percentage_docx= round((similarity_score_docx[0][1]*100),2)
        st.write(f'Match percentage with the Job description: {match_percentage_docx}')

        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = match_percentage_docx,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Match with JD"}))

        st.plotly_chart(fig, use_container_width=True)

        job_predictor = JobPredictor()
        resume_position = job_predictor.predict(resume_docx)

        chart_data = pd.DataFrame({
            "position": [cl for cl in job_predictor.le.classes_],
            "match": job_predictor.predict_proba(resume_docx)
        })

        fig = px.bar(chart_data, x="position", y="match",
                        title=f'Resume matched to: {resume_position}')
        st.plotly_chart(fig, use_container_width=True)

    else:
        pass

st.title("HR Resume Screener")
uploaded_files = st.file_uploader("Choose a multiple files", type=["docx"], accept_multiple_files=True)
if st.button("Submit",key='2'):
    if uploaded_files is not None:
        job_predictor = JobPredictor()
        job_positions = {x: 0 for x in [cl for cl in job_predictor.le.classes_]}
        match_percentage = {}
        for uploaded_file in uploaded_files:
            resume_docx = docx2txt.process(uploaded_file)
            resume_position = job_predictor.predict(resume_docx)
            job_positions[resume_position] += 1

            job_description= docx2txt.process("temp_jd.docx")
            text_docx= [resume_docx, job_description]
            words_docx_list = text_tokenizer.tokenize(resume_docx)
            words_docx_list=[s.translate(remove_characters) for s in words_docx_list]
            count_docx = cv.fit_transform(text_docx)
            similarity_score_docx = cosine_similarity(count_docx)
            match_percentage_docx= round((similarity_score_docx[0][1]*100),2)
            match_percentage[uploaded_file.name] = match_percentage_docx


        match_chart_data = pd.DataFrame({
            "document": match_percentage.keys(),
            "percentage": match_percentage.values()
        })

        fig = px.bar(match_chart_data, x="document", y="percentage", title='Document Matched Percentage')
        st.plotly_chart(fig, use_container_width=True)

        resume_position = job_predictor.predict(job_description)
        total_matched = job_positions[resume_position]
        total_files = len(uploaded_files)
        
        st.write(f'Position of the Job description: {resume_position}')

        fig = go.Figure(go.Indicator(
            mode = "delta+number",
            # gauge = {'axis': {'visible': False}},
            delta = {'reference': len(uploaded_files)},
            value = total_matched,
            domain = {'row': 0, 'column': 0},
            title = {'text': f"{total_matched} out of {len(uploaded_files)} Resume falls on same category of JD."}))

        st.plotly_chart(fig, use_container_width=True)
        
        df = pd.DataFrame({
            'names': ['Matched', 'Unmatched'], 
            'values': [total_matched, total_files]
        })
        fig = px.pie(df, values='values', names='names')
        st.plotly_chart(fig, use_container_width=True)

        chart_data = pd.DataFrame({
            "position": job_positions.keys(),
            "match": job_positions.values()
        })

        fig = px.bar(chart_data, x="position", y="match", title=f'Resume Job Position distribution')
        st.plotly_chart(fig, use_container_width=True)
