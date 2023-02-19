# Resume Screener
This is a Python program that uses natural language processing techniques to match resumes with job descriptions. The program takes a job description in the form of a Word document and a resume in the form of a separate Word document, and then uses cosine similarity to determine how well the resume matches the job description.

The match percentage is displayed on a gauge chart, and the predicted job position for the resume is displayed. The application also includes a feature to upload multiple resumes for screening and displays the match percentage and predicted job position for each resume on a chart. The application uses the sklearn library for text processing, nltk for tokenization, docx2txt for reading word documents, and streamlit for building the web app. It also includes a class JobPredictor that predicts the job position of the given resume using a trained model saved as pickles.

## Prerequisites
Before running this program, you will need to install the packages using following command:
```sh 
pip install -r requirement.txt
```

## How to use
- Clone this repository to your local machine.
- Install the required packages (see Prerequisites above).
- Place the job description Word document in the project directory and name it "temp_jd.docx".
- Run the program using 
    ```sh
    streamlit run resume_screener.py.
    ````
- Use the file uploader to upload your resume.
- Click the "Submit" button to see how well your resume matches the job description.

## Explanation of code
The program uses the docx2txt library to convert the Word documents to text, and the CountVectorizer class from scikit-learn to create a sparse matrix of word counts for each document. The program then uses cosine similarity from scikit-learn to compare the two sparse matrices and compute a similarity score.

The program also uses the pickle library to load pre-trained models for label encoding, word vectorization, and classification.

Finally, the program uses the streamlit library to create a web application with a user interface for uploading the job description and resume documents.

## Author
This program was created by [Lakpa Sherpa](https://slakpa.com.np).