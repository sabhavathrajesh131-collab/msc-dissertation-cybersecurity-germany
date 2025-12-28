# Jupyter Notebook Description

## Notebook Title
GDELT Cybersecurity Analysis – German Financial Sector (2019–2024)

## Purpose of the Notebook
This Jupyter Notebook provides a transparent and reproducible demonstration of the analytical workflow used in the MSc dissertation titled:

**"Cybersecurity Incident Response and Recovery in the German Financial Sector (2019–2024):  
A Mixed-Methods Analysis Using Event-Based Data and Machine Learning"**

The notebook is intended to support academic assessment by illustrating data loading, preprocessing, classification, and result generation.

## Key Design Principle
This notebook has been intentionally designed to **run without external runtime downloads**, such as NLTK corpora, to ensure compatibility with:

- University-managed systems  
- Restricted or offline environments  
- Corporate or firewall-limited networks  

To achieve this, the notebook uses **scikit-learn’s built-in English stopword list** instead of NLTK stopwords.

## Data Source
The primary data source is the **Global Database of Events, Language, and Tone (GDELT v2.1)**, accessed via the GDELT Document API.

Due to the size and continuously updating nature of the GDELT dataset, the full dataset is not included in this repository.  
Instead, the notebook demonstrates data extraction logic and processing steps using live API calls and representative samples.

Official GDELT website:  
https://www.gdeltproject.org

## Analytical Workflow
The notebook follows a structured analytical pipeline:

1. Data loading from the GDELT Document API  
2. Text preprocessing using regular expressions and built-in stopwords  
3. Baseline rule-based classification  
4. Machine learning classification using:
   - Logistic Regression
   - Support Vector Machine (SVM)  
5. Model comparison and generation of output tables  

Each step is documented to support clarity and reproducibility.

## Tools and Libraries
- Python  
- Pandas and NumPy for data handling  
- Scikit-learn for NLP and machine learning  
- Requests for API communication  

No external NLP datasets or downloads are required.

## Outputs
The notebook produces:
- A table showing the distribution of cybersecurity events  
- A model comparison table reporting accuracy and F1-score  

These outputs directly support the empirical results presented in the dissertation.

## Ethical Considerations
The notebook uses only publicly available, open-access data.  
No personal or sensitive information is processed.  
All analysis complies with academic data ethics and responsible research practices.

## Intended Use
This notebook is intended solely for:
- Academic evaluation  
- Methodological transparency  
- Demonstration of analytical approach  

It is not intended for operational or commercial cybersecurity deployment.

## Author
Rajesh  
MSc Business Analytics  
2025
