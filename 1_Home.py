import streamlit as st
import pandas as pd
import openpyxl


st.set_page_config(
    page_title="Loan Approval Dashboard",
    page_icon="👋",
    layout="wide"
)
cleaned_data = pd.read_excel("Cleaned.xlsx", engine='openpyxl', index_col=0)
cleaned_data['Result']= cleaned_data.apply(
    lambda row: 'Good' if row['Label'] == 0 else 'Bad', axis=1
    )

st.header("About the Assignment")
st.subheader("Dataset Background:")

st.markdown(
    """
SMEs that have come to Validus for a loan would need to submit their financials(in PDF, scanned, picture format). After OCR, the financials are usually stored in csv format with the company's later performance label. For these tasks, you would need to analyze based on the financial data provided in the attachment.""")
st.subheader("Tasks:")
st.markdown(
    """
Process the test set with any programming language you are familiar with (Python/R/Excel VBA), to convert the test set into the same format as the train set.
Use data provided for predictive modeling, use the variable 'Label' as prediction label.
(Optional) Provide some business suggestions based on your model. You can submit using a PDF report or some dashboarding tool such as Power BI or Tableau."""
)
data_input =pd.read_excel("train_data.xlsx", engine='openpyxl', sheet_name='Raw_data')

data_input