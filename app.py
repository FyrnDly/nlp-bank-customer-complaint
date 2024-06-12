import streamlit as st
from module import load_data, get_predict

st.set_page_config(
    page_title="NLP Classification Bank Customer Complaints",
    page_icon="🏦",
)

def intro():
    st.markdown('''
    # 🏦 Bank Customer Complaint 🏦
    ## 📖 Project Summary 📖
    ---

    ### 💣 Problem Statement
    > Banks often receive a multitude of customer complaints. Due to the sheer volume, bank customer service teams frequently struggle to categorize these complaints accurately. Consequently, **the complaint resolution process slows down, leading to customer dissatisfaction**.

    ### 🤖 NLP Implementation
    > o address this issue, we can employ Natural Language Processing (NLP) technology. With **NLP, we can create a system that automatically recognizes the content of customer complaints and determines the appropriate product category**. This will make the complaint handling process faster and more efficient.


    ### 🎯 Target
    > The goal of using this NLP system is to **speed up the response time** of the customer service team and **improve the accuracy** in classifying complaints. The target is to create an NLP system with an **accuracy rate of at least 90%**, measured by metrics such as Accuracy, Precision, Recall, and F1-Score.

    With this system, banks are expected to respond to customer complaints more quickly and accurately, thereby increasing customer satisfaction and the efficiency of the customer service team. I hope this explanation helps you convey your project more clearly to the readers.
    ''')
    
def model():
    data = load_data()
    
    example = data[data['complaints_len'] <= 150].sample(1)
    example_input = example['complaints'].values[0]
    example_label = example['label'].values[0]
    
    st.markdown('# 🤖 Model Prediction 🤖')
    
    form_model = st.form("Predict Customer Complaints")    
    form_model.markdown(f'''
    ## Predict Customer Complaints
    ### Example Input Product: **{example_label}**
    > {example_input}
    ''')
    
    input_complaints = form_model.text_area('Input Complaint')
    submit = form_model.form_submit_button('Submit Complaint')
    
    if submit:
        result, detail = get_predict(input_complaints)
        
        form_model.subheader(f'Output: {result}')   
        for label, value in detail.items():
            form_model.markdown(f'{label}: {value}')
    
page_names_to_funcs = {
    "Summary Project": intro,
    "Model Predict": model,
}

selected_page = st.sidebar.selectbox("Go to Page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()