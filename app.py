import dill
import streamlit as st
import encoder_data
import pandas as pd
page_bg_img="""
<style>
[data-testid="stAppViewContainer"]{
    background-image:url(""D:\MACHINE LEARING AT UNIVERSITY\pixabay_brain-stroke_1200.jpg"");
}
</style>
"""
st.markdown(page_bg_img,unsafe_allow_html=True) 
feature=['age',
 'hypertension',
 'heart_disease',
 'ever_married',
 'avg_glucose_level',
 'bmi']
with st.sidebar:
    st.title("Stroke detection app")
    st.image("./pixabay_brain-stroke_1200.jpg")
    df = pd.DataFrame.from_dict({
        'age':[float(st.slider("age",1.00000, 80.000000, 25.0))],
    'hypertension':[st.radio(
        "hypertension",
        ('Yes',"No"))],
    'heart_disease':[st.radio(
        "Heart_disease",
        ('Yes',"No"))],
    'ever_married':[st.radio(
        "ever_married",
        ('Yes',"No"))],
    'avg_glucose_level':[st.slider("avg_glucose_level",50.0, 250.00	, 60.0)],
    'bmi':[st.slider("BMI",15.000000, 40.0000, 20.0)],
})
st.title("Stroke Detection App")
new_data=encoder_data.new_data_num(df)
with open("xgboost_model_os_weights.dill", "rb") as f:
    model = dill.load(f)
st.write("Probability stroke: ",model.predict_proba(new_data)[0][1]*100,"%")
st.image("./00.-Machine-Learing.png")