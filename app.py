import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

st.set_page_config(layout='wide')

scaler = joblib.load('Scaler.pkl')

st.title('Prediccion de Clasificacion de Restaurantes App')



st.caption('Esta aplicacion te ayuda a predecir la clasificacion de un restaurante.')

st.divider()

costo_promedio = st.number_input('Por favor ingrese el costo estimado para dos personas: ', min_value=50, max_value= 999999, value= 1000, step = 200)

reserva_de_mesa = st.selectbox("¿El restaurante tiene reserva de mesa?", ['Si', 'No'])

delivery = st.selectbox("¿El restaurante tiene delivery?", ['Si', 'No'])

rango_de_precio = st.selectbox('Cual es el rango de precios (1 Barato - 4 Caro)',[1,2,3,4])



predictbutton = st.button('¡Predecir!')

st.divider()

model = joblib.load('mlmodel.pkl')

estado_de_reserva = 1 if reserva_de_mesa == 'Yes' else 0

delyveri_estdo = 1 if delivery == 'Yes' else 0

values = [[costo_promedio,estado_de_reserva,delyveri_estdo,rango_de_precio]]
my_X_values = np.array(values)

X = scaler.transform(my_X_values)

if predictbutton:
    st.snow()
    prediction = model.predict(X)
    
    if prediction < 2.5:
        st.write('Pobre')
    elif prediction < 3.5:
        st.write('Promedio')
    elif prediction < 4.0:
        st.write('Bueno')
    elif prediction < 4.5:
        st.write('¡Muy bueno!')
    else:
        st.write('¡Excelente!')
 
 