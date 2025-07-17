import pandas as pd

import streamlit as st

import matplotlib.pyplot as plt

import seaborn as sns

import locale

import pickle

# rb (bytes leídos): Para leer y no escribir
rf_pickle = open('random_forest_penguin.pickle', 'rb')
map_pickle = open('output_penguin.pickle', 'rb')
rfc = pickle.load(rf_pickle)
unique_penguin_mapping = pickle.load(map_pickle)
rf_pickle.close()
map_pickle.close()

locale.setlocale(locale.LC_ALL, 'es_ES.UTF-8')

#Es necesario importar las depencendias necesarias
from datetime import date
from datetime import datetime

#Día actual
today = date.today()

#Fecha actual
now = datetime.now()


dayName = today.strftime('%A %d de %B del Año %Y')


# Configuración de la página

st.set_page_config(layout='centered', page_title='Talento Tech')

t1, t2 = st.columns([0.3, 0.7])
t1.image('Matrix_code.webp', width=200)
t2.title('Martes 17')
t2.markdown('**tel:** 3218725370 **| email:** danielmabofilo@gmail.com')

with st.sidebar:
    st.write('Sidebar')

steps=st.tabs(['Penguins', 'Pestaña 1','Pestaña 3', 'Apps.co','Epidemiología','Árboles','trabajo Final'])

df = pd.read_csv('penguins.csv')

with steps[0]:
    st.write('Información 1')
    st.write(df.head())
    st.write('Hoy es ', dayName)
    especie = st.selectbox('Seleccione la especie',
                           ['Adelie','Gentoo', 'Chinstrap'])
    
    variable_x = st.selectbox('Seleccione la variable x', 
                           ['bill_length_mm','bill_depth_mm', 'flipper_length_mm','body_mass_g'])
    variable_y = st.selectbox('Seleccione la variable y', 
                           ['bill_length_mm','bill_depth_mm', 'flipper_length_mm','body_mass_g'])
    
    df = df[df['species']==especie]

    fig, ax = plt.subplots()
    ax = sns.scatterplot(x=df[variable_x], y=df[variable_y])
    plt.xlabel(variable_x)
    plt.ylabel(variable_y)
    plt.title('Gráfica de la especie {} de pingüinos'.format(especie))
    st.pyplot(fig)

with steps[1]:
    st.markdown('# Podemos usar $\LaTeX$ $$\dfrac{\pi}{2}$$')
    sns.set_style('darkgrid')
    df1=pd.read_csv('penguins.csv')
    markers={'Adelie':'x','Gentoo':'s','Chistrap':'0'}
    fig1, ax1=plt.subplots()
    ax1=sns.scatterplot(data=df1, x=variable_x, y=variable_y,
                        hue='species', markers=markers)
    plt.xlabel(variable_x)
    plt.ylabel(variable_y)
    st.pyplot(fig1)

with steps[2]:
    st.title('Árboles')
    df2=pd.read_csv('trees.csv')
    st.dataframe(df2)
    dfg=pd.DataFrame(df2.groupby(['dbh'])['tree_id'].count())
    dfg.columns=['tree_count']
    t1, t2, t3 = st.columns([0.3, 0.3, 0.3])

    st.line_chart(dfg)
    st.bar_chart(dfg)
    st.area_chart(dfg)

    t1.line_chart(dfg)
    t2.bar_chart(dfg)
    t3.area_chart(dfg)

    df3 = df2.dropna(subset=['longitude','latitude'])
    st.map(df3)

with steps[3]:
    st.title('apps.co')
    df3 = pd.read_csv('Beneficiarios_de_la_Iniciativa_Apps.co_20250619.csv',encoding='UTF-8')
    st.dataframe(df3)

with steps[4]:
    st.title('Epidemiologia Colombia')
    df4 = pd.read_csv('Eventos_Epidemiol_gicos_20250619.csv',encoding='UTF-8')
    st.dataframe(df4)

with steps[5]:
    # Para ver si carga el modelo
    st.write(rfc)
    st.write(unique_penguin_mapping)
    # Opciones para el ususario
    island = st.selectbox('Isla', options=['Biscoe', 'Dream', 'Torgerson'])
    specie = st.selectbox('Specie', options=['Gentoo', 'Adelie','Chinstrap'])
    bill_length = st.number_input('Bill Length (mm)', min_value=0)
    bill_depth = st.number_input('Bill Depth (mm)', min_value=0)
    flipper_length = st.number_input('Flipper Length (mm)', min_value=0)
    body_mass = st.number_input('Body Mass (g)', min_value=0)
    st.write('Los datos ingresados son {}'.format([island, specie, bill_length, bill_depth, flipper_length, body_mass]))
    # Codificación para las islas
    island_biscoe, island_dream, island_torgerson = 0, 0, 0
    if island == 'Biscoe':
        island_biscoe = 1
    elif island == 'Dream':
        island_dream = 1
    elif island == 'Torgerson':
        island_torgerson = 1
    
    # Codificación para el sexo
    specie_gentoo, specie_adelie, specie_chinstrap = 0, 0, 0
    if specie == 'Gentoo':
        specie_gentoo = 1
    elif specie == 'Adelie':
        specie_adelie = 1
    elif specie == 'Chinstrap':
        specie_chinstrap = 1
    
    # Modelo ML
    new_prediction = rfc.predict([[bill_length, bill_depth, flipper_length, body_mass, island_biscoe,island_dream, island_torgerson, specie_gentoo, specie_adelie, specie_chinstrap]])
    prediction_sex = unique_penguin_mapping[new_prediction][0]
    st.write('El sexo del pingüino es {}'.format(prediction_sex))

with steps[6]:
    st.title('Trabajo Final')
    df5 = pd.read_csv('Estado_de_la_prestaci_n_del_servicio_de_energ_a_en_Zonas_No_Interconectadas_20250702.csv',encoding='UTF-8')
    st.dataframe(df5)



