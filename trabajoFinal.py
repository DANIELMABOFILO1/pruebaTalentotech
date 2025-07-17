import pandas as pd

import streamlit as st

import matplotlib.pyplot as plt

import seaborn as sns

import pickle

--select distinct fecha_demanda_maxima,  
--from energias.servicios_centros_poblados 
--order by fecha_demanda_maxima asc;

--select * from divapola.centros_poblados;

select energia_reactiva,centros_poblados 
from energias.servicios_centros_poblados
inner join centros_poblados