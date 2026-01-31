# Simulador de Transporte Urbano — Ciudad Lineal

Aplicación interactiva desarrollada en **Streamlit** para simular el equilibrio de transporte en una ciudad lineal considerando distintos modos de viaje, congestión, infraestructura y variables socioeconómicas.

La herramienta permite experimentar con políticas de transporte y observar cómo cambian:

- Tiempos de viaje  
- Elección modal  
- Congestión  
- Uso de infraestructura  
- Emisiones estimadas  

La aplicación interactiva se puede acceder directamente desde el navegador [aquí](https://titirilquenv1.streamlit.app/)
 

## 1. Requisitos

Necesitas tener instalado:

- **Python 3.10 o superior**
- **pip** (viene con Python)
- Conexión a internet para instalar dependencias

 

## 2. Instalar Python

Descarga Python desde:  
https://www.python.org/downloads/

Durante la instalación **marca esta opción**:

```
Add Python to PATH
```

Luego verifica que quedó bien instalado:

```bash
python --version
```

Si no funciona, prueba:

```bash
python3 --version
```

 

## 3. Descargar el proyecto

### Opción A — Con Git

```bash
git clone https://github.com/TU_USUARIO/TU_REPOSITORIO.git
cd TU_REPOSITORIO
```

### Opción B — Manual

1. Presiona **Code → Download ZIP** en GitHub  
2. Descomprime el archivo  
3. Abre la carpeta del proyecto en una terminal  

 

## 5. Instalar dependencias


```bash
pip install -r requirements.txt
```

Si no funciona, instala manualmente:

```bash
pip install streamlit numpy pandas matplotlib seaborn
```

 

## 6. Ejecutar la aplicación

Desde la carpeta del proyecto:

```bash
streamlit run app.py
```

Se abrirá el navegador automáticamente.

Si no se abre, copia la dirección que aparece en la terminal, normalmente:

```
http://localhost:8501
```


## 7. Archivos importantes

```
app.py        → Aplicación principal  
Ciudad2.py    → Modelo urbano (obligatorio)  
requirements.txt  
README.md
```

`Ciudad2.py` debe estar en la **misma carpeta** que `app.py`.



## 8. Qué permite hacer el simulador

Puedes modificar parámetros como:

- Densidad poblacional  
- Infraestructura vial  
- Capacidad de transporte público  
- Tarifas  
- Políticas pro-auto o pro-transporte público  
- Penalizaciones por caminata o bicicleta  

Y observar:

- Distribución modal  
- Tiempos promedio por grupo  
- Niveles de congestión  
- Estimaciones de emisiones  



## 9. Problemas comunes

### Python no se reconoce

Reinstala Python y asegúrate de marcar:

```
Add Python to PATH
```

 

### Streamlit no se reconoce

```bash
pip install streamlit
```

 

### Error relacionado con `Ciudad2`

Verifica que el archivo:

```
Ciudad2.py
```

esté en la misma carpeta que `app.py`.

 

### Puerto ocupado

Ejecuta en otro puerto:

```bash
streamlit run app.py --server.port 8502
```

## 10. Licencia

GPLv3
El uso de este repositorio es de libre acceso y modificación, pero no se puede cambiar su uso a uno privativo. 
