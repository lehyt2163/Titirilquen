import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import random
import copy

from Ciudad2 import Ciudad



# CONFIGURACIÓN Y PRESETS
st.set_page_config(page_title="Simulador Transporte - Ciudad Lineal", layout="wide")

# Definición de Escenarios (Presets)
PRESETS_CIUDAD = {
    "Personalizado": {},
    "Compacta": {"largo_ciudad": 12, "densidad": 250},
    "Base": {"largo_ciudad": 20, "densidad": 180},
    "Dispersa": {"largo_ciudad": 30, "densidad": 100}
}

PRESETS_MODOS = {
    "Personalizado": {},
    "TP Gratis": {"tarifa": 0, "parking": 6000, "num_pistas": 2, "num_estaciones": 10, "bencina": 120, "cap_bici": 800, "frec_max": 35, "cap_tren": 1200},
    "Tarificación Vial": {"tarifa": 800, "parking": 15000, "num_pistas": 2, "num_estaciones": 10, "bencina": 120, "cap_tren": 1200, "cap_bici": 800, "frec_max": 20},
    "Pro-Auto": {"tarifa": 1000, "parking": 3000, "num_pistas": 3, "num_estaciones": 8, "bencina": 100, "cap_tren": 1000, "cap_bici": 500, "frec_max": 6},
    "Pro-Bici": {"tarifa": 800, "parking": 6000, "num_pistas": 2, "cap_bici": 5000, "frec_max": 20, "bencina": 120, "cap_tren": 1200, "num_estaciones": 10},
    "Vehículos híbridos": {"num_pistas": 2, "bencina": 65, "tarifa": 800, "parking": 6000, "frec_max": 20, "cap_tren": 1200, "num_estaciones": 10, "cap_bici": 800},
    "Máx Metro": {"tarifa": 400, "num_estaciones": 20, "frec_max": 50, "cap_tren": 1200, "parking": 6000, "bencina": 120, "num_pistas": 2, "cap_bici": 800},
    "Ciclorrecreovía": {"num_pistas": 1, "cap_bici": 6000, "tarifa": 800, "parking": 6000, "bencina": 120, "frec_max": 20, "cap_tren": 1200}
}

# Inicialización de Session State para persistencia de datos
defaults = {
    "largo_ciudad": 20, "num_estaciones": 10, "num_pistas": 2,
    "frec_max": 20, "cap_tren": 1200, "cap_bici": 800,
    "tarifa": 800, "parking": 6000, "bencina": 120,
    "densidad": 100, "teletrabajo_factor": 1.0
}

for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# Callbacks para aplicar presets
def aplicar_preset_ciudad():
    seleccion = st.session_state.preset_ciudad
    if seleccion != "Personalizado":
        vals = PRESETS_CIUDAD[seleccion]
        for k, v in vals.items():
            st.session_state[k] = v

def aplicar_preset_modos():
    seleccion = st.session_state.preset_modos
    if seleccion != "Personalizado":
        vals = PRESETS_MODOS[seleccion]
        for k, v in vals.items():
            if k in st.session_state: # Solo actualizar si existe la key
                st.session_state[k] = v

# CLASES Y CONFIGURACIÓN FÍSICA


class CiudadLineal:
    def __init__(self, n_celdas, ancho_celda):
        self.n_celdas = n_celdas
        self.cbd_index = n_celdas // 2
        self.ancho_celda = ancho_celda
        self.largo_total = n_celdas * ancho_celda

CONFIG_DEMANDA = {
    "globales": {
        "v_auto": 31, "v_metro": 35, "v_bici": 14, "v_caminata": 4.8, 
        "costo_combustible_km": 120, "costo_tarifa_metro": 800, "costo_parking": 6000,
        "factor_emision_auto": 0.180, "factor_emision_metro": 0.040
    },
    "estratos": {
        1: { 
            "prob_teletrabajo": 0.40, 
            "prob_auto": 0.90, 
            "prob_jornada_flexible": 0.50, "prob_part_time": 0.05,
            "jornada": {"horas_rigido": 9.0, "horas_flexible": 8.0, "horas_part_time": 4.0},
            
            "betas": { "asc_auto": 1.5, "asc_metro": -0.2, "asc_bici": -0.9, "asc_caminata": -0.5, "b_tiempo_viaje": -0.055, "b_costo": -0.00008, "b_tiempo_espera": -0.05, "b_tiempo_caminata": -0.15, "penalizaciones_fisicas": {"bici_10": -0.09, "bici_20": -0.15, "bici_30": -0.5, "walk_5": -0.09, "walk_15": -0.18, "walk_25": -0.4} } 
        },
        2: { 
            "prob_teletrabajo": 0.20, 
            "prob_auto": 0.60, 
            "prob_jornada_flexible": 0.30, "prob_part_time": 0.10,
            "jornada": {"horas_rigido": 9.0, "horas_flexible": 8.5, "horas_part_time": 4.5},

            "betas": { "asc_auto": 0.7889, "asc_metro": 0.1040, "asc_bici": -0.6818, "asc_caminata": 0.1, "b_tiempo_viaje": -0.0331, "b_costo": -0.0002, "b_tiempo_espera": -0.0243, "b_tiempo_caminata": -0.0440, "penalizaciones_fisicas": {"bici_10": -0.0634, "bici_20": -0.1, "bici_30": -0.4, "walk_5": -0.05, "walk_15": -0.09, "walk_25": -0.2} } 
        },
        3: { 
            "prob_teletrabajo": 0.05, 
            "prob_auto": 0.30, 
            "prob_jornada_flexible": 0.10, "prob_part_time": 0.15,
            "jornada": {"horas_rigido": 9.5, "horas_flexible": 9.0, "horas_part_time": 5.0},

            "betas": { "asc_auto": 0.2, "asc_metro": 0.25, "asc_bici": -0.4, "asc_caminata": 0.4, "b_tiempo_viaje": -0.0150, "b_costo": -0.0006, "b_tiempo_espera": -0.0150, "b_tiempo_caminata": -0.0250, "penalizaciones_fisicas": {"bici_10": -0.0300, "bici_20": -0.0500, "bici_30": -0.7, "walk_5": -0.0250, "walk_15": -0.0400, "walk_25": -0.08} } 
        }
    }
}

# FUNCIONES DE OFERTA
def demora_auto_tramo(ubicacion_centro_km, demanda, v_m, a, l_veh, gap, L_ciudad_km, N_pistas, alpha_a, beta_a):
    f_a = 1 if a >= 3.5 else (0.9 if 3 < a < 3.5 else 0.75)
    v_l = v_m * f_a
    C = ((1000 / (l_veh + gap)) * v_l) / 4
    # Protección contra 0 pistas
    if N_pistas < 1: N_pistas = 1
    Capacidad_direccion = C * N_pistas

    N = len(demanda)
    idx_centro = int((ubicacion_centro_km / L_ciudad_km) * N)
    idx_centro = max(0, min(idx_centro, N - 1))

    t0 = ((L_ciudad_km / N) / v_l) * 60
    t_usuarios = np.zeros(N)
    flujos_en_via = np.zeros(N)
    demanda_aj = demanda.copy()
    demanda_aj[idx_centro] = 0

    if idx_centro > 0:
        d_izq = demanda_aj[:idx_centro]
        flujo_izq = np.cumsum(d_izq)
        flujos_en_via[:idx_centro] = flujo_izq
        with np.errstate(divide='ignore', invalid='ignore'):
            t_local = t0 * (1 + alpha_a * ((flujo_izq / Capacidad_direccion)**beta_a))
        t_usuarios[:idx_centro] = np.cumsum(t_local[::-1])[::-1] - (t_local / 2)

    if idx_centro < N - 1:
        d_der = demanda_aj[idx_centro+1:]
        flujo_der = np.cumsum(d_der[::-1])[::-1]
        flujos_en_via[idx_centro+1:] = flujo_der
        with np.errstate(divide='ignore', invalid='ignore'):
            t_local = t0 * (1 + alpha_a * ((flujo_der / Capacidad_direccion)**beta_a))
        t_usuarios[idx_centro+1:] = np.cumsum(t_local) - (t_local / 2)

    return t_usuarios, flujos_en_via, Capacidad_direccion, alpha_a, beta_a, v_l

def demora_bici_tramo(ubicacion_centro_km, capacidad, demanda, v_media, L_ciudad_km, alpha, beta, pendiente_porcentaje):
    N = len(demanda)
    idx_centro = int((ubicacion_centro_km / L_ciudad_km) * N)
    idx_centro = max(0, min(idx_centro, N - 1))
    dx = L_ciudad_km / N
    
    def velocidad(v_media, pendiente):
        factor = -0.0579 * pendiente + 0.9992 if pendiente > 0 else -0.0455 * pendiente + 1
        return max(min(v_media * factor, 45), 5)
        
    v_izq = velocidad(v_media, pendiente_porcentaje)
    v_der = velocidad(v_media, -pendiente_porcentaje)
    t0_izq = (dx / v_izq) * 60; t0_der = (dx / v_der) * 60
    
    t_usuarios = np.zeros(N)
    flujos_en_pista = np.zeros(N)
    demanda_aj = demanda.copy()
    demanda_aj[idx_centro] = 0
    
    if idx_centro > 0:
        d_izq = demanda_aj[:idx_centro]
        flujo_izq = np.cumsum(d_izq)
        flujos_en_pista[:idx_centro] = flujo_izq
        t_local_izq = t0_izq*(1 + alpha * ((flujo_izq / capacidad)**beta))
        t_ac_izq= np.cumsum(t_local_izq[::-1])[::-1]
        t_usuarios[:idx_centro] = t_ac_izq - (t_local_izq / 2)
        
    if idx_centro < N - 1:
        d_der = demanda_aj[idx_centro+1:]
        flujo_der = np.cumsum(d_der[::-1])[::-1]
        flujos_en_pista[idx_centro+1:] = flujo_der
        t_local_der = t0_der*(1 + alpha * ((flujo_der / capacidad)**beta))
        t_ac_der = np.cumsum(t_local_der)
        t_usuarios[idx_centro+1:] = t_ac_der - (t_local_der / 2)
        t_usuarios[idx_centro] = 0
        flujos_en_pista[idx_centro] = 0
    return t_usuarios,flujos_en_pista

def oferta_tren(Demanda, L_ciudad, x_centro, v_t, K, n_s, v_c, tasa_carga, frec_min, frec_max):
    N = len(Demanda)
    dx = L_ciudad / N
    x_parcelas = np.arange(N) * dx + (dx / 2)

    dist_entre_est = L_ciudad / n_s
    estaciones = np.unique(np.concatenate([
        np.arange(x_centro, L_ciudad + 0.01, dist_entre_est),
        np.arange(x_centro, -0.01, -dist_entre_est)
    ]))
    estaciones = np.sort(estaciones[(estaciones >= 0) & (estaciones <= L_ciudad)])
    num_s = len(estaciones)

    pax_suben = np.zeros(num_s)
    dist_acceso = np.zeros(N)
    loc_estacion_acceso = np.zeros(N)
    indices_estacion_usuario = np.zeros(N, dtype=int)

    for i in range(N):
        dists = np.abs(estaciones - x_parcelas[i])
        idx_est = np.argmin(dists)
        dist_acceso[i] = dists[idx_est]
        loc_estacion_acceso[i] = estaciones[idx_est]
        indices_estacion_usuario[i] = idx_est
        if i != int((x_centro / L_ciudad) * N):
             pax_suben[idx_est] += Demanda[i]

    idx_centro_est = np.argmin(np.abs(estaciones - x_centro))
    carga_por_tramo = np.zeros(num_s - 1)
    carga_al_salir_estacion = np.zeros(num_s)

    acum = 0
    for i in range(idx_centro_est):
        acum += pax_suben[i]
        if i < len(carga_por_tramo):
            carga_por_tramo[i] = acum
            carga_al_salir_estacion[i] = acum

    acum = 0
    for i in range(num_s - 1, idx_centro_est, -1):
        acum += pax_suben[i]
        if i-1 >= 0:
            carga_por_tramo[i-1] = acum
            carga_al_salir_estacion[i] = acum

    carga_maxima = np.max(carga_por_tramo) if len(carga_por_tramo) > 0 else 0
    f_teorica = carga_maxima / K
    f_op = np.clip(f_teorica, frec_min, frec_max)

    capacidad_maxima_sistema = frec_max * K
    t_espera_base = (1/(2*f_op))*60 if f_op > 0 else 0

    t_espera_por_estacion = np.zeros(num_s)
    alfa = 0.5
    beta = 4
    for i in range(num_s):
        if i == idx_centro_est:
            t_espera_por_estacion[i] = 0
            continue
        carga_local = carga_al_salir_estacion[i]
        ratio = carga_local / capacidad_maxima_sistema if capacidad_maxima_sistema > 0 else 0
        factor = 1 if ratio <= 1 else alfa * (ratio ** beta)
        t_espera_por_estacion[i] = t_espera_base * factor

    t_acceso_min = (dist_acceso / v_c) * 60
    t_espera_min = t_espera_por_estacion[indices_estacion_usuario]
    t_viaje_min = (np.abs(loc_estacion_acceso - x_centro) / v_t) * 60
    t_total = t_acceso_min + t_espera_min + t_viaje_min

    return t_acceso_min, t_espera_min, t_viaje_min, t_total, f_op, carga_por_tramo, estaciones

# FUNCIONES DE DEMANDA
def generar_poblacion(n_celdas, cbd_index, personas_por_celda, config_estratos):
    datos = []
    id_counter = 1
    for i in range(n_celdas):
        if i == cbd_index: continue
        for _ in range(personas_por_celda):
            estrato = random.choice([1, 2, 3])
            
            prob_tele = config_estratos[estrato]["prob_teletrabajo"]
            prob_auto = config_estratos[estrato].get("prob_auto", 0.5)
            
            teletrabaja = random.random() < prob_tele
            tiene_auto = random.random() < prob_auto
            minuto_entrada = int(np.random.normal(540, 30))
            
            usuario = {
                "id_unico": id_counter, "celda_origen": i, "estrato": estrato,
                "teletrabaja": teletrabaja, "tiene_auto": tiene_auto,
                "min_entrada": minuto_entrada
            }
            datos.append(usuario)
            id_counter += 1
    return datos

# UTILIDADES
def calcular_utilidades(usuario, ciudad, config, vector_tiempos):
    estrato = usuario['estrato']; celda = usuario['celda_origen']
    betas = config['estratos'][estrato]['betas']
    penal = betas['penalizaciones_fisicas']
    gl = config['globales']
    dist_km = abs(ciudad.cbd_index - celda) * ciudad.ancho_celda
    
    # Tiempos Base
    if vector_tiempos is None:
        t_auto = (dist_km / gl['v_auto']) * 60
        t_bici = (dist_km / gl['v_bici']) * 60
        t_tren_viaje = (dist_km / gl['v_metro']) * 60
        t_tren_esp = 5; t_tren_acc = 10
    else:
        d = vector_tiempos[celda]
        t_auto = d['Auto_Total']
        t_bici = d['Bici_Total']
        t_tren_viaje = d['Tren_Viaje_En_Vehiculo']
        t_tren_esp = d['Tren_Espera']
        t_tren_acc = d['Tren_Acceso']

    t_cam = (dist_km / gl['v_caminata']) * 60
    
    c_auto = (dist_km * gl['costo_combustible_km']) + gl['costo_parking']
    # AUTO
    v_auto_base = betas['asc_auto'] + betas['b_tiempo_viaje']*t_auto + betas['b_costo']*c_auto
    
    if usuario.get('tiene_auto', True):
        v_auto = v_auto_base
    # Si no tiene auto no puede viajar
    else:
        v_auto = -9999.0
     
    c_metro = gl['costo_tarifa_metro']
    # METRO
    v_metro = betas['asc_metro'] + betas['b_tiempo_viaje']*t_tren_viaje + \
              betas['b_tiempo_espera']*t_tren_esp + betas['b_tiempo_caminata']*t_tren_acc + \
              betas['b_costo']*c_metro
    
    # BICICLETA
    if t_bici > 45: v_bici = -9999.0
    else:
        p = 0
        if t_bici > 10: p += penal['bici_10']
        if t_bici > 20: p += penal['bici_20']
        if t_bici > 30: p += penal['bici_30']
        v_bici = betas['asc_bici'] + betas['b_tiempo_viaje']*t_bici + p
    
    # CAMINATA
    if t_cam > 30: v_cam = -9999.0
    else:
        p = 0
        if t_cam > 5: p += penal['walk_5']
        if t_cam > 15: p += penal['walk_15']
        if t_cam > 25: p += penal['walk_25']
        v_cam = betas['asc_caminata'] + betas['b_tiempo_caminata']*t_cam + p
        
    return {"Auto": v_auto, "Metro": v_metro, "Bici": v_bici, "Caminata": v_cam}

# ELECCIÓN DE MODO (LOGIT)
def elegir_modo(utilidades, tiene_auto):
    utils_filtradas = utilidades.copy()
    if not tiene_auto:
        utils_filtradas.pop("Auto", None) 
    modos = list(utils_filtradas.keys())
    v = np.array(list(utils_filtradas.values()))
    v = v - np.max(v)
    exp_v = np.exp(v)
    probs = exp_v / np.sum(exp_v)
    return random.choices(modos, weights=probs, k=1)[0]

# ITERACIONES
def correr_iteracion(poblacion, ciudad, config, tiempos_oferta):
    conteo = {"Auto": 0, "Metro": 0, "Bici": 0, "Caminata": 0, "Teletrabajo": 0}
    dem_auto = np.zeros(ciudad.n_celdas)
    dem_metro = np.zeros(ciudad.n_celdas)
    dem_bici = np.zeros(ciudad.n_celdas)
    
    for p in poblacion:
        if p['teletrabaja']:
            p['Modo'] = "Teletrabajo"
            p['Utilidad_Elegida'] = 0
            conteo["Teletrabajo"] += 1
            continue
        
        utils = calcular_utilidades(p, ciudad, config, tiempos_oferta)
        modo = elegir_modo(utils, p['tiene_auto'])
        
        p['Modo'] = modo 
        p['Utilidad_Elegida'] = utils[modo]
        
        conteo[modo] += 1
        idx = p['celda_origen']
        
        if modo == "Auto": dem_auto[idx] += 1
        elif modo == "Metro": dem_metro[idx] += 1
        elif modo == "Bici": dem_bici[idx] += 1
            
    return conteo, dem_auto, dem_metro, dem_bici

# =============================================================================
# INTERFAZ GRÁFICA
# =============================================================================

st.title("Simulador Ciudad Lineal")

PALETA_MODOS = {
    "Auto": "#FF8C00",      # Naranja 
    "Bici": "#228B22",      # Verde 
    "Caminata": "#0000FF",  # Azul 
    "Metro": "#FF0000",     # Rojo 
    "Teletrabajo": "#A9A9A9" # Gris
}

# --- SECCIÓN DE ESCENARIOS ---
with st.container():
    st.markdown("### Configuración de Escenarios")
    col_preset1, col_preset2 = st.columns(2)
    
    with col_preset1:
        st.selectbox("Estructura de Ciudad", list(PRESETS_CIUDAD.keys()), 
                     index=1, key="preset_ciudad", on_change=aplicar_preset_ciudad)
        
    with col_preset2:
        st.selectbox("Políticas de Transporte", list(PRESETS_MODOS.keys()), 
                     index=0, key="preset_modos", on_change=aplicar_preset_modos)
    
    st.markdown("---")

# --- CONTROLES MANUALES ---
tab_conf1, tab_conf2, tab_conf3 = st.tabs(["Infraestructura", "Economía", "Población"])

with tab_conf1:
    c1, c2 = st.columns(2)
    largo_ciudad = c1.slider("Largo Ciudad (km)", 5, 40, key="largo_ciudad")
    num_estaciones = c1.slider("Estaciones Metro", 3, 30, key="num_estaciones")
    num_pistas = c1.slider("Pistas Auto (por sentido)", 1, 4, key="num_pistas")
    
    frec_max = c2.slider("Frecuencia Máx (trenes/h)", 4, 60, key="frec_max")
    cap_tren = c2.number_input("Capacidad Tren (pax)", 500, 2000, key="cap_tren")
    cap_bici = c2.number_input("Capacidad Ciclovía (bici/h)", 100, 6000, key="cap_bici", step=100)

with tab_conf2:
    c1, c2 = st.columns(2)
    tarifa = c1.slider("Tarifa Metro ($)", 0, 2000, step=50, key="tarifa")
    parking = c1.slider("Estacionamiento ($)", 0, 15000, step=500, key="parking")
    bencina = c2.slider("Bencina ($/km)", 50, 300, step=10, key="bencina")

with tab_conf3:
    densidad = st.slider("Densidad Población (pax/celda)", 10, 500, key="densidad")
    teletrabajo_factor = st.slider("Factor Teletrabajo", 0.0, 2.0, key="teletrabajo_factor")

if st.button("▶ SIMULAR EQUILIBRIO", type="primary"):
    
    # 1. Setup
    config_sim = CONFIG_DEMANDA.copy()
    config_sim["globales"]["costo_tarifa_metro"] = tarifa
    config_sim["globales"]["costo_combustible_km"] = bencina
    config_sim["globales"]["costo_parking"] = parking
    
    # Ajuste profundo para teletrabajo
    config_sim["estratos"] = copy.deepcopy(CONFIG_DEMANDA["estratos"])
    for e in config_sim["estratos"]:
            config_sim["estratos"][e]["prob_teletrabajo"] *= teletrabajo_factor

    n_celdas = 1001
    total_hogares_estimados = int(densidad * n_celdas) # densidad viene del slider
    H_vec = [
        int(total_hogares_estimados * 0.10),
        int(total_hogares_estimados * 0.40),
        int(total_hogares_estimados * 0.50)
    ]
    
    mi_ciudad = Ciudad(
            L=n_celdas, 
            CBD=n_celdas//2, 
            H=H_vec, 
            y=[120.0, 50.0, 10.0], # Ingresos referenciales por estrato (puedes parametrizar esto)
            ancho_celda=largo_ciudad/n_celdas
        )
    
    with st.spinner("Generando demografía de la población..."):
            poblacion = mi_ciudad.generar_poblacion_completa(config=config_sim)
            # Ciudad2 no asigna autos, así que lo hacemos aquí usando la config
            for p in poblacion:
                prob_auto = config_sim["estratos"][p["estrato"]]["prob_auto"]
                p["tiene_auto"] = random.random() < prob_auto
            # ================================
    
    tiempos_actuales = None
    historia = {"Auto": [], "Metro": [], "Bici": [], "Caminata": []}
    final_data = {}
    frec_operativa_final = 0
    
    # BUCLE DE ITERACIONES
    MAX_ITER = 12 #número de iteraciones
    progress = st.progress(0)
    status = st.empty()
    
    for it in range(MAX_ITER):
        status.text(f"Iteración {it+1}/{MAX_ITER} - Equilibrando red...")
        
        conteo, d_auto, d_metro, d_bici = correr_iteracion(poblacion, mi_ciudad, config_sim, tiempos_actuales)
        for m in historia: historia[m].append(conteo[m])
        
        # OFERTA
        t_a, f_a, cap_a, alp_a, bet_a, vl_a = demora_auto_tramo(
            mi_ciudad.largo_total/2, d_auto, 31, 3.5, 5, 2, 
            mi_ciudad.largo_total, num_pistas, 0.8, 2
        )
        
        t_b, f_b = demora_bici_tramo(
            mi_ciudad.largo_total/2, cap_bici, d_bici, 14, 
            mi_ciudad.largo_total, 0.5, 2, 0
        )
        
        t_ac, t_esp, t_v, t_tot, f_op, carga, est_x = oferta_tren(
            d_metro, mi_ciudad.largo_total, mi_ciudad.largo_total/2, 35, 
            cap_tren, num_estaciones, 4.8, 6, 10, frec_max
        )
        
        nuevos = {}
        for i in range(n_celdas):
            nuevos[i] = {
                'Auto_Total': t_a[i], 'Bici_Total': t_b[i],
                'Tren_Acceso': t_ac[i], 'Tren_Espera': t_esp[i], 'Tren_Viaje_En_Vehiculo': t_v[i]
            }
            
        if tiempos_actuales is None:
            tiempos_actuales = nuevos
        else:
            f = 1.0 / (it + 1)
            for i in range(n_celdas):
                for k in nuevos[i]:
                    tiempos_actuales[i][k] = (f * nuevos[i][k]) + ((1-f) * tiempos_actuales[i][k])
                    
        if it == MAX_ITER - 1:
            final_data = {
                't_auto': t_a, 't_metro': t_tot, 't_bici': t_b, 
                'f_auto': f_a, 'carga_metro': carga, 'estaciones': est_x,
                't_espera': t_esp,
                'bpr_params': {'C': cap_a, 'a': alp_a, 'b': bet_a, 'VL': vl_a}
            }
            frec_operativa_final = f_op
            
        progress.progress((it+1)/MAX_ITER)
    
    status.success("Equilibrio Alcanzado")
    
    #  Resultados Generales
    tot_viajes = sum(conteo.values())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Población Total", f"{tot_viajes:,}")
    c2.metric("Partición Metro", f"{conteo['Metro']/tot_viajes:.1%}")
    c3.metric("Partición Auto", f"{conteo['Auto']/tot_viajes:.1%}")
    c4.metric("Frecuencia", f"{frec_operativa_final:.1f} trenes/h")
    
    # --- PROCESAMIENTO DE DATOS INDIVIDUALES ---
    df_pob = pd.DataFrame(poblacion)
    df_pob['Origen_Km'] = df_pob['celda_origen'] * mi_ciudad.ancho_celda

    # Calculamos el tiempo específico para cada usuario según su elección
    # (Vectorizado para mayor velocidad)
    df_pob['Tiempo_Viaje_Min'] = 0.0
    
    # Extraemos arrays de tiempos finales
    t_auto_arr = final_data['t_auto']
    t_metro_arr = final_data['t_metro']
    t_bici_arr = final_data['t_bici']
    
    # Calculamos caminata (que es constante por distancia)
    dist_cbd_km = np.abs(np.arange(n_celdas) - mi_ciudad.cbd_index) * mi_ciudad.ancho_celda
    t_walk_arr = (dist_cbd_km / 4.8) * 60
    
    # Asignamos usando máscaras booleanas (mucho más rápido que apply)
    mask_auto = df_pob['Modo'] == 'Auto'
    df_pob.loc[mask_auto, 'Tiempo_Viaje_Min'] = t_auto_arr[df_pob.loc[mask_auto, 'celda_origen']]
    
    mask_metro = df_pob['Modo'] == 'Metro'
    df_pob.loc[mask_metro, 'Tiempo_Viaje_Min'] = t_metro_arr[df_pob.loc[mask_metro, 'celda_origen']]
    
    mask_bici = df_pob['Modo'] == 'Bici'
    df_pob.loc[mask_bici, 'Tiempo_Viaje_Min'] = t_bici_arr[df_pob.loc[mask_bici, 'celda_origen']]
    
    mask_walk = df_pob['Modo'] == 'Caminata'
    df_pob.loc[mask_walk, 'Tiempo_Viaje_Min'] = t_walk_arr[df_pob.loc[mask_walk, 'celda_origen']]

    # PESTAÑAS
    t1, t2, t3, t4, t5, t6, t7, t8, t9 = st.tabs([
    "Convergencia", "Tiempos (Curvas)", "Reparto Modal", 
    "Carga Metro", "Estadísticas y Promedios", "Datos Celdas", 
    "Población", "Emisiones", "Distribución Residencial" # <--- NUEVA PESTAÑA
    ])

    with t1:
        st.subheader("Convergencia del Equilibrio")
        st.line_chart(pd.DataFrame(historia))
        
        total_pob = len(df_pob)
        tele_count = len(df_pob[df_pob['Modo'] == 'Teletrabajo'])
        viajes_activos = total_pob - tele_count
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Población Total", f"{total_pob:,}")
        c2.metric("Viajes Físicos", f"{viajes_activos:,}")
        c3.metric("En Teletrabajo", f"{tele_count:,}")

    with t2:
            x_km = np.linspace(0, mi_ciudad.largo_total, n_celdas)
            
            # GRÁFICO 1: Tiempos Totales
            st.markdown("#### 1. Tiempos Totales de Viaje (Por ubicación)")
            fig, ax = plt.subplots(figsize=(8,3))
            
            ax.plot(x_km, final_data['t_auto'], label='Auto', color=PALETA_MODOS["Auto"], linewidth=2)
            ax.plot(x_km, final_data['t_metro'], label='Metro Total', color=PALETA_MODOS["Metro"], linewidth=2)
            ax.plot(x_km, final_data['t_bici'], label='Bici', color=PALETA_MODOS["Bici"], linewidth=2)
            
            # Si t_walk_arr no se calculó antes, lo calculamos aquí por seguridad
            t_walk_local = (np.abs(x_km - mi_ciudad.largo_total/2) / 4.8) * 60
            ax.plot(x_km, t_walk_local, label='Caminata', color=PALETA_MODOS["Caminata"], linestyle='--')
            
            ax.set_xlabel("Origen (km)"); ax.set_ylabel("Minutos")
            ax.set_ylim(0, 120)
            ax.legend(); ax.grid(True, alpha=0.3)
            st.pyplot(fig)

            # GRÁFICO 2: Tiempo de Espera (NUEVO)
            st.markdown("#### 2. Desglose: Tiempo de Espera Metro")
            st.info("Este gráfico aisla el tiempo que los usuarios pasan esperando en el andén (afectado por congestión).")
            
            fig2, ax2 = plt.subplots(figsize=(8,3))
            ax2.plot(x_km, final_data['t_espera'], label='Tiempo Espera', 
                    color=PALETA_MODOS["Metro"], linestyle='-.', linewidth=2)
            
            ax2.fill_between(x_km, final_data['t_espera'], color=PALETA_MODOS["Metro"], alpha=0.1)
            
            ax2.set_xlabel("Origen (km)")
            ax2.set_ylabel("Minutos")
            ax2.set_ylim(bottom=0)
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            st.pyplot(fig2)

    with t3:
            st.subheader("Análisis Detallado de Elección Modal")
            df_viajeros = df_pob[df_pob['Modo'] != 'Teletrabajo']
            
            if not df_viajeros.empty:
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**1. Reparto Modal por Ubicación**")
                    fig1, ax1 = plt.subplots()
                    sns.histplot(data=df_viajeros, x='Origen_Km', hue='Modo', 
                                multiple="stack", ax=ax1, palette=PALETA_MODOS)
                    ax1.set_ylabel("N° de Personas")
                    st.pyplot(fig1)
                
                with c2:
                    st.markdown("**2. Elección por Nivel Socioeconómico**")
                    fig2, ax2 = plt.subplots()
                    df_estrato = df_viajeros.groupby(['estrato', 'Modo']).size().unstack(fill_value=0)
                    df_estrato_pct = df_estrato.div(df_estrato.sum(axis=1), axis=0)
                    colores_estrato = [PALETA_MODOS[col] for col in df_estrato_pct.columns]
                    df_estrato_pct.plot(kind='bar', stacked=True, ax=ax2, color=colores_estrato)
                    ax2.set_ylabel("Proporción de Viajes")
                    st.pyplot(fig2)

                c3, c4 = st.columns(2)
                with c3:
                    st.markdown("**3. Impacto de la Tenencia de Auto**")
                    fig3, ax3 = plt.subplots()
                    df_car = df_viajeros.groupby(['tiene_auto', 'Modo']).size().unstack(fill_value=0)
                    df_car_pct = df_car.div(df_car.sum(axis=1), axis=0)
                    colores_car = [PALETA_MODOS[col] for col in df_car_pct.columns]
                    df_car_pct.plot(kind='bar', stacked=True, ax=ax3, color=colores_car)
                    ax3.set_xticklabels(['Sin Auto', 'Con Auto'], rotation=0)
                    ax3.set_ylabel("Proporción de Viajes")
                    st.pyplot(fig3)
                
                with c4:
                    st.markdown("**4. Dispersión de Utilidades**")
                    fig4, ax4 = plt.subplots()
                    sns.scatterplot(data=df_viajeros.sample(min(800, len(df_viajeros))), 
                                    x='Origen_Km', y='Utilidad_Elegida', hue='Modo', 
                                    ax=ax4, alpha=0.4, palette=PALETA_MODOS, s=15)
                    ax4.set_ylabel("Utilidad")
                    st.pyplot(fig4)
            else:
                st.warning("No hay datos de viajes físicos para mostrar.")

    with t4:
        est = final_data['estaciones']
        x_mid = (est[:-1] + est[1:]) / 2
        carga = final_data['carga_metro']
        if len(x_mid) != len(carga): x_mid = x_mid[:len(carga)]
        
        cap_ofertada = frec_operativa_final * cap_tren
        cap_maxima_sistema = frec_max * cap_tren 
        
        st.markdown("#### 1. Perfil de Carga vs Capacidades")
        fig, ax = plt.subplots(figsize=(8,3))
        ax.bar(x_mid, carga, width=0.5, color=PALETA_MODOS["Metro"], alpha=0.6, label='Demanda (Pax)')
        ax.axhline(y=cap_ofertada, color='orange', linestyle='--', linewidth=2, label=f'Oferta ({frec_operativa_final:.1f} tph)')
        ax.axhline(y=cap_maxima_sistema, color='black', linestyle='-', linewidth=2, label='Cap. Máxima')
        ax.set_xlabel("Ubicación (km)")
        ax.set_ylabel("Pasajeros/hora")
        ax.legend(loc='upper right')
        st.pyplot(fig)

        st.markdown("#### 2. Tiempo de Espera por Origen (Congestión)")
        x_km = np.linspace(0, mi_ciudad.largo_total, n_celdas)
        fig_wait, ax_wait = plt.subplots(figsize=(8,3))
        ax_wait.plot(x_km, final_data['t_espera'], color=PALETA_MODOS["Metro"], linestyle='-.', linewidth=2, label="Tiempo Espera (min)")
        ax_wait.set_xlabel("Ubicación Origen (km)")
        ax_wait.set_ylabel("Minutos")
        ax_wait.grid(True, alpha=0.3); ax_wait.legend()
        st.pyplot(fig_wait)
        
    with t5:
            st.subheader(" Tiempos de Viaje Promedio")
            
            # Filtramos teletrabajo para este análisis
            df_stats = df_pob[df_pob['Modo'] != 'Teletrabajo']
            
            if not df_stats.empty:
                # --- SECCIÓN 1: POR MODO ---
                st.markdown("#### 1. Por Modo de Transporte")
                promedios_modo = df_stats.groupby('Modo')['Tiempo_Viaje_Min'].mean().reset_index()
                promedios_modo = promedios_modo.sort_values('Tiempo_Viaje_Min')
                
                c1, c2 = st.columns([1, 2])
                
                with c1:
                    st.markdown("**Tabla de Promedios**")
                    st.dataframe(
                        promedios_modo.style.format({"Tiempo_Viaje_Min": "{:.1f} min"}),
                        hide_index=True
                    )
                    
                    st.markdown("**Conteo por Estrato**")
                    # Mostramos conteo simple
                    st.dataframe(df_pob.groupby(['estrato', 'Modo']).size().unstack(fill_value=0))

                with c2:
                    fig5, ax5 = plt.subplots(figsize=(6, 3))
                    sns.barplot(data=promedios_modo, x='Modo', y='Tiempo_Viaje_Min', 
                                palette=PALETA_MODOS, ax=ax5, edgecolor="black")
                    
                    # Etiquetas
                    for p in ax5.patches:
                        ax5.annotate(f'{p.get_height():.1f} min', 
                                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                                    ha='center', va='center', xytext=(0, 5), textcoords='offset points')
                                    
                    ax5.set_ylabel("Tiempo Promedio (min)")
                    ax5.set_ylim(0, promedios_modo['Tiempo_Viaje_Min'].max() * 1.2)
                    ax5.grid(axis='y', alpha=0.3)
                    st.pyplot(fig5)
                
                st.divider()

                # --- SECCIÓN 2: POR ESTRATO (CORREGIDO) ---
                st.markdown("#### 2. Por Estrato Socioeconómico")
                
                # Agrupamos por estrato
                promedios_estrato = df_stats.groupby('estrato')['Tiempo_Viaje_Min'].mean().reset_index()
                
                # CORRECCIÓN: Definimos las llaves como strings ('1', '2', '3') para evitar el ValueError
                # También incluimos los enteros por seguridad.
                colores_estratos = {
                    1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c",
                    "1": "#1f77b4", "2": "#ff7f0e", "3": "#2ca02c"
                }
                
                c3, c4 = st.columns([1, 2])
                
                with c3:
                    st.markdown("**Tabla por Estrato**")
                    df_tabla_estrato = promedios_estrato.copy()
                    # Mapeo para mostrar nombres en la tabla
                    df_tabla_estrato['Estrato'] = df_tabla_estrato['estrato'].map({1: "Alto", 2: "Medio", 3: "Bajo"})
                    
                    st.dataframe(
                        df_tabla_estrato[['Estrato', 'Tiempo_Viaje_Min']].style.format({"Tiempo_Viaje_Min": "{:.1f} min"}),
                        hide_index=True
                    )

                with c4:
                    fig6, ax6 = plt.subplots(figsize=(6, 3))
                    
                    # Aseguramos que la columna sea del tipo que espera el diccionario (string para el gráfico)
                    promedios_estrato['estrato_str'] = promedios_estrato['estrato'].astype(str)
                    
                    sns.barplot(data=promedios_estrato, x='estrato_str', y='Tiempo_Viaje_Min', 
                                palette=colores_estratos, ax=ax6, edgecolor="black")
                    
                    for p in ax6.patches:
                        ax6.annotate(f'{p.get_height():.1f} min', 
                                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                                    ha='center', va='center', xytext=(0, 5), textcoords='offset points')
                    
                    ax6.set_ylabel("Tiempo Promedio (min)")
                    ax6.set_xlabel("Estrato")
                    ax6.set_xticklabels(["1 (Alto)", "2 (Medio)", "3 (Bajo)"])
                    ax6.set_ylim(0, promedios_estrato['Tiempo_Viaje_Min'].max() * 1.2)
                    ax6.grid(axis='y', alpha=0.3)
                    st.pyplot(fig6)

            else:
                st.info("No hay viajes físicos registrados.")
        
    with t6:
            st.subheader(" Datos Celdas")
            x_km = np.linspace(0, mi_ciudad.largo_total, n_celdas)
            df_resumen_celdas = pd.DataFrame({
                "Ubicacion_KM": x_km,
                "Tiempo_Auto_Min": final_data['t_auto'],
                "Tiempo_Metro_Min": final_data['t_metro']
            })
            st.download_button("Descargar CSV", df_resumen_celdas.to_csv(index=False), "datos_celdas.csv")

    with t7:
            st.subheader("👥 Auditoría")
            st.dataframe(df_pob.head(500), use_container_width=True)
    with t8:
        st.header("Estimación de Emisiones Totales (CO2)")
        st.markdown(r"""
        **Metodología:**
        * 🚗 **Auto:** Emisión dinámica ($FE = 2467.4 \cdot v^{-0.699}$) calculada tramo a tramo según la congestión local.
        * 🚇 **Metro:** Emisión lineal basada en distancia recorrida por pasajero (Factor estático).
        """)

        if 'f_auto' in final_data and 'carga_metro' in final_data:
            # === 1. CÁLCULOS BASADOS EN POBLACIÓN (La manera elegante) ===
            # Recuperamos factores fijos
            f_metro_kg_km = config_sim["globales"]["factor_emision_metro"]

            # Acumuladores
            co2_metro_tot_kg = 0
            dist_auto_tot_km = 0
            dist_metro_tot_km = 0

            # Iteramos una sola vez sobre la población para sumar distancias y emisiones fijas
            for p in poblacion:
                # Distancia real de viaje (Manhattan 1D)
                dist_km = abs(p['celda_origen'] - mi_ciudad.cbd_index) * mi_ciudad.ancho_celda

                if p['Modo'] == 'Metro':
                    # Cálculo exacto del segundo código original
                    co2_metro_tot_kg += dist_km * f_metro_kg_km
                    dist_metro_tot_km += dist_km

                elif p['Modo'] == 'Auto':
                    # Solo sumamos distancia aquí, la emisión depende de la velocidad del tramo
                    dist_auto_tot_km += dist_km

            # === 2. CÁLCULOS DINÁMICOS AUTO (Arrays espaciales) ===
            # Necesario para aplicar la fórmula de velocidad v^-0.699 correctamente en cada tramo
            flujos_auto = final_data['f_auto']
            params = final_data['bpr_params']
            CAP, ALF, BET, V_0 = params['C'], params['a'], params['b'], params['VL']
            dx_km = mi_ciudad.largo_total / n_celdas

            # A) Reconstruir velocidad local
            with np.errstate(divide='ignore', invalid='ignore'):
                grado_sat = flujos_auto / CAP
                factor_demora = 1 + ALF * (grado_sat ** BET)
                v_local_kmh = V_0 / factor_demora
            v_local_kmh = np.nan_to_num(v_local_kmh, nan=V_0)

            # B) Calcular emisiones autos espacialmente
            def obtener_factor_auto(v):
                v_clamped = np.clip(v, 1.0, 120.0)
                return 2467.4 * (v_clamped**(-0.699)) # g/km

            factores_auto_g_km = obtener_factor_auto(v_local_kmh)
            emisiones_auto_vec_g = flujos_auto * dx_km * factores_auto_g_km # Vector espacial

            total_auto_kg = np.sum(emisiones_auto_vec_g) / 1000

            carga_metro_tramos = final_data['carga_metro']
            estaciones = final_data['estaciones']
            emisiones_metro_vec_g = np.zeros(n_celdas)
            f_metro_g_pkm = f_metro_kg_km * 1000

            for j in range(len(carga_metro_tramos)):
                start_km = estaciones[j]; end_km = estaciones[j+1]
                idx_start = int((start_km / mi_ciudad.largo_total) * n_celdas)
                idx_end = int((end_km / mi_ciudad.largo_total) * n_celdas)
                val_celda = carga_metro_tramos[j] * f_metro_g_pkm * dx_km
                if idx_end > idx_start:
                    emisiones_metro_vec_g[idx_start:idx_end] = val_celda

            # C) Emisiones Totales por Celda
            emisiones_totales_vec_kg = (emisiones_auto_vec_g + emisiones_metro_vec_g)/1000

            # === 3. RESULTADOS Y KPIs ===
            total_sistema_kg = total_auto_kg + co2_metro_tot_kg
            intensidad_auto = (total_auto_kg * 1000 / dist_auto_tot_km) if dist_auto_tot_km > 0 else 0

            k1, k2, k3 = st.columns(3)
            k1.metric("Total Emisiones", f"{total_sistema_kg:.1f} kg CO2/h")
            k2.metric("Contribución Auto", f"{(total_auto_kg/total_sistema_kg):.1%}" if total_sistema_kg > 0 else "0%")
            k3.metric("Intensidad Promedio Auto", f"{intensidad_auto:.1f} g/km")

            st.divider()

            # === 4. GRÁFICO PERFIL TOTAL (Estilo solicitado) ===
            st.markdown("#### 🏭 Perfil Espacial de Emisiones Totales")
            fig, ax = plt.subplots(figsize=(10, 5))
            x_ax = np.linspace(0, mi_ciudad.largo_total, n_celdas)

            # Área Rellena (Total)
            ax.fill_between(x_ax, 0, emisiones_totales_vec_kg,
                            color='#A52A2A', alpha=0.7, label='Emisiones Totales (Auto + Metro)')

            # Contorno Negro
            ax.plot(x_ax, emisiones_totales_vec_kg, color='black', linewidth=1)

            ax.set_ylabel("Generación de CO2 Total (kg/h)", fontweight='bold')
            ax.set_xlabel("Ubicación (km)")

            # Eje Velocidad (Secundario)
            ax2 = ax.twinx()
            ax2.plot(x_ax, v_local_kmh, color='lime', linestyle='--', linewidth=2, label='Velocidad Autos')
            ax2.set_ylabel('Velocidad Vial (km/h)', color='green')
            ax2.set_ylim(0, V_0 * 1.4)

            # Título y Leyenda
            ax.set_title(f"Perfil de Emisiones - Total: {total_sistema_kg:.1f} kg/h")
            ax.grid(True, alpha=0.3)

            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines + lines2, labels + labels2, loc='upper right')

            st.pyplot(fig)

        else:
            st.warning("⚠️ Ejecuta la simulación para ver el análisis ambiental.")
    with t9:
        st.subheader("🗺️ Distribución Espacial de Hogares por Estrato")
        st.markdown("""
        Este gráfico visualiza el resultado del **Modelo de Localización (Ciudad2)**. 
        Muestra cómo los distintos grupos socioeconómicos compiten por el suelo y se segregan en el espacio.
        """)

        if not df_pob.empty:
            # Configuración de Colores Semánticos para Estratos
            # Estrato 1 (Alto): Azul (Suelen tener alta disponibilidad de pago)
            # Estrato 2 (Medio): Naranja
            # Estrato 3 (Bajo): Verde
            colores_estratos = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c"}
            labels_estratos = {1: "Estrato 1 (Alto)", 2: "Estrato 2 (Medio)", 3: "Estrato 3 (Bajo)"}
            
            # Crear una copia para etiquetar mejor la leyenda
            df_plot = df_pob.copy()
            df_plot['Estrato_Label'] = df_plot['estrato'].map(labels_estratos)

            # --- GRÁFICO PRINCIPAL ---
            fig9, ax9 = plt.subplots(figsize=(10, 5))
            
            # Usamos histplot apilado para ver la densidad en cada km
            sns.histplot(
                data=df_plot, 
                x='Origen_Km', 
                hue='Estrato_Label', 
                palette={"Estrato 1 (Alto)": "#1f77b4", "Estrato 2 (Medio)": "#ff7f0e", "Estrato 3 (Bajo)": "#2ca02c"},
                multiple="stack",
                bins=501,  # 50 barras a lo largo de la ciudad para buena resolución
                edgecolor="white",
                linewidth=0.5,
                alpha=0.8,
                ax=ax9
            )
            
            ax9.set_xlabel("Ubicación en la Ciudad (km) [0 = Periferia Izq, Centro = Mitad, Fin = Periferia Der]")
            ax9.set_ylabel("Cantidad de Hogares")
            ax9.set_title("Perfil de Densidad y Segregación Residencial")
            ax9.grid(True, alpha=0.3, linestyle='--')
            
            st.pyplot(fig9)
            
            st.divider()
            
            # --- MÉTRICAS DE SEGREGACIÓN ---
            st.markdown("#### 📊 Indicadores de Localización")
            
            # Calculamos la ubicación promedio (distancia al centro 0) para ver quién vive más lejos
            centro_ciudad_km = mi_ciudad.largo_total / 2
            df_plot['Distancia_CBD'] = abs(df_plot['Origen_Km'] - centro_ciudad_km)
            
            promedios_dist = df_plot.groupby('estrato')['Distancia_CBD'].mean()
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Distancia Media E1 (Alto)", f"{promedios_dist.get(1, 0):.1f} km del centro")
            c2.metric("Distancia Media E2 (Medio)", f"{promedios_dist.get(2, 0):.1f} km del centro")
            c3.metric("Distancia Media E3 (Bajo)", f"{promedios_dist.get(3, 0):.1f} km del centro")
            
            st.info("Nota: Si las barras de un color se concentran en el centro y otras en los extremos, el modelo está replicando fenómenos de gentrificación o expulsión.")
            
        else:
            st.warning("No hay datos de población para mostrar.")