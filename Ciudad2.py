import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from itertools import combinations
import random
from scipy.special import logsumexp

"""
# USO:
# reemplazar :
# mi_ciudad = generar_ciudad(...)
# poblacion = mi_ciudad.generar_poblacion_completa(mi_ciudad, CONFIG_DEMANDA) con:

from Ciudad2 import *

mi_ciudad = Ciudad(L=1001, CBD = 501, H = [33300, 33300, 33300], y = [100.0, 20.0, 4.0],  ancho_celda=0.01)
# los parámetros del constructor se pueden ajustar

# Generar lista de diccionarios (lista de hogares/usuarios)
poblacion = mi_ciudad.generar_poblacion_completa(CONFIG_DEMANDA) # (en teoría el mismo config, tiene que estar en algun lado)
# usar poblacion como se usaría normalmente


## Extras:

# función  para actualizar, se pueden modificar varios parámetros y recalcula la asignación
# T = construir_T_desde_csv("Ciudad\Suelo\microdatos_individuos.csv", len(ciudad.H), ciudad.L)
ciudad.actualizar(T, alpha=[1,1,1])  # con un T: (H,I) funcion de penalizacion de transporte 


# tambien viene con un coso para plotear distribuciones
ciudad.dibujar_hogares()
ciudad.plot_stratum_composition_by_parcel()
"""
CONFIG_DEMANDA = {
    "globales": {
        "v_auto": 31, "v_metro": 35, "v_bici": 14, "v_caminata": 4.8,
        "costo_combustible_km" : 120, "costo_tarifa_metro": 800, "costo_parking": 6000 #CLP
    },
    "estratos": {
        1: { # ESTRATO ALTO
            "prob_teletrabajo": 0.40, "prob_jornada_flexible": 0.50, "prob_part_time": 0.05,
            "jornada": {"horas_rigido": 9.0, "horas_flexible": 8.0, "horas_part_time": 4.0},
            "betas": {
                "asc_auto": 1.5, "asc_metro": -0.2, "asc_bici": -0.9, "asc_caminata": -0.5,
                "b_tiempo_viaje": -0.055,
                "b_tiempo_espera": -0.05,
                "b_tiempo_caminata": -0.15,
                "b_costo": -0.00008,
                "penalizaciones_fisicas": {
                    "bici_10": -0.09, "bici_20": -0.15, "bici_30": -0.5,
                    "walk_5": -0.09,  "walk_15": -0.18, "walk_25": -0.4
                }
            }
        },
        2: { # ESTRATO MEDIO
            "prob_teletrabajo": 0.20, "prob_jornada_flexible": 0.30, "prob_part_time": 0.10,
            "jornada": {"horas_rigido": 9.0, "horas_flexible": 8.5, "horas_part_time": 4.5},
            "betas": {
                "asc_auto": 0.7889, "asc_metro": 0.1040, "asc_bici": -0.6818, "asc_caminata": 0.1,
                "b_tiempo_viaje": -0.0331,
                "b_tiempo_espera": -0.0243,
                "b_tiempo_caminata": -0.0440,
                "b_costo": -0.0002,
                "penalizaciones_fisicas": {
                    "bici_10": -0.0634, "bici_20": -0.1, "bici_30": -0.4,
                    "walk_5": -0.05,  "walk_15": -0.09, "walk_25": -0.2
                }
            }
        },
        3: { # ESTRATO BAJO
            "prob_teletrabajo": 0.05, "prob_jornada_flexible": 0.10, "prob_part_time": 0.15,
            "jornada": {"horas_rigido": 9.5, "horas_flexible": 9.0, "horas_part_time": 5.0},
            "betas": {
                "asc_auto": 0.2, "asc_metro": 0.25, "asc_bici": -0.4, "asc_caminata": 0.4,
                "b_tiempo_viaje": -0.0150,
                "b_tiempo_espera": -0.0150,
                "b_tiempo_caminata": -0.0250,
                "b_costo": -0.0006,
                "penalizaciones_fisicas": {
                    "bici_10": -0.0300, "bici_20": -0.0500, "bici_30": -0.7,
                    "walk_5": -0.0250,  "walk_15": -0.0400, "walk_25": -0.08
                }
            }
        }
    }
}

class Ciudad():
    """
    Clase que almacena la info. de una ciudad entera para la simulación
    La ciudad tiene un tamaño L, que corresponden a las "parcelas" de terreno.
    Un distrito de negocios central (CBD)
    Una lista de parcelas, cada una puede contener una cantidad de casas, dado por generar_oferta_normal(...)
    cada casa se identifica por un estrato (1, 2 o 3)
    cada estrato puede tener comportamientos distintos dependiendo de los valores vectoriales
    alpha, 
    la oferta de hogares se genera al construir el objeto
    """
    def __init__(self, L=1001, CBD = 501, H = [33300, 33300, 33300], y = [100.0, 20.0, 4.0],  ancho_celda=0.01, lambda_h=[1,1,1], alpha=[1.3,1.2,1.1], rho=[1,1,1]):
        # Parámetros (input)
        self.L = L; """Cantidad de Parcelas"""
        self.parcelas: list[list[int]] = [[] for i in range(L)] # una lista de listas, cada lsita interior representa las casas en el terreno
        self.cbd_index = int(CBD)
        self.H: list[int] = np.array(H) 
        self.y = np.array(y) # ingresos

        # Atributos necesarios para interacción con demanda
        self.n_celdas = L
        self.ancho_celda = ancho_celda
        self.largo_total = L * ancho_celda
        self.total_households = self.H.sum()

        # inicializacion de la ciudad
        self.T = [[abs(i - CBD) for i in range(self.L)] for _ in range(len(self.H))]
        # T[h, i] fn de transporte para cada estrato y parcela

        self.S = self.generar_oferta_normal(L,self.total_households, self.cbd_index)
        # S[i] fn de oferta, de hogares disponibles en cada terreno

        self.u = np.zeros(len(H))
        self.p = np.zeros(self.total_households)
        self.Q = np.zeros((len(H), L))

        print("Ciudad generada, resolviendo equilibrio")
        self.resolver_equilibrio_logit(lambda_h, alpha, rho)
        print("Asignando hogares")
        self.asignar_hogares_simple()

    def generar_poblacion_completa(self, config = CONFIG_DEMANDA):
        """ Entrega el estado actual de la población y la ciudad como un diccionario para ser usado en el módulo demanda
        Según la config"""
        poblacion = []
        id_counter = 1

        print(f"Generando población completa para {self.L} celdas...")

        for i, t in enumerate(self.parcelas):
            for h in t:
                estrato = h

                #Teletrabajo
                prob_tele = config["estratos"][estrato]["prob_teletrabajo"]
                teletrabaja = random.random() < prob_tele

                #Flexible
                prob_flex = config["estratos"][estrato]["prob_jornada_flexible"]
                es_flexible = random.random() < prob_flex

                #Calcular Entrada 
                minuto_entrada = asignar_horario_entrada_discreto(estrato, config, intervalo=15)
                # -----------------------

                #Calcular Duración y Salida
                duracion_min, tipo_jornada = calcular_duracion_jornada(estrato, es_flexible, config)
                minuto_salida = minuto_entrada + duracion_min
                
            
                usuario = {
                    "id_unico": id_counter,
                    "celda_origen": i,
                    "estrato": estrato,
                    "teletrabaja": teletrabaja,
                    "es_flexible": es_flexible,
                    "tipo_jornada": tipo_jornada,
                    "hora_entrada": formato_hora(minuto_entrada),
                    "hora_salida": formato_hora(minuto_salida),
                    "duracion_horas": duracion_min / 60,
                    "min_entrada": minuto_entrada,
                    "min_salida": minuto_salida
                }
                poblacion.append(usuario)
                id_counter += 1

        return poblacion        

    def generar_oferta_normal(self, I, N, CBD, stdv=None) -> list[int]:
        """
        Genera uan oferta posible de capacidades por terreno para la ciudad de los parámetros 
        como una "normal" discreta que excluye el CBD.
        
        :param I: Cantidad de parcelas o terrenos de la ciudad
        :param N: Cantidad de habitantes totales
        :param CBD: Posición del centro donde no se construiran terrenos
        :para
        :return: S[i] vector de capacidades por terreno (cantidad de hogares)
        :rtype: np.array[int]
        """
        if stdv is None:
            stdv = min(CBD, I - 1 - CBD) / 2

        S = np.zeros(I, dtype=int)

        # muestreamos en el intervalo sin CBD
        samples = np.random.normal(loc=CBD, scale=stdv, size=N)

        for s in samples:
            i = int(np.round(s))

            #saltar cbd sin distorcionar (mucho)
            if i >= CBD:
                i += 1
            # reflejar en los bordes
            if i < 0:
                i = -i
            if i >= I:
                i = 2*(I-1) - i
            S[i] += 1

        self.S = S
        return S

    def resolver_equilibrio_logit(
            self,
            lambda_h, 
            alpha,
            rho,
            T = None, 
            beta=1.0, 
            tol= 1e-8,
            max_iter = 10000): # Delicadisimo
        """
        Resuelve utilidades normalizadas u barra según ecuación (5.4) de Fco. Martínez Microeconomic modelling in urban science 
        :param lambda_h: Marginal utility income, que tanto importa el dinero para la utilidad.
        Regula la disposición de cada hogar de gastar dinero.
        :param alpha: Factor que multiplica la función de transporte
        :param rho: Factor que penaliza la densidad 
        :param T: (H, I) matriz función de costo de transporte, si no se proporciona, se usa penalización lineal.
        """
        if T is None:
            T = np.array(self.T)
        lambda_h = np.asarray(lambda_h)
        alpha = np.asarray(alpha)
        rho = np.asarray(rho)

        # f_h(z_i) / lambda_h
        f_div_lambda = (
            - alpha[:, None] * T
            - rho[:, None] * self.S[None, :]
        )/ lambda_h[:, None]

        logZ = (
            np.log(self.H)[:, None]      # (H,1)
            + beta * (
                self.y[:, None]          # (H,1)
                + f_div_lambda           # (H,I)
            )
        )
        H = len(self.H)
        I = self.L
        self.S = np.asarray(self.S, dtype=float).reshape(-1)
        assert self.S.shape == (self.L,)
        #print(f_div_lambda.shape)
        assert f_div_lambda.shape == (H, I)
        assert logZ.shape == (H, I)


        def F(u_bar):
            """
            Operador de punto fijo según ecuación (5.4)
            usando logsumexp (numéricamente estable)
            """

            # log denom_i = log sum_g H_g exp(beta(y_g + f_gi/lambda_g - u_g))
            log_denom = logsumexp(
                logZ - beta * u_bar[:, None],
                axis=0
            )  # shape (I,)

            # log numerador_h = log sum_i S_i * exp(beta(y_h + f_hi/lambda_h) - log denom_i)
            log_num = (
                beta * (self.y[:, None] + f_div_lambda)
                - log_denom[None, :]
            )

            log_num += np.log(self.S)[None, :]


            u_new = (1 / beta) * logsumexp(log_num, axis=1)

            # normalización (invarianza por traslación)
            u_new -= u_new[0]

            return u_new

 
        # iteración
        u_bar = np.zeros(len(self.H))

        for it in range(max_iter):
            u_new = F(u_bar)
            if np.linalg.norm(u_new - u_bar) < tol:
                print(f"Convergió en {it} iteraciones")
                break
            u_bar = u_new


        # Precios
        log_p = logsumexp(
            np.log(self.H)[:, None]
            + beta * (self.y[:, None] - u_bar[:, None] + f_div_lambda),
            axis=0
        )
        p = log_p / beta


        # Probabilidades

        Q = np.zeros((len(self.H), I))

        for i in range(I):
            log_q = (
                np.log(self.S[i])
                + beta * (self.y + f_div_lambda[:, i] - u_bar - p[i])
            )

            Q[:, i] = np.exp(log_q - logsumexp(log_q))

        self.u = u_bar
        self.p = p
        self.Q = Q
        return

    def resolver_equilibrio_frechet(
            self,
            lambda_h,
            alpha,
            rho,
            T=None,
            beta=1.0,
            tol=1e-8,
            max_iter=10000):

        """
        Versión Frechét del equilibrio bid-auction.
        Misma estructura que resolver_equilibrio_logit.
        """

        if T is None:
            T = np.asarray(self.T)

        lambda_h = np.asarray(lambda_h)
        alpha = np.asarray(alpha)
        rho = np.asarray(rho)

        H = len(self.H)
        I = self.L

        self.S = np.asarray(self.S, dtype=float).reshape(-1)

        # ---------------------------------
        # f_h(i) / lambda_h
        # ---------------------------------

        f_div_lambda = (
            - alpha[:, None] * T
            - rho[:, None] * self.S[None, :]
        ) / lambda_h[:, None]

        # log w_hi  (willingness to pay en log)
        logw = self.y[:, None] + f_div_lambda

        # log Z estructural
        logZ = np.log(self.H)[:, None] + beta * logw

        # ---------------------------------
        # Operador de punto fijo
        # ---------------------------------

        def F(u_bar):

            # log denom_i = log sum_g H_g exp(beta(logw_gi - u_g))
            log_denom = logsumexp(
                logZ - beta * u_bar[:, None],
                axis=0
            )

            # log numerador_h
            log_num = (
                np.log(self.S)[None, :]
                + beta * logw
                - log_denom[None, :]
            )

            u_new = (1 / beta) * logsumexp(log_num, axis=1)

            u_new -= u_new[0]

            return u_new

        # ---------------------------------
        # Iteración
        # ---------------------------------

        u_bar = np.zeros(H)

        for it in range(max_iter):
            u_new = F(u_bar)

            if np.linalg.norm(u_new - u_bar) < tol:
                print(f"Convergió en {it} iteraciones")
                break

            u_bar = u_new

        # ---------------------------------
        # Precios Frechét
        # ---------------------------------

        log_p = logsumexp(
            np.log(self.H)[:, None]
            + beta * (logw - u_bar[:, None]),
            axis=0
        )

        p = log_p / beta

        # ---------------------------------
        # Probabilidades Frechét
        # ---------------------------------

        Q = np.zeros((H, I))

        for i in range(I):

            log_q = (
                np.log(self.S[i])
                + beta * (logw[:, i] - u_bar - p[i])
            )

            Q[:, i] = np.exp(log_q - logsumexp(log_q))

        # ---------------------------------
        # Guardar
        # ---------------------------------

        self.u = u_bar
        self.p = p
        self.Q = Q

        return u_bar, p, Q



    def asignar_hogares_simple(self):
        print("Asignando hogares simple (barrido por rondas)")

        num_estratos, num_parcelas = self.Q.shape

        try:
            assert sum(self.S) == sum(self.H)
            assert len(self.S) == num_parcelas
            assert len(self.H) == num_estratos
        except AssertionError as e:
            print("Los parámetros no calzan entre sí")
            print(e)
            return

        S_rest = np.asarray(self.S, dtype=int).copy()
        H_rest = np.asarray(self.H, dtype=int).copy()

        parcelas = [[] for _ in range(num_parcelas)]

        # Mientras queden espacios en alguna parcela
        while S_rest.sum() > 0:

            # Recorremos parcelas de izquierda a derecha
            for i in range(num_parcelas):

                if S_rest[i] == 0:
                    continue  # ya está llena

                pesos = self.Q[:, i].copy()

                # bloquear estratos sin hogares
                pesos[H_rest == 0] = 0.0

                masa = pesos.sum()
                if masa == 0:
                    print("Error: no hay hogares disponibles para asignar")
                    return

                probs = pesos / masa
                h = np.random.choice(num_estratos, p=probs)

                parcelas[i].append(int(h) + 1)
                H_rest[h] -= 1
                S_rest[i] -= 1

        self.parcelas = parcelas


    def actualizar(self, T= None, lambda_h=[1,1,1], alpha=[1.3,1.2,1.1], rho=[1,1,1]):
        """ Actualiza la fn de transporte y recalcula asignaciones.
        adicionalmente se pueden modificar los valores de la función de amenity"""
        if T is not None: self.T = np.array(T)
        self.resolver_equilibrio_logit(lambda_h, alpha, rho, T)
        self.asignar_hogares_simple()


    def dibujar_hogares(self):
        """Plotea el estado actual de la ciudad y sus parcelas como un barplot
        adicionalmente se"""
        num_parcelas = len(self.parcelas)

        conteos = np.zeros((len(self.H), num_parcelas))

        for i, parcela in enumerate(self.parcelas):
            for h in parcela:
                conteos[h-1, i] += 1 # el h es estrato y ala vez indice, hay problemas tontos como este -1 por eso

        bottom = np.zeros(num_parcelas)

        colores = ["tab:blue", "tab:orange", "tab:green"]
        etiquetas = [f"Estrato {h+1}" for h in range(len(self.H))]

        plt.figure(figsize=(12, 4))

        for h in range(len(self.H)):
            plt.bar(
                range(num_parcelas),
                conteos[h],
                bottom=bottom,
                color=colores[h],
                label=etiquetas[h]
            )
            bottom += conteos[h]

        plt.xlabel("Parcela")
        plt.ylabel("Número de hogares")
        plt.title("Asignación espacial de hogares por estrato")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_stratum_composition_by_parcel(self, labels=None):
        """
        Q[h, i]: matriz de probabilidades
        Grafica barras apiladas:
            - una barra por parcela
            - colores = estratos
            - altura = proporción del estrato en la parcela
        """
        Q = np.asarray(self.Q)
        H, I = Q.shape

        # composición relativa por parcela
        col_sum = Q.sum(axis=0, keepdims=True)
        col_sum[col_sum == 0] = 1.0  # seguridad numérica
        P = Q / col_sum

        x = np.arange(I)
        bottom = np.zeros(I)

        plt.figure(figsize=(10, 4))
        print("iniciando graficación de ciudad")
        for h in range(H):
            plt.bar(
                x,
                P[h],
                bottom=bottom,
                label=labels[h] if labels else f"Estrato {h+1}"
            )
            bottom += P[h]

        plt.xlabel("Parcela i")
        plt.ylabel("Proporción")
        plt.title("Composición por estrato en cada parcela")
        plt.legend()
        plt.tight_layout()
        plt.show()

# =================================
# Funciones Auxiliares de los pibes
# =================================


def formato_hora(minutos): #minutos a horas
    h = int(minutos // 60) % 24
    m = int(minutos % 60)
    return f"{h:02d}:{m:02d}"

def redondear_horario(minutos_reales, intervalo): #redondeo a intervalo
    if intervalo <= 0: return int(minutos_reales)
    bloques = round(minutos_reales / intervalo)
    return int(bloques * intervalo)

def calcular_duracion_jornada(estrato, es_flexible, config):
    params = config["estratos"][estrato]
    params_j = params["jornada"]

    es_part_time = random.random() < params["prob_part_time"]

    if es_part_time:
        return int(params_j["horas_part_time"] * 60), "Part-Time"
    elif es_flexible:
        return int(params_j["horas_flexible"] * 60), "Flexible"
    else:
        return int(params_j["horas_rigido"] * 60), "Rígido"

import numpy as np
import pandas as pd

def construir_T_desde_csv(
    path_csv: str,
    H: int,
    I: int,
    col_estrato="Estrato",
    col_parcela="Celda_Origen",
    col_logsum="LogSuma_Utilidad"
):
    import numpy as np
    import pandas as pd

    df = pd.read_csv(path_csv)

    df = df[[col_estrato, col_parcela, col_logsum]].copy()

    # --- FIX ESTRATO TEXTO ---
    df[col_estrato] = (
        df[col_estrato]
        .astype(str)
        .str.extract(r"(\d+)")
        .astype(int)
        - 1
    )

    # parcela a entero
    df[col_parcela] = df[col_parcela].astype(int)

    # promedio por (h, i)
    grouped = (
        df
        .groupby([col_estrato, col_parcela])[col_logsum]
        .mean()
    )

    T = np.full((H, I), -np.inf)

    for (h, i), val in grouped.items():
        if 0 <= h < H and 0 <= i < I:
            T[h, i] = val

    return T


def asignar_horario_entrada_discreto(estrato, config, intervalo=15):
    """Calcula la hora de entrada basada en probabilidad y estrato"""
    prob_flex = config["estratos"][estrato]["prob_jornada_flexible"]

    #Media y Desviación por estrato(Sigma)
    perfiles = {
        1: {'media': 540, 'sigma_rigido': 20, 'sigma_flex': 60}, # 9:00 AM
        2: {'media': 510, 'sigma_rigido': 15, 'sigma_flex': 40}, # 8:30 AM
        3: {'media': 480, 'sigma_rigido': 10, 'sigma_flex': 20}  # 8:00 AM
    }

    es_flexible = random.random() < prob_flex
    p = perfiles[estrato]
    sigma = p['sigma_flex'] if es_flexible else p['sigma_rigido']

    minuto_continuo = np.random.normal(loc=p['media'], scale=sigma)
    return redondear_horario(minuto_continuo, intervalo)



"""
# Juego de uso
ciudad = Ciudad()
ciudad.dibujar_hogares()
ciudad.plot_stratum_composition_by_parcel()

T = construir_T_desde_csv("Ciudad\Suelo\microdatos_individuos.csv", len(ciudad.H), ciudad.L)

ciudad.actualizar(T, alpha=[1,1,1])
ciudad.plot_stratum_composition_by_parcel()
"""

