# Aplicación web Flask para resolver el Problema de la Bauxita con PuLP
# Laura Sánchez y Luis Herrera

from flask import Flask, request, render_template
from pulp import *

bauxita = Flask(__name__)

@bauxita.route("/", methods=["GET", "POST"])
def modelo_bauxita():
    resultado = None
    funcion_objetivo = None
    plantas_abiertas = {}

    if request.method == "POST":
        # Capturar costos fijos desde el formulario
        try:
            costo_fijo_B = float(request.form["costoB"])
            costo_fijo_C = float(request.form["costoC"])
            costo_fijo_D = float(request.form["costoD"])
            costo_fijo_E = float(request.form["costoE"])
        except ValueError:
            return render_template("bauxita.html", error="Por favor ingrese valores numéricos válidos.")

        # ------------------ MODELO ------------------
        modelo = LpProblem("Problema_Bauxita", LpMinimize)

        # Conjuntos
        MINAS = ["A", "B", "C"]
        PLANTAS = ["B", "C", "D", "E"]
        ESMALTADO = ["D", "E"]

        # Parámetros
        cap_mina = {"A": 36000, "B": 52000, "C": 28000}
        cap_planta = {"B": 40000, "C": 20000, "D": 30000, "E": 80000}
        cap_esmaltado = {"D": 4000, "E": 7000}

        costo_explotacion = {"A": 420, "B": 360, "C": 540}
        costo_fijo = {"B": costo_fijo_B, "C": costo_fijo_C, "D": costo_fijo_D, "E": costo_fijo_E}
        costo_produccion = {"B": 330, "C": 320, "D": 380, "E": 240}
        costo_esmaltado = {"D": 8500, "E": 5200}

        ctran_b = {
            ("A", "B"): 400, ("A", "C"): 2010, ("A", "D"): 510, ("A", "E"): 1920,
            ("B", "B"): 10, ("B", "C"): 630, ("B", "D"): 220, ("B", "E"): 1510,
            ("C", "B"): 1630, ("C", "C"): 10, ("C", "D"): 620, ("C", "E"): 940
        }

        ctran_a = {
            ("B", "D"): 220, ("B", "E"): 1510,
            ("C", "D"): 620, ("C", "E"): 940,
            ("D", "D"): 0, ("D", "E"): 1615,
            ("E", "D"): 1465, ("E", "E"): 0
        }

        demanda = {"D": 1000, "E": 1200}
        rend_bauxita = {"A": 0.06, "B": 0.08, "C": 0.062}
        rend_alumina = 0.4

        # Variables de decisión
        x = LpVariable.dicts("x", (MINAS, PLANTAS), lowBound=0)
        y = LpVariable.dicts("y", (PLANTAS, ESMALTADO), lowBound=0)
        w = LpVariable.dicts("w", PLANTAS, lowBound=0, upBound=1, cat=LpBinary)

        # Función objetivo
        modelo += (
            lpSum(costo_explotacion[i] * x[i][j] for i in MINAS for j in PLANTAS)
            + lpSum(costo_produccion[j] * y[j][k] for j in PLANTAS for k in ESMALTADO)
            + lpSum(costo_esmaltado[k] * y[j][k] for j in PLANTAS for k in ESMALTADO)
            + lpSum(ctran_b[(i, j)] * x[i][j] for i in MINAS for j in PLANTAS)
            + lpSum(ctran_a[(j, k)] * y[j][k] for j in PLANTAS for k in ESMALTADO)
            + lpSum(costo_fijo[j] * w[j] for j in PLANTAS)
        )

        # Restricciones
        for i in MINAS:
            modelo += lpSum(x[i][j] for j in PLANTAS) <= cap_mina[i]

        for j in PLANTAS:
            modelo += lpSum(x[i][j] for i in MINAS) <= cap_planta[j] * w[j]

        for k in ESMALTADO:
            modelo += lpSum(y[j][k] for j in PLANTAS) <= cap_esmaltado[k]

        for k in ESMALTADO:
            modelo += lpSum(y[j][k] for j in PLANTAS) == demanda[k]

        for j in PLANTAS:
            modelo += lpSum(rend_bauxita[i] * x[i][j] for i in MINAS) == lpSum(y[j][k] for k in ESMALTADO)

        # Resolver modelo
        modelo.solve(PULP_CBC_CMD(msg=0))
        estado = LpStatus[modelo.status]
        funcion_objetivo = value(modelo.objective)

        plantas_abiertas = {j: int(w[j].varValue) for j in PLANTAS}

        resultado = f"Estado: {estado}, Costo total: ${funcion_objetivo:,.2f}"

    return render_template("bauxita.html", resultado=resultado,
                           funcion_objetivo=funcion_objetivo,
                           plantas_abiertas=plantas_abiertas)

if __name__ == "__main__":
    bauxita.run(debug=True)
    