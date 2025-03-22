# CASO DE ESTUDIO - SISTEMAS ENERGÉTICOS 25
!pip install -q pulp
!pip install matplotlib
import pulp
import math

# DATA
P = 7 # number of consumption points
Coord = [[74,34,0],[244,200,55],[195,428,85],[354,342,88],[392,356,94],[431,334,94],[495,272,95]] # point coordinates
L = [[0.0 for p in range(P)] for d in range(P)] # distance between points
for p in range(P):
  for d in range(P):
    L[p][d] = math.sqrt((Coord[p][0]-Coord[d][0])**2+(Coord[p][1]-Coord[d][1])**2+(Coord[p][2]-Coord[d][2])**2)
ED = [300,1000,300,300,1000,300,200] # energy demand
PD = [200,500,200,200,500,200,150] # power demand
AD = 3 # required autonomy
S = 3 # types of panels
ES = [738,2387,2539] # energy of panels
PS = [170,550,585] # power of panels
CS = [142,250,273] # cost of panels
Z = 3 # types of panel controllers
PZ = [1000,10000,25000] # power of panel controllers
CZ = [120,1150,2200] # cost of panel controllers
A = 2 #
EA = [[95,445],[357,1460],[581,2135],[498,1889],[504,1913],[488,1856],[512,1926]]# energy of turbines
PA = [300,1200] # power of turbines
CA = [974,2737] # cost of turbines
R = 2 # types of turbine controllers
PR = [420,1440] # power of turbine controllers
CR = [165,285] # cost of turbine controllers
B = 2 # types of batteries
EB = [5120,10240] # capacity of batteries
CB = [1279,2557] # cost of batteries
I = 2 # types of inverters
PI = [20000,25000] # power of inverters
CI = [2967,3491] # cost of inverters
CC = 2 # cost of wires
CM = 100 # cost of meters
M = 10000000 # very high value

# VARIABLES
xs = pulp.LpVariable.dicts("Panels",(range(P),range(S)),0,None,pulp.LpInteger)
xz = pulp.LpVariable.dicts("Panel controllers",(range(P),range(Z)),0,None,pulp.LpInteger)
xa = pulp.LpVariable.dicts("Turbines",(range(P),range(A)),0,None,pulp.LpInteger)
xr = pulp.LpVariable.dicts("Turbine controllers",(range(P),range(R)),0,None,pulp.LpInteger)
xb = pulp.LpVariable.dicts("Batteries",(range(P),range(B)),0,None,pulp.LpInteger)
xi = pulp.LpVariable.dicts("Inverters",(range(P),range(I)),0,None,pulp.LpInteger)
fe = pulp.LpVariable.dicts("Eflow",(range(P),range(P)),0,None,pulp.LpContinuous)
fp = pulp.LpVariable.dicts("Pflow",(range(P),range(P)),0,None,pulp.LpContinuous)
xg = pulp.LpVariable.dicts("Generators",range(P),0,1,pulp.LpInteger)
xc = pulp.LpVariable.dicts("Lines",(range(P),range(P)),0,1,pulp.LpInteger)
xm = pulp.LpVariable.dicts("Meters",range(P),0,1,pulp.LpInteger)

# OBJECTIVE FUNCTION
model = pulp.LpProblem("System",pulp.LpMinimize)
model += pulp.lpSum(CA[a]*xa[p][a] for p in range(P) for a in range(A)) + pulp.lpSum(CR[r]*xr[p][r] for p in range(P) for r in range(R)) + pulp.lpSum(CS[s]*xs[p][s] for p in range(P) for s in range(S)) + pulp.lpSum(CZ[z]*xz[p][z] for p in range(P) for z in range(Z)) + pulp.lpSum(CB[b]*xb[p][b] for p in range(P) for b in range(B)) + pulp.lpSum(CI[i]*xi[p][i] for p in range(P) for i in range(I)) + pulp.lpSum(CM*xm[p] for p in range(P)) + pulp.lpSum(L[p][d]*CC*xc[p][d] for p in range(P) for d in range(P) if p!=d)

# CONSTRAINTS
for p in range(P):
  model += pulp.lpSum(xs[p][s] for s in range(S)) <= M*xg[p]
for p in range(P):
  model += pulp.lpSum(xa[p][a] for a in range(A)) + pulp.lpSum(xs[p][s] for s in range(S)) >= xg[p]
for p in range(P):
  model += pulp.lpSum(fe[q][p] for q in range(P) if q!=p) + pulp.lpSum(ES[s]*xs[p][s] for s in range(S)) + pulp.lpSum(EA[p][a]*xa[p][a] for a in range(A)) >= ED[p] + pulp.lpSum(fe[p][d] for d in range(P) if d!=p)
for p in range(P):
  model += pulp.lpSum(fp[q][p] for q in range(P) if q!=p) + pulp.lpSum(PI[i]*xi[p][i] for i in range(I)) >= PD[p] + pulp.lpSum(fp[p][d] for d in range(P) if d!=p)
for p in range(P):
  model += pulp.lpSum(EB[b]*xb[p][b] for b in range(B)) + M*(1-xg[p]) >= AD*(ED[p] + pulp.lpSum(fe[p][d] for d in range(P) if d!=p))
for p in range(P):
  for i in range(I):
    xi[p][i] <= M*xg[p]
for p in range(P):
  for d in range(P):
    if p!=d:
      model += fe[p][d] <= M*xc[p][d]
for p in range(P):
  for d in range(P):
    if p!=d:
      model += fp[p][d] <= M*xc[p][d]
for p in range(P):
  model += pulp.lpSum(xc[q][p] for q in range(P) if q!=p) + xg[p] <= 1
for p in range(P):
  model += pulp.lpSum(xa[p][a] for a in range(A)) <= M*xg[p]
for p in range(P):
  model += pulp.lpSum(PR[r]*xr[p][r] for r in range(R)) >= pulp.lpSum(PA[a]*xa[p][a] for a in range(A))
for p in range(P):
  model += pulp.lpSum(PZ[z]*xz[p][z] for z in range(Z)) >= pulp.lpSum(PS[s]*xs[p][s] for s in range(S))
for p in range(P):
  model += pulp.lpSum(xc[p][d] for d in range(P) if d!=p) <= M*xm[p]
for p in range(P):
  model += pulp.lpSum(xc[q][p] for q in range(P) if q!=p) <= xm[p]

solver = pulp.PULP_CBC_CMD(msg=True, warmStart=True)
model.solve(solver)

# PRINT RESULTS
print(model.objective.value())

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Dibuja los puntos
for p in range(P):
    x, y, z = Coord[p]
    ax.scatter(x, y, z, c='blue', s=50)
    ax.text(x, y, z, f'{p}', color='black')

# Dibuja las líneas de conexión
for p in range(P):
    for d in range(P):
        if p != d and pulp.value(xc[p][d]) > 0.5:
            x_vals = [Coord[p][0], Coord[d][0]]
            y_vals = [Coord[p][1], Coord[d][1]]
            z_vals = [Coord[p][2], Coord[d][2]]
            ax.plot(x_vals, y_vals, z_vals, c='red')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Red de conexión óptima')
plt.show()

plt.figure(figsize=(12, 8))

# Dibujamos los nodos
for p in range(P):
    x, y = Coord[p][0], Coord[p][1]
    plt.scatter(x, y, color='black', s=100, label='Nodo' if p == 0 else "")
    plt.text(x + 5, y + 5, f'{p}', fontsize=10)

    # Paneles solares
    for s in range(S):
        cantidad = pulp.value(xs[p][s])
        if cantidad and cantidad > 0:
            plt.scatter(x, y, marker='^', color='orange', s=40 + cantidad*10, label='Panel solar' if p == 0 and s == 0 else "")

    # Turbinas
    for a in range(A):
        cantidad = pulp.value(xa[p][a])
        if cantidad and cantidad > 0:
            plt.scatter(x, y, marker='P', color='green', s=40 + cantidad*10, label='Turbina' if p == 0 and a == 0 else "")

    # Baterías
    for b in range(B):
        cantidad = pulp.value(xb[p][b])
        if cantidad and cantidad > 0:
            plt.scatter(x, y, marker='s', color='blue', s=40 + cantidad*10, label='Batería' if p == 0 and b == 0 else "")

    # Inversores
    for i in range(I):
        cantidad = pulp.value(xi[p][i])
        if cantidad and cantidad > 0:
            plt.scatter(x, y, marker='D', color='red', s=40 + cantidad*10, label='Inversor' if p == 0 and i == 0 else "")

    # Controladores de panel
    for z_ in range(Z):
        cantidad = pulp.value(xz[p][z_])
        if cantidad and cantidad > 0:
            plt.scatter(x, y, marker='*', color='purple', s=40 + cantidad*10, label='Ctrl. panel' if p == 0 and z_ == 0 else "")

    # Controladores de turbina
    for r in range(R):
        cantidad = pulp.value(xr[p][r])
        if cantidad and cantidad > 0:
            plt.scatter(x, y, marker='X', color='brown', s=40 + cantidad*10, label='Ctrl. turbina' if p == 0 and r == 0 else "")

# Dibujamos las conexiones (líneas entre nodos)
for p in range(P):
    for d in range(P):
        if p != d and pulp.value(xc[p][d]) > 0.5:
            x1, y1 = Coord[p][0], Coord[p][1]
            x2, y2 = Coord[d][0], Coord[d][1]
            plt.plot([x1, x2], [y1, y2], 'gray', linestyle='dashed')

plt.title("Distribución de Equipos en la Red (2D)")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
plt.tight_layout()
plt.show()

import numpy as np

# Datos para el gráfico
nodos = list(range(P))
demanda_energia = ED

generacion_panel = [sum(ES[s] * pulp.value(xs[p][s]) for s in range(S)) for p in nodos]
generacion_turbina = [sum(EA[p][a] * pulp.value(xa[p][a]) for a in range(A)) for p in nodos]
generacion_total = [generacion_panel[p] + generacion_turbina[p] for p in nodos]
balance = [generacion_total[p] - ED[p] for p in nodos]

# Gráfico
bar_width = 0.25
index = np.arange(len(nodos))

plt.figure(figsize=(12, 7))

# Demanda
plt.barh(index, demanda_energia, bar_width, color='red', label='Demanda Energía')

# Generación
plt.barh(index + bar_width, generacion_total, bar_width, color='green', label='Generación Total')

# Balance
colors = ['blue' if b >= 0 else 'orange' for b in balance]
plt.barh(index + 2*bar_width, balance, bar_width, color=colors, label='Balance (Gen - Dem)')

# Etiquetas
plt.yticks(index + bar_width, [f'Punto {p}' for p in nodos])
plt.xlabel('Energía [Wh]')
plt.title('Balance Energético por Punto de Consumo')
plt.legend()
plt.grid(True, axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Datos para el gráfico de almacenamiento
energia_en_baterias = [sum(EB[b] * pulp.value(xb[p][b]) for b in range(B)) for p in range(P)]
energia_requerida = [AD * ED[p] for p in range(P)]  # autonomía mínima requerida

index = np.arange(P)
bar_width = 0.35

plt.figure(figsize=(12, 7))

# Energía en baterías
plt.barh(index, energia_en_baterias, bar_width, color='cyan', label='Energía Almacenada (Baterías)')

# Energía requerida para autonomía
plt.barh(index + bar_width, energia_requerida, bar_width, color='gray', label='Energía Requerida (Autonomía)')

# Etiquetas
plt.yticks(index + bar_width / 2, [f'Punto {p}' for p in range(P)])
plt.xlabel('Energía [Wh]')
plt.title('Comparación de Energía Almacenada vs. Requerida por Autonomía')
plt.legend()
plt.grid(True, axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


print("\n--- RESULTADOS DE LAS VARIABLES ---\n")

# Paneles solares por tipo y punto
print("Paneles solares:")
for p in range(P):
    for s in range(S):
        val = pulp.value(xs[p][s])
        if val > 0:
            print(f"  Punto {p}: {int(val)} unidades del tipo {s}")

# Controladores de panel
print("\nControladores de panel:")
for p in range(P):
    for z in range(Z):
        val = pulp.value(xz[p][z])
        if val > 0:
            print(f"  Punto {p}: {int(val)} unidades del tipo {z}")

# Turbinas
print("\nTurbinas:")
for p in range(P):
    for a in range(A):
        val = pulp.value(xa[p][a])
        if val > 0:
            print(f"  Punto {p}: {int(val)} unidades del tipo {a}")

# Controladores de turbina
print("\nControladores de turbina:")
for p in range(P):
    for r in range(R):
        val = pulp.value(xr[p][r])
        if val > 0:
            print(f"  Punto {p}: {int(val)} unidades del tipo {r}")

# Baterías
print("\nBaterías:")
for p in range(P):
    for b in range(B):
        val = pulp.value(xb[p][b])
        if val > 0:
            print(f"  Punto {p}: {int(val)} unidades del tipo {b}")

# Inversores
print("\nInversores:")
for p in range(P):
    for i in range(I):
        val = pulp.value(xi[p][i])
        if val > 0:
            print(f"  Punto {p}: {int(val)} unidades del tipo {i}")

# Generadores activados
print("\nGeneradores activos:")
for p in range(P):
    if pulp.value(xg[p]) > 0.5:
        print(f"  Punto {p}: Activado")

# Medidores instalados
print("\nMedidores instalados:")
for p in range(P):
    if pulp.value(xm[p]) > 0.5:
        print(f"  Punto {p}: Instalado")

# Conexiones activas
print("\nConexiones (líneas eléctricas) activas:")
for p in range(P):
    for d in range(P):
        if p != d and pulp.value(xc[p][d]) > 0.5:
            print(f"  Línea activa: {p} -> {d}")

# Flujo de energía
print("\nFlujo de energía:")
for p in range(P):
    for d in range(P):
        if p != d and pulp.value(fe[p][d]) > 0.01:
            print(f"  {p} -> {d}: {pulp.value(fe[p][d]):.2f} Wh")

# Flujo de potencia
print("\nFlujo de potencia:")
for p in range(P):
    for d in range(P):
        if p != d and pulp.value(fp[p][d]) > 0.01:
            print(f"  {p} -> {d}: {pulp.value(fp[p][d]):.2f} W")

print(f"\nCosto total mínimo: {model.objective.value():,.2f} $")
