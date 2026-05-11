# Proyecto #2 - Paint the Fence

Implementacion en Python para comparar dos enfoques sobre el problema de cobertura de intervalos con costo minimo:

- solucion exacta con programacion dinamica pseudo-polynomial;
- heuristica greedy basada en eficiencia cobertura/costo.

## Estructura

- `paint_the_fence.py`: modelos y algoritmos.
- `scripts/benchmark.py`: generacion de datos, ajuste polinomial y artefactos de resultados.
- `tests/test_paint_the_fence.py`: pruebas basicas de consistencia.
- `report/main.tex`: informe en LaTeX.
- `results/`: CSV, figura y fragmentos LaTeX generados por el benchmark.

## Ejecucion

Instalar dependencias:

```bash
python3 -m pip install -r requirements.txt
```

Ejecutar pruebas:

```bash
python3 -m unittest discover -s tests
```

Generar resultados empiricos:

```bash
python3 scripts/benchmark.py
```

Compilar el informe:

```bash
cd report
pdflatex main.tex
pdflatex main.tex
```
