# Proyecto #2 - Paint the Fence

Javier Eduardo España Pacheco #23361
Angel Esteban Esquit Hernández #23221
Roberto José Barreda Siekavizza #23354

Implementacion en Python para comparar dos enfoques sobre el problema de cobertura de intervalos con costo minimo:

## Estructura

- `PaintTheFence.py`: modelos y algoritmos.
- `scripts/Benchmark.py`: generacion de datos, ajuste polinomial y artefactos de resultados.
- `tests/TestPaintTheFence.py`: pruebas basicas de consistencia.
- `report/Main.tex`: informe en LaTeX.
- `results/`: CSV, figura y fragmentos LaTeX generados por el benchmark.

## Ejecucion

Instalar dependencias:

```bash
python3 -m pip install -r requirements.txt
```

Ejecutar pruebas:

```bash
python3 -m unittest discover -s tests -p "Test*.py"
```

Generar resultados empiricos:

```bash
python3 scripts/Benchmark.py
```

Compilar el informe:

```bash
cd report
pdflatex Main.tex
pdflatex Main.tex
```
