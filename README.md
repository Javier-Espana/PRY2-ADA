# Proyecto #2 - Paint the Fence

Javier Eduardo España Pacheco #23361
Angel Esteban Esquit Hernández #23221
Roberto José Barreda Siekavizza #23354

Implementacion en Python para estudiar una formulacion por subsets del problema de cobertura con costo minimo:

## Estructura

- `PaintTheFence.py`: modelos y algoritmos por subsets.
- `scripts/benchmark.py`: generacion de datos, ajuste polinomial y artefactos de resultados para subsets.
- `tests/TestPaintTheFence.py`: pruebas basicas de consistencia.
- `report/main.tex`: informe en LaTeX.
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
python3 scripts/benchmark.py
```

Compilar el informe:

```bash
cd report
pdflatex Main.tex
pdflatex Main.tex
```
