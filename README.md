# Proyecto #2 - Paint the Fence

Javier Eduardo España Pacheco #23361 - [Video](https://youtu.be/IIzBpsYeb2Q)
Angel Esteban Esquit Hernández #23221 - [Video](https://youtu.be/Bq5J55khO2I)
Roberto José Barreda Siekavizza #23354 - [Video](https://www.youtube.com/watch?v=Xak9TT8B8m8)

**Repositorio:** [https://github.com/Javier-Espana/PRY2-ADA](https://github.com/Javier-Espana/PRY2-ADA)

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
