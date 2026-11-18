# H2 — Cambio de confianza tras respuestas correctas e incorrectas

## Datos y definición

Fuente: `ATC_EXTRACTOR/datos_combinados.xlsx` (hoja `TRIALS`, 1209 filas; 16 participantes con deltas válidos). Se usó el libro consolidado existente; `ATC_EXTRACTOR/unir_excel.py` es el proceso del repo que apila los Excel individuales. No se encontró en el repo otra convención o análisis previo de H2 ni material CRID o plantilla de informe; para recomendar prueba se aplica la regla estándar de Shapiro–Wilk con α = 0,05. La variable principal es `correct`: 1 = respuesta del participante correcta; 0 = incorrecta. `automation_error` se presenta aparte y mide si la sugerencia automática fue errónea.

Se excluyó el bloque manual 1 y se analizaron los bloques automatizados 2–5. Se descartaron pares no consecutivos y los que no tienen `trust` o `automation_error`. `correct = 1` se clasifica como acierto; `correct = 0` como fallo. Si `correct` tiene un valor distinto de 0/1 (incluido timeout o valor ausente), ese ensayo se clasifica como fallo y se conserva si tiene `trust` y `automation_error` válidos, incluso si `answer = no_response`. Los `answer = no_response` binarios se excluyen.

**Convención temporal:** para el trial consecutivo t, `delta_trust = trust_t − trust_(t−1)` y el análisis principal se clasifica con `correct_t`, la respuesta del participante en el ensayo t, después del cual se registró `trust_t`. El análisis complementario se clasifica con `automation_error_t`. El emparejamiento se hace dentro de cada participante, sesión, fase y bloque; nunca cruza límites de bloque.

En los bloques automatizados había 1024 filas: 41 con `no_response`, 0 sin confianza, 0 sin `correct` y 0 sin `automation_error`. Se conservaron 829 deltas válidos, incluidos 75 tras una respuesta incorrecta y 0 por un valor no binario/timeout en `correct` clasificado como fallo. Para la prueba pareada global se incluyeron 16 participantes; 0 no tenían ambas condiciones y se excluyeron de la comparación pareada.

## Descriptivos de cambios por ensayo

| Resultado de la respuesta del participante | N deltas | Media | DE | Mediana | Q1 | Q3 | IQR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Acierto | 754 | 0.021 | 0.786 | 0.000 | 0.000 | 0.000 | 0.000 |
| Fallo | 75 | -0.653 | 1.268 | 0.000 | -1.000 | 0.000 | 1.000 |

## Comparación pareada por participante

Las medias se calcularon por participante y condición agrupando fases y bloques automatizados. La diferencia se definió como media tras acierto menos media tras fallo.

| Medida | Resultado |
|---|---:|
| N participantes pareables | 16 |
| Media del delta tras acierto (DE) | 0.023 (0.062) |
| Media del delta tras fallo (DE) | -0.828 (0.900) |
| Diferencia por participante (media; DE) | 0.850; 0.924 |
| Diferencia por participante (mediana; IQR) | 0.670; 1.177 |
| Shapiro–Wilk de las diferencias | W = 0.853; p = 0,015 |
| t de Student pareada | t(15) = 3.680; p = 0,002 |
| IC 95 % de la diferencia | [0.358, 1.343] |
| Cohen d<sub>z</sub> | 0.920 |
| Wilcoxon de rangos signados | W = 1.000; p = 0,001 |
| Correlación rank-biserial emparejada | 0.981 |
| Prueba recomendada por Shapiro–Wilk | Wilcoxon (no paramétrico; Shapiro-Wilk p < 0,05) |

**Interpretación:** Se incluyeron 16 participantes con ambas condiciones. La diferencia media por participante (acierto − fallo) fue 0.850 (DE = 0.924; IC 95 % [0.358, 1.343]). La t pareada dio t(15) = 3.680, p = 0,002, d<sub>z</sub> = 0.920; Wilcoxon W = 1.000, p = 0,001, r<sub>rb</sub> = 0.981. Shapiro–Wilk: W = 0.853, p = 0,015. Según el criterio de normalidad (α = 0,05), se recomienda como prueba principal: **Wilcoxon (no paramétrico; Shapiro-Wilk p < 0,05)**. En promedio, la confianza descendió tras respuestas incorrectas y permaneció prácticamente estable tras respuestas correctas; por tanto, los datos respaldan que el cambio es más positivo tras un acierto que tras un fallo, pero no muestran un aumento claro tras los aciertos.

## Análisis complementario con `automation_error`

Este análisis responde a otra pregunta: cómo cambia la confianza tras una sugerencia automática correcta o errónea. Con `automation_error`, la diferencia media acierto − fallo fue 0.826 puntos (n = 16; t(15) = 5.192, p < 0,001; Wilcoxon p < 0,001). Hubo 109 deltas tras una sugerencia errónea, 0 fuera del bloque 4. La tabla por fase y bloque para esta variable también se guarda en el libro Excel.

## Exploratorio por fase y bloque

Las comparaciones por celda son exploratorias; los valores p no se ajustaron por comparaciones múltiples.

| Fase | Bloque | Pares | Δ tras acierto (media) | Δ tras fallo (media) | Diferencia media | t | p t | W | p Wilcoxon |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ECOLOGICAL | 2 | 0 | — | — | — | — | — | — | — |
| ECOLOGICAL | 3 | 5 | 0.150 | -0.800 | 0.950 | 1.205 | = 0,295 | 2.000 | = 0,500 |
| ECOLOGICAL | 4 | 12 | -0.153 | -0.958 | 0.806 | 1.795 | = 0,100 | 12.500 | = 0,141 |
| ECOLOGICAL | 5 | 7 | 0.310 | -0.714 | 1.024 | 1.904 | = 0,106 | 1.000 | = 0,125 |
| NEUTRAL | 2 | 2 | 0.146 | -1.500 | 1.646 | 1.082 | = 0,475 | 0.000 | = 0,500 |
| NEUTRAL | 3 | 1 | 0.125 | 0.000 | 0.125 | — | — | 0.000 | = 1,000 |
| NEUTRAL | 4 | 15 | -0.165 | -0.644 | 0.479 | 1.712 | = 0,109 | 23.000 | = 0,209 |
| NEUTRAL | 5 | 6 | 0.146 | -0.833 | 0.979 | 1.835 | = 0,126 | 1.000 | = 0,125 |

## Figuras

![Medias pareadas y distribución por participante](h2_paired_change.png)

![Histograma y gráfico Q–Q de las diferencias](h2_normality_diagnostics.png)

## Archivos de apoyo

- `h2_results.xlsx`: resultados principales tabulares en hojas que usan `correct` y análisis complementarios de `automation_error` en las hojas `sens_*`.
