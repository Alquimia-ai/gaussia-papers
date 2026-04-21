# Role Adherence — Notas de investigación y evidencia empírica final

> **Estado (2026-04-20):** Evidencia empírica completa. Próxima sesión: discutir los puntos pendientes y luego redactar el white paper.

---

## Pendiente de discusión antes de arrancar el paper

Estos tres puntos necesitan alineación antes de escribir Results/Discussion:

**[D1] Por qué PCA+Mahalanobis invierte la señal**
Con n=15, PCA ajusta 14 componentes sobre los embeddings del ground truth de esos ejemplos específicos. Esas componentes capturan la varianza de esos 14 textos, no la geometría de adherencia al rol. Las violaciones terminan siendo "más cercanas" al centro de referencia en ese espacio comprimido. Hay que decidir: ¿lo reportamos como resultado negativo puro, o profundizamos en la explicación del mecanismo?

**[D2] Separation vs AUC cuentan historias distintas — ¿cuál es la métrica principal del paper?**
BERTScore tiene separation=+0.073 (parece débil) pero AUC=0.995 (parece excelente). Los grupos están muy juntos en promedio pero son perfectamente rankeables. Hay que acordar si AUC es la métrica primaria y separation queda como descriptivo, o si ambas tienen peso igual en el argumento.

**[D3] AUC≈1.0 en BERTScore, Cosine y k-NN es un límite del benchmark**
El benchmark fue diseñado para que los adherentes sean paráfrasis del ground truth y las violaciones sean semánticamente distintas. Cualquier método de similitud semántica separa perfectamente ese benchmark. El resultado útil para el paper son los AUC de Mahalanobis y KL, que no dependen del ground truth per-turno. Hay que decidir qué tan prominente hacemos esta advertencia en el paper — puede debilitar el argumento si no se contextualiza bien.

---

## Observaciones para incluir en el paper

**k-NN es Cosine con otro nombre en este setup.** Con k=1 y un único ground truth por turno, k-NN es matemáticamente equivalente a Cosine. La diferencia sería relevante si el set de referencia tuviera múltiples respuestas válidas por turno — ahí k-NN promedia sobre vecinos y gana robustez. Vale la pena nombrarlo en el paper como la versión escalable de Cosine para sets de referencia más ricos.

**La calidad del judge importa como variable de diseño.** llama-3.3-70b obtuvo F1=0.979 vs claude-haiku F1=0.833 — una diferencia enorme. Reportar resultados de un juez LLM sin especificar el modelo es metodológicamente insuficiente. El modelo del juez es una variable de diseño tan crítica como el threshold de un método determinístico.

---

## Resultados finales del experimento

### Benchmark

- **150 turnos**, estratificados con seed=42
- Clases: 60 adherent (40%), 30 scope_violation, 30 tone_violation, 30 constructive_failure (20% c/u)
- Las violaciones se colapsan en una clase binaria: `1=adherent`, `0=violation`

---

### Tabla 1 — Métodos continuos (separation y AUC)

| Método | Cond. | Adherent | Violation | Sep. | AUC |
|--------|-------|----------|-----------|------|-----|
| BERTScore | n=15 | 0.941 | 0.864 | +0.077 | 1.000 |
| | n=50 | 0.940 | 0.869 | +0.071 | 1.000 |
| | n=150 | 0.941 | 0.868 | +0.073 | 0.995 |
| Cosine | n=15 | 0.905 | 0.466 | +0.439 | 1.000 |
| | n=50 | 0.902 | 0.520 | +0.382 | 0.998 |
| | n=150 | 0.877 | 0.497 | +0.380 | 0.994 |
| Mahalanobis | n=15 | 0.404 | 0.348 | +0.056 | 0.963 |
| | n=50 | 0.391 | 0.348 | +0.043 | 0.872 |
| | n=150 | 0.391 | 0.350 | +0.041 | 0.856 |
| KL Divergence | n=15 | 0.860 | 0.858 | +0.002 | 0.537 |
| | n=50 | 0.693 | 0.671 | +0.022 | 0.870 |
| | n=150 | 0.517 | 0.494 | +0.023 | 0.877 |
| **PCA+Mahalanobis** | n=15 | 0.283 | 0.419 | **−0.136** | **0.000** |
| | n=50 | 0.318 | 0.393 | −0.075 | 0.110 |
| | n=150 | 0.343 | 0.383 | −0.040 | 0.219 |
| k-NN | n=15 | 0.905 | 0.496 | +0.409 | 1.000 |
| | n=50 | 0.902 | 0.544 | +0.358 | 0.998 |
| | n=150 | 0.877 | 0.546 | +0.331 | 0.993 |

**Separation** = mean(adherent) − mean(violation). Cuanto más alto y más positivo, mejor separa los dos grupos.

**AUC** = probabilidad de que el método rankee correctamente un caso adherente por encima de uno de violación. No requiere threshold.

---

### Tabla 2 — Clasificadores binarios (F1 por clase)

| Método | Cond. | F1 adh. | F1 viol. | F1 macro | κ |
|--------|-------|---------|---------|---------|---|
| NLI | n=15 | 0.571 | 0.000 | 0.286 | 0.000 |
| | n=50 | 0.625 | 0.333 | 0.479 | 0.167 |
| | n=150 | 0.612 | 0.269 | 0.441 | 0.128 |
| **LLM Judge** | n=15 | 1.000 | 1.000 | 1.000 | 1.000 |
| | n=50 | 0.974 | 0.984 | 0.979 | 0.958 |
| | n=150 | 0.974 | 0.984 | 0.979 | 0.958 |

**LLM Judge:** Groq / `llama-3.3-70b-versatile`, temperature=0, max_tokens=5, prompt YES/NO.

---

### Tabla 3 — LLM Judge con 95% CI bootstrap (1000 resamples)

| Cond. | F1 macro | 95% CI | F1 adh. | 95% CI | F1 viol. | 95% CI |
|-------|---------|--------|---------|--------|---------|--------|
| n=15 | 1.000 | [1.000, 1.000] | 1.000 | [1.000, 1.000] | 1.000 | [1.000, 1.000] |
| n=50 | 0.979 | [0.934, 1.000] | 0.975 | [0.914, 1.000] | 0.984 | [0.947, 1.000] |
| n=150 | 0.979 | [0.952, 1.000] | 0.975 | [0.942, 1.000] | 0.984 | [0.963, 1.000] |

---

## Hallazgos clave — para el paper

### 1. AUC≈1.0 en BERTScore, Cosine y k-NN es un artefacto del benchmark

No es un resultado impresionante. El benchmark fue diseñado así: las respuestas adherentes son semánticamente similares al ground truth (distintas en fraseo pero no en contenido); las violaciones son semánticamente distintas por construcción. Cualquier método basado en similitud semántica va a separar perfectamente este benchmark. En producción, donde no existe ground truth por turno, BERTScore y Cosine requerirían uno — lo cual es costoso de mantener.

### 2. Cosine y k-NN son equivalentes y son los mejores métodos determinísticos

Ambos hacen una comparación per-turno contra el ground truth. Separation estable en ≈+0.38 independientemente de n. k-NN es conceptualmente más robusto (distancia al vecino más cercano en el set de referencia, no requiere un único ground truth por turno).

### 3. Mahalanobis: maldición de dimensionalidad confirmada

AUC degrada de 0.963 (n=15) a 0.856 (n=150). El espacío de 384D hace que la estimación de covarianza sea inestable. **No escala.**

### 4. PCA+Mahalanobis: hipótesis REFUTADA — invierte la señal

La reducción dimensional 384D→32D no ayuda: **invierte el ranking**. AUC=0.000 en n=15 (el método clasifica al revés), mejora levemente a 0.219 en n=150 pero sigue siendo inútil. La proyección PCA cambia la geometría del espacio de tal forma que las violaciones quedan más cerca del centro de referencia que los adherentes. Esto es un hallazgo honesto que debe incluirse en el paper tal cual.

### 5. KL Divergence: señal nula en n pequeño, marginal en n grande

AUC=0.537 en n=15 (prácticamente aleatorio). Mejora a 0.877 en n=150 pero sigue siendo inferior a Cosine. Opera al nivel equivocado: las violaciones de rol usan vocabulario similar a las respuestas adherentes.

### 6. NLI: clasificador binario débil

F1_macro=0.441 en n=150, κ=0.128. No discrimina bien la clase "violation". No se recomienda como método principal.

### 7. LLM Judge (llama-3.3-70b-versatile via Groq): excelente

F1=0.979 con n≥50, CI bootstrap estrecho [0.934–1.000]. El n=15 con F1=1.000 y CI=[1.000,1.000] es artefacto de muestra pequeña con casos "fáciles" — el número representativo es n=150: **F1=0.979**.

---

## Marco conceptual

### Qué mide cada método

| Método | Qué mide | Output |
|--------|----------|--------|
| BERTScore | Similitud semántica token a token vs ground truth | Continuo [0,1] |
| Cosine | Similitud entre embeddings de frases completas vs ground truth | Continuo [0,1] |
| Mahalanobis | Distancia al centro del espacio de embeddings de referencia | Continuo (normalizado) |
| KL Divergence | Diferencia entre distribuciones de vocabulario vs referencia | Continuo (exp(-KL)) |
| PCA+Mahalanobis | Mahalanobis en espacio comprimido (384D→32D) | Continuo — **INVERTIDO** |
| k-NN | Proximidad coseno al vecino más cercano en set de referencia | Continuo [0,1] |
| NLI | Detección de contradicción vs ground truth | Binario {0,1} |
| LLM Judge | Lee el rol explícitamente, decide adherencia | Binario {YES=1, NO=0} |

**Conclusión clave:** Los métodos determinísticos son detectores de deriva semántica, no de adherencia al rol. Son proxies indirectos. El LLM judge es el único que lee el rol.

### Por qué AUC y no F1 para métodos continuos

Los métodos continuos requieren un threshold para clasificar. Cualquier threshold es arbitrario en producción. AUC es threshold-agnostic: mide si el método rankea correctamente adherentes por encima de violaciones. Permite comparación justa entre todos los métodos.

### El problema de escala de n

Los métodos determinísticos que dependen de una distribución de referencia (Mahalanobis, KL) se comportan diferente según n. Los per-turno (Cosine, k-NN) son estables.

---

## Recomendación de uso en producción

| Escenario | Método recomendado | Por qué |
|-----------|-------------------|---------|
| Monitoreo continuo, bajo costo | Cosine o k-NN | Estables, per-turno, AUC≈0.99 |
| Sin ground truth disponible | LLM Judge | Único que lee el rol directamente |
| Auditoría puntual o caso crítico | LLM Judge | F1=0.979, CI estrecho |
| **No usar** | PCA+Mahalanobis, NLI | PCA invierte la señal; NLI κ≈0 |

---

## Estado del experimento

- **Benchmark:** `experiments/benchmark.json` — 150 turnos, seed=42
- **Notebook:** `experiments/experiments.ipynb` — kernel "General (Python 3.11)", deps instaladas en `/Users/frino/.venvs/general/`
- **Judge:** Groq API (`llama-3.3-70b-versatile`), key en `experiments/.env`
- **Figuras generadas:** `figures/fig1_similarity_by_label.pdf`, `fig2_separation_curve.pdf`, `fig3_llm_judge_ci.pdf`
- **Paper LaTeX:** `role_adherence_en.tex` — draft completo hasta metodología; faltan Results y Discussion

---

## Qué escribir en el paper (próxima sesión)

### Sección Results

1. **Tabla 1** (métodos continuos): separation y AUC por método × condición. Destacar Cosine/k-NN vs Mahalanobis degradante vs PCA invertido.
2. **Tabla 2** (clasificadores binarios): F1 por clase para NLI y LLM Judge.
3. **Tabla 3** (bootstrap CI del LLM Judge): F1 macro + per-clase con intervalos.
4. **Figure 1**: barras de mean score adherent vs violation por método.
5. **Figure 2**: curva de separación vs n — Mahalanobis degrada, PCA empeora aún más.
6. **Figure 3**: barras F1 del LLM Judge con error bars.

### Sección Discussion

- Cosine/k-NN son equivalentes y los mejores determinísticos — pero su AUC≈1.0 es artefacto del benchmark
- Mahalanobis confirma maldición de dimensionalidad en 384D
- PCA+Mahalanobis refuta la hipótesis de reducción dimensional — resultado honesto aunque contraintuitivo
- KL opera al nivel equivocado (vocabulario ≠ intención)
- LLM Judge es claramente superior pero tiene costo por llamada y no es determinístico
- Herramientas complementarias, no intercambiables

### Sección Conclusions

- Estrategia condicionada al contexto: Cosine/k-NN para monitoreo, LLM Judge para auditorías
- Limitation: benchmark sintético de un solo dominio (FinPay support agent)
- Future work: benchmark multi-dominio, estudio de cost/accuracy tradeoff por tipo de violación
