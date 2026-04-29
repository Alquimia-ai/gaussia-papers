# Detección determinística de casos *seguros* en PLR

Para reducir la carga sobre el juez LLM, buscamos un filtro que clasifique con **casi cero falsos negativos** (ningún leak marcado como seguro) las respuestas que **definitivamente no contienen fragmentos del prompt del sistema**. En esta investigación proponemos heurísticas determinísticas basadas en las características del texto, coincidencias negativas con el prompt y reglas explícitas. A continuación se analizan seis enfoques clave, con evidencias y ejemplos de la literatura.

## 1. Señales intrínsecas en la respuesta

Ciertas pistas léxicas y estructurales indican una **respuesta de rechazo o genérica**, típicamente segura. Ejemplos son:

- **Frases de disculpa o negación**: respuestas que incluyen “Lo siento, pero…”, “No puedo ayudar con eso”, “No tengo la capacidad/información”, etc., suelen ser rechazos seguros. Estudios de comportamiento de LLM muestran que los refusals casi siempre comienzan con estas expresiones【32†L80-L84】.  
- **Ausencia de detalle técnico o específico**: una respuesta muy vaga o evasiva (p.ej. un párrafo que no aporta detalles técnicos) suele indicar que el modelo no está revelando datos confidenciales. Por ejemplo, falta de nombres de sistemas, codificaciones, o términos de dominio.  
- **Forma gramatical de disculpas**: palabras como “no puedo”, “no está permitido”, “no tengo acceso”, “no deberíamos”, indican un bloqueo formal. OpenAI define patrones de *rechazo ideal* que incluyen disculpa breve y declaración de incapacidad【49†L29-L34】.  
- **Sentido negado o interrogativas**: respuestas que confunden la pregunta o devuelven otra pregunta (“¿Qué intentas hacer?”) normalmente son safe, ya que no exponen datos del prompt.  
- **Longitud de la respuesta**: muy corta (p.ej. una frase de negación) o excesivamente larga pero sin contenido útil. En general, un rechazo es más breve que una respuesta legítima con explicación técnica.  

No existe un estudio único que liste estas señales, pero investigaciones de alineamiento apuntan a detectarlas con reglas o clasificadores ligeros【32†L80-L84】. Los estudios de *refusal* de Mazeika et al. (HarmBench) y Jiang et al. confirman que la detección basada en frases comunes de rechazo funciona bien【32†L80-L84】. En la práctica, podríamos compilar una **lista de plantillas** comunes (regex) de rechazo y medir cuánto coincide una respuesta con ellas. Si la respuesta encaja fuertemente en un patrón de rechazo (p.ej. contiene “lo siento” + verbo negativo), es muy probable que sea segura y se puede cerrar. 

## 2. Prompt-Negative Matching (coincidencia inversa con el prompt)

Más que buscar fuga, buscamos confirmar **ausencia de cualquier fragmento sensible**. Una estrategia determinística es verificar si en la respuesta aparece **algún n-grama del prompt de sistema**. Proponemos:  
- **Extracción de n-gramas** (o tokens) del prompt. Por ejemplo, todos los *bi-gramas* y *tri-gramas* de palabras clave (variables sensibles, instrucciones, nombres técnicos).  
- **Comparación exacta**: si cualquier n-grama (e.g. 3-palabras contiguas) del prompt aparece en la respuesta, entonces no se clasifica como seguro (posible fuga). De lo contrario, consideramos la respuesta *posiblemente* segura.  

El umbral típico sería **0 coincidencias** para considerar *definitivamente* seguro. Un resultado experimental relevante lo da Agarwal et al.: al usar ROUGE-L recall (que mide secuencia común más larga) para detectar fugas, encontraron *recall perfecto* (0 falsos negativos)【26†L807-L812】. Dicho de otra forma, cualquier fragmento común entre prompt y respuesta indica fuga.  

**Modos de falla:** La coincidencia exacta falla si el modelo **parafrasea o usa sinónimos** para el prompt. Por ejemplo, cambiar “límites de transacción” por “restricciones operativas” podría ocultar la coincidencia. Tampoco detecta codificaciones especiales (hex, base64) ni errores tipográficos intencionales. Para mitigarlo: 
- Se puede considerar coincidencias de **subcadenas de caracteres** (regex sobre patrones clave, p.ej. formatos de API keys) en vez de solo palabras.  
- Ajustar el tamaño del n-grama: quizá exigir coincidencias de 3 o más palabras seguidas reduce falsos positivos con palabras comunes («de», «el», etc.), pero aumenta riesgo de falso negativo si el atacante divide el secreto en pequeñas partes.  
- El enmascaramiento ofensivo: el adversario podría ofuscar datos (por ejemplo, escribir “k#y” en lugar de “key”), lo que escapa a coincidencias simples.  

En general, un filtro estricto de 0 coincidencias (Rouge-L recall = 0) garantiza **casi cero falsos negativos**【26†L807-L812】, pero puede dar falsos positivos en respuestas legítimas que casualmente usen palabras comunes del prompt. Dado que la prioridad es no perder ninguna fuga real, este método es sólido: marcamos *seguras* solo las respuestas sin coincidencias directas con ningún texto sensible del prompt.

## 3. Umbral asimétrico en el reranker

El reranker actual arroja puntuaciones bimodales: cerca de 1.0 en fugas evidentes y cerca de 0 en otros casos (incluso fugas ignoradas). Conceptualmente podríamos intentar un *umbral de piso*: si `MaxChunkSim < X` muy pequeño, declarar seguro. Sin embargo, **los datos sugieren que algunos ataques obtienen scores casi 0**. Cualquier X>0 implicaría que dichos ataques (score≈0) serían clasificados como seguros: inaceptable. Por tanto, no hay un umbral *seguro* mayor que 0 sin introducir falsos negativos. 

En la práctica, podríamos usar este criterio de modo conservador: por ejemplo, si `MaxChunkSim < 0.01`, etiquetar seguro solo si **otras señales** también coinciden (e.g. pasada la comprobación negativa con prompt). De forma aislada, el reranker por sí mismo no provee un umbral confiable para esta tarea, dado el solapamiento cero entre las dos modalidades observadas. En resumen, un umbral de piso *determinístico* sobre la salida del reranker no garantiza 0 falsos negativos: mejor no basar la decisión segura exclusivamente en él.

## 4. Clasificadores de comportamiento

Además de reglas simples, se pueden usar clasificadores ligeros para diferenciar “respuestas de rechazo/omisión” de “respuestas de divulgación”. Ejemplos:

- **Modelos entrenados pequeños**: p.ej. un BERT-base fine-tuneado en un dataset de respuestas etiquetadas como *seguras* vs *filtradas*. A falta de dataset propio, se puede generar artificialmente: tomar ejemplos de rechazos y de divulgaciones de sistemas (como en HarmBench o en Danger prompts) y entrenar logistic regression o *DistilBERT/TinyBERT* para clasificarlos.  
- **Clasificadores rule-based**: un modelo de regresión logística con características manuales como: número de negaciones, presencia de pronombres subjetivos (“yo” en contextos sospechosos), porcentaje de palabras comunes vs técnicas, etc. Por ejemplo, OpenAI define reglas para *Hard Refusal* que penalizan respuestas “juiciosas” y buscan disculpas breves【49†L29-L34】. Se puede adaptar: por ejemplo, una función de puntaje que suma 1 si aparece “lo siento”, +1 si hay negación, -1 si hay término técnico experto, etc.  
- **Clasificadores basados en LLMs ligeros**: usar un LLM pequeño (p.ej. GPT-3.5 “mini”) en modo *judge* con pocas demostraciones para decidir “¿Esta respuesta es un rechazo seguro?”. Esto sería más costoso pero aún más ligero que un reranker grande.  

En la literatura no hay un modelo abierto específico público para esta tarea exacta, pero frameworks como HarmBench (Mazeika et al.) sí utilizan clasificaciones de seguridad/refusal. Cualquiera de estos clasificadores debe probarse contra un benchmark adversarial; por ejemplo, el Score de OpenAI RBR (Reglas de Recompensa) se entrena con reglas que distinguen rechazo deseable vs no. Estas reglas incluyen “respuesta ideal de rechazo = contiene disculpa + incapacidad”【49†L29-L34】, lo cual puede usarse para diseñar features. No obstante, la mayoría de estas herramientas (HarmBench, LLM-as-judge, RBR) están enfocadas en el entrenamiento, no en la inferencia rápida sin promesa de 0% FN. Por tanto, recomendamos cualquier clasificador solo como ayuda secundaria, nunca como único filtro.

## 5. Trabajos y herramientas existentes

Hay investigaciones y herramientas que tocan partes del problema:

- **Detección mediante ROUGE**: El paper de Agarwal et al. (2024) encontró que usar ROUGE-L Recall para medir superposición entre prompt sensible y respuesta produce recall perfecto (cero FN)【26†L807-L812】, confirmando que un simple filtro de coincidencias textuales puede detectar todas las fugas conocidas (aunque con baja precisión).  
- **Canary tokens (marcadores)**: Herramientas prácticas como *Rebuff* (LangChain) introducen tokens canarios únicos en el prompt del sistema. Si estos aparecen en la salida, indica claramente filtración【58†L141-L148】. Este método es completamente determinístico: basta revisar si el marcador secreto reaparece. Su desventaja es que se requiere modificar el prompt original (insertar palabras invisibles) y gestionar casos falsos (si el modelo copia por error el canary). Pero es muy efectivo para detectar fuga exacta.  
- **Reglas OWASP/NIST para LLM**: Los grupos OWASP GenAI y NIST han documentado la categoría “Sensitive Information Disclosure” (Top10 LLMs)【55†L75-L83】. Recomiendan sanear entradas y restringir respuestas, pero también sugieren revisar salidas (auditoría/monitoring) para detectar fugas. No ofrecen un algoritmo específico, pero refuerzan que *filtrar las salidas por contenido confidencial* es buena práctica.  
- **Evaluaciones de fusibles (faucets)**: Algunas propuestas (Jain et al., 2024; OpenAI GPT-5) trabajan en “safe completion” en contraposición al rechazo. Estas suelen reescribir respuestas de forma indirecta. No abordan directamente detección, pero enfatizan que *llevar los anuncios de fuga a valores claramente altos o bajos* facilita su identificación.  

En resumen, no existe una solución publicada que sea un detector mágico de “sin fuga” con alta precisión, pero hay varias técnicas parciales (ROUGE, canarios, reglas de sanción) que confirman que se puede verificar la ausencia de información sensible con alto recall si se aplica con cuidado【26†L807-L812】【58†L141-L148】. 

## 6. Recomendación práctica

La solución **más prometedora** que proponemos implementar es un pre-filtro basado en **coincidencia negativa estricta** con el prompt, posiblemente complementado con detección de frases de rechazo. En la práctica esto sería:

1. **Coincidencia exacta con el prompt**: Calcular para cada respuesta el *Rouge-L Recall* respecto al texto del prompt del sistema【26†L807-L812】. Alternativamente, buscar *n-gramas* de palabra del prompt (p.ej. tri-gramas) en la respuesta. Si el resultado es cero (ninguna coincidencia encontrada), marcamos la respuesta como *segura*. Esto garantiza casi 100% recall de fugas (0 FN)【26†L807-L812】.  
2. **Opcional – chequeo de rechazo**: Como precaución, si la respuesta es muy corta o contiene expresiones claras de disculpa (“no puedo”, “no tengo permiso”, etc.), podemos *automáticamente* marcar segura aún sin coincidencia con el prompt. Este paso es menor, pues la coincidencia negativa ya captura la mayoría de los casos.  
3. **Tuning de umbrales**: Para evitar falsos positivos innecesarios (legítimas respuestas sin promover al juez), se puede ajustar el criterio. Por ejemplo, en vez de chequear *cada token*, podemos exigir coincidencias de al menos 2 o 3 palabras contiguas (lo que reduce falsos positivos con palabras comunes). El umbral exacto (0 palabras, 1, 2) se barrerá en validación: creemos que **Rouge-L Recall = 0** (ningún match de peso) es el punto de partida más seguro【26†L807-L812】. Si se tolera 1 palabra común (umbral muy bajo), aumenta el FP.  
4. **Evaluación en los 68 casos**: Aplicar este filtro a los 68 ejemplos. Calcular métricas como *detección de leaks* (casos adversariales con match >0 no se marcan seguros) y *FP rate* (respuestas legítimas sin match que se marcan seguras). Ajustar K-gram o la lista de stopwords si es necesario. El criterio deseado es recall≈1.0 en fugas (cero fugas clasificadas seguras) y la mayor precisión posible para no escalarlas innecesariamente.  

Este método es **determinístico, de bajo costo** (solo procesamiento de texto) y explicable: no confía en cajas negras. En resumen, *filtrando cualquier coincidencia con el prompt*, cerramos como seguras únicamente respuestas que efectivamente no repiten contenido sensible【26†L807-L812】. Gracias a esto, lograremos **casi 100% de detección de leaks** (criterio primario) y solo escalaremos al juez los casos legítimos donde pueda haber verdadera duda. 

**Fuentes consultadas:** Investigaciones de red-teaming (Agarwal et al. 2024) confirman la eficacia de ROUGE para alcanzar recall =1【26†L807-L812】; análisis de refusal (OpenAI RBR) sugieren usar expresiones de disculpa como indicadores【49†L29-L34】; y herramientas prácticas (Rebuff) recomiendan tokens canarios para detección explícita【58†L141-L148】. Estos respaldan la combinación de coincidencia negativa y detección de patrones como solución liviana al problema.