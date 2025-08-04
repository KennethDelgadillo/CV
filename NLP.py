import os
import fitz
# NLP. Tokenizer
import spacy
from spacy.matcher import PhraseMatcher
#print(spacy.__version__)
# Moddel for embeddings
from sentence_transformers import SentenceTransformer
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

# INICIALIZACION DE SpaCy
nlp = spacy.load("en_core_web_sm")
#nlp = spacy.load("en_core_web_sm", disable=["parser"]) # Desactivo el "parser" para despues suplirlo con "sentencizer"
#nlp.add_pipe("sentencizer") # Divide el texto en oraciones
matcher = PhraseMatcher(nlp.vocab)

# CARGAR DATA SET DE SKILLS ------------
filename = "skills200.txt"
try:
  with open(filename, "r", encoding="utf-8") as file:  
    list_of_skills = [line.strip().lower() for line in file] 
except FileNotFoundError:
  print(f"El archivo '{filename}' no se encontró.")
  list_of_skills = ["python", "javascript", "machine learning", "nlp", "data analysis", "artificial intelligence", "pandas", "git"]

# LISTAR KEYWORDS COMO SKILLS O KEYPHRASES.
list_of_key_phrases = ["experience in", "worked with", "developed", "skilled in"]

# AÑADIR LAS LISTAS DE KEYWORDS AL matcher DE PhraseMatcher
matcher.add("SKILLS", [nlp.make_doc(skill) for skill in list_of_skills]) # convertir cada skill en un doc
matcher.add("KEY_PHRASES", [nlp.make_doc(phrase) for phrase in list_of_key_phrases])

# FUNCION CARGAR ARCHIVO Y PROCESADO
def loadFile(filename):
  _, ext = os.path.splitext(filename)
  if ext == ".txt":
    try:
      with open(filename, "r", encoding="utf-8") as file:
        CV = file.read()
    except FileNotFoundError:
      print(f"Error: The file {filename} was not found.")
      CV = ""
  elif ext == ".pdf":
    try:
      pdf = fitz.open(filename)
      CV = ""
      for page in pdf:
        CV += page.get_text()
      pdf.close()  
    except FileNotFoundError:
      print(f"Error: The file {filename} was not found.")
      CV = ""
  else:
    raise ValueError(f"File format not supported: {ext}")

  lines = [line.strip() for line in CV.split("\n") if line.strip()]
  preprocessed_CV = ""

  for line in lines:
    if line.endswith("."):
        preprocessed_CV += line + " "  # Mantén el punto existente y agrega espacio
    else:
        preprocessed_CV += line + ". "  # Agrega un punto y espacio si no termina en "."
  CV = preprocessed_CV

  doc = nlp(CV.lower())
  return doc

# FUNCION PARA BUSCAR COINCIDENCIAS ENTRE EL MATCHER Y EL DOCUMENTO
def findInfoMatch(doc):
  results = []
  # ENCUENTRA LAS FECHAS. USANDO EL lable "DATE" automaticamente Y "CARDINAL" PARA FILTRAR NUMEROS QUE PODRIAN SER FECHAS PERO NO FUERON DETECTADOS
  for sent in doc.sents:
    dates = [
      ent for ent in sent.ents 
      if ent.label_ == "DATE" or (ent.label_=="CARDINAL" and ent.text.isdigit() and 1900<int(ent.text)<2099)
    ]

    # BUSCA Y GUARDA TODAS LAS COINCIDENCIAS ENTRE EL DOCUEMENTO Y LAS KEYWORDS
    matched_skills = []
    matched_key_phrases = []
    matches_in_sent = matcher(sent)
    for match_id, start, end in matches_in_sent:
      span = doc[start:end]
      label = nlp.vocab.strings[match_id]
      if label == "SKILLS":
        matched_skills.append(span.text)
      elif label == "KEY_PHRASES":
        matched_key_phrases.append(span.text)
    
    # LE DA FORMATO A LA INFORMACION OBTENIDA Y LA GUARDA
    if dates and (matched_skills or matched_key_phrases):
      results.append({
        "dates": dates,
        "skills": matched_skills,
        "phrases": matched_key_phrases,
        "context": sent.text.strip().replace("\n"," ")
      })
    elif matched_skills:
      results.append({
        "skills": matched_skills,
        "context": sent.text.strip().replace("\n"," ")
      })
  return results

# FUNCION PARA EVALULAR EL CV VS LA POSICION
def evaluate_CV_vs_Position(cv_info, jp_info):
  # CREA EL SET DE HABILIDADES DEL CANDIDATO EN SU CV Y LAS HABILIDADES REQUERIDAS 
  cv_skills = set([skill for entry in cv_info for skill in entry["skills"]])
  jp_skills = set([skill for entry in jp_info for skill in entry["skills"]])

  required_skills_matched = cv_skills & jp_skills
  required_score = len(required_skills_matched) / len(jp_skills) * 100 if jp_skills else 0
  print(f"El score obtenido es {required_score:.2f}. Las habilidades solicitadas son {jp_skills} y el cantidato tiene {required_skills_matched}")

# FUNCION QUE APLICA EL MODELO DE EMBEDDING A LAS ORACIONES
def get_embeddings(sentences):
    return model.encode(sentences)

# FUNCION QUE HACE LA EVALUACION Y COMPARA LOS EMBEDDING DEL CV VS POSICION
def findInfoEmbedding(docCV, docJP):
  CV_sentences = [sent.text.strip() for sent in docCV.sents if len(sent.text.strip()) > 0]
  JP_sentences = [sent.text.strip() for sent in docJP.sents if len(sent.text.strip()) > 0]

  CV_embeddings = get_embeddings(CV_sentences)
  JP_embeddings = get_embeddings(JP_sentences)

  similarity_matrix = cosine_similarity(JP_embeddings,CV_embeddings)

  results = []
  for jp_idx, jp_sentence in enumerate(JP_sentences):
    # Obtiene la oración del CV con mayor similitud para esta oración del Puesto
    most_similar_idx = np.argmax(similarity_matrix[jp_idx])
    most_similar_score = similarity_matrix[jp_idx][most_similar_idx]
    most_similar_sentence = CV_sentences[most_similar_idx]

    # Guardar el resultado
    results.append({
      "job_sentence": jp_sentence,
      "most_similar_cv_sentence": most_similar_sentence,
      "similarity_score": most_similar_score
    })

  for result in results:
    if result['similarity_score'] > 0.69:
      print(f"Job Sentence: '{result['job_sentence']}'")
      print(f"Most Similar CV Sentence: '{result['most_similar_cv_sentence']}'")
      print(f"Similarity Score: {result['similarity_score']:.2f}")
      print()

# --------------------------------------------------------------------------
# MANDA A LLAMAR LA FUNCION PARA LEER EL ARCHIVO
#filename = "testcv0.txt" # testcv0
filename = "CV Kenneth.pdf"
docCV = loadFile(filename)
filename = "testjp2.txt" # testjp0
docJP = loadFile(filename)

# MANDA A LLAMAR LA FUNCION PARA HACER MATCH DE KEYWORDS
cv_info = findInfoMatch(docCV)
jp_info = findInfoMatch(docJP)

# EVALUAR EL CV VS LA POSICION
evaluate_CV_vs_Position(cv_info, jp_info)

# MANDA A LLAMAR LA FUNCION QUE COMPARA LOS EMBEDDINGS DEL CV Y POSICION
findInfoEmbedding(docCV, docJP)