import spacy
from spacy.matcher import PhraseMatcher
#print(spacy.__version__)

# INICIALIZACION DE SpaCy
nlp = spacy.load("en_core_web_sm")
#nlp = spacy.load("en_core_web_sm", disable=["parser"]) # Desactivo el "parser" para despues suplirlo con "sentencizer"
#nlp.add_pipe("sentencizer") # Divide el texto en oraciones
matcher = PhraseMatcher(nlp.vocab)

# CARGAR ARCHIVO
filename = "testcv0.txt" # testcv0 , testjp0
try:
  with open(filename, "r", encoding="utf-8") as file:
    CV = file.read()
except FileNotFoundError:
  print(f"Error: The file {filename} was not found.")
  CV = ""

# provando inicia
lines = [line.strip() for line in CV.split("\n") if line.strip()]
preprocessed_CV = ""
for line in lines:
    if line.endswith("."):
        preprocessed_CV += line + " "  # Mantén el punto existente y agrega espacio
    else:
        preprocessed_CV += line + ". "  # Agrega un punto y espacio si no termina en "."
CV = preprocessed_CV
# termina provando

doc = nlp(CV.lower())
print(doc.text)

# LISTAR KEYWORDS COMO SKILLS, JOBTITLE, ETC.
list_of_skills = ["python", "javascript", "machine learning", "nlp", "data analysis", "artificial intelligence", "pandas", "git"]
list_of_job_title = ["software engineer", "data engineer", "java developer", "testing engineer"]
list_of_key_phrases = ["experience in", "worked with", "developed", "skilled in"]

# AÑADIR LAS LISTAS DE KEYWORDS AL matcher DE PhraseMatcher
matcher.add("SKILLS", [nlp.make_doc(skill) for skill in list_of_skills]) # convertir cada skill en un doc
matcher.add("JOBTITLES", [nlp.make_doc(title) for title in list_of_job_title])
matcher.add("KEY_PHRASES", [nlp.make_doc(phrase) for phrase in list_of_key_phrases])

# FUNCION PARA BUSCAR COINCIDENCIAS ENTRE EL MATCHER Y EL DOCUMENTO
def findInfo(doc):
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
def evaluate_CV_vs_position(cv_info, jp_info):
  # CREA EL SET DE HABILIDADES DEL CANDIDATO EN SU CV Y LAS HABILIDADES REQUERIDAS 
  cv_skills = set([skill for entry in cv_info for skill in entry["skills"]])
  jp_skills = set([skill for entry in jp_info for skill in entry["skills"]])

  required_skills_matched = cv_skills & jp_skills
  required_score = len(required_skills_matched) / len(jp_skills) * 100 if jp_skills else 0
  print(f"El score obtenido es {required_score}. Las habilidades solicitadas son {jp_skills} y el cantidato tiene {required_skills_matched}")

cv_info = findInfo(doc)
jp_info = findInfo(doc)
print(jp_info)

cv_info = [{"skills": ["artificial intelligence", "git", "nlp"]}] # datos de prueba
evaluate_CV_vs_position(cv_info, jp_info)