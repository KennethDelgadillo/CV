from transformers import pipeline

# Cargar un modelo genérico de prueba preentrenado en inglés
generator = pipeline("text-generation", model="bigscience/bloom-560m")
#generator = pipeline("text2text-generation", model="google/flan-t5-base")

def promtpCVvsJP(CV, JP):
    prompt = f"""
        Evaluate how well the following resume matches the job description. Provide a score from 1 to 10, followed by professional feedback explaining the score and suggesting improvements.

        Original resume:
        {CV}

        Job description:
        {JP}

        Output:
        Score: <A score from 1 to 10 indicating the relevance of the resume to the job description.>
        Feedback: <Detailed feedback explaining why the resume scored this way and suggesting specific improvements.>
        """
    
    prompt_test = f"""
        Analyze how well the following resume matches the job description. Assign a relevance score from 1 to 10, and provide professional feedback explaining the score and suggesting ways to improve.

        Resume:
        {CV}

        Job description:
        {JP}

        Output:
        Score: <A score from 1 to 10>
        Feedback: <Detailed explanation and suggestions for improvement>
        """
    prompt_prueba = f"""
        Read the next CV and tell me what do you think about that.
        CV: {CV}
        """
    result = generator("Complete: 'Hello...'", max_new_tokens=60, num_return_sequences=1, temperature=0.3)
    #result = generator("Complete: 'Hello...'", max_new_tokens=60, num_return_sequences=1)
    return result[0]['generated_text']

CV = """
John Doe, Software Engineer with 3 years of experience in Python and JavaScript. Skilled in back-end development and API design.
"""
JP = """
We are looking for a Python developer with experience in Docker, CI/CD pipelines, and familiarity with AWS.
"""

# Generar texto
result = promtpCVvsJP(CV, JP)
print("Gen AI says: ", result)