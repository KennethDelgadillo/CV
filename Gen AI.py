from llama_cpp import Llama

llm = Llama.from_pretrained(
    repo_id="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
    filename="mistral-7b-instruct-v0.1.Q2_K.gguf",
)
CV = """
Software Engineer with 5 years of experience in Python, JavaScript, and machine learning.
Worked at TechCorp from 2018 to 2028 developing web applications and data analysis tools.
Skilled in NLP, data visualization, and cloud computing.
Bachelor's degree in Computer Science from State University.
"""

JP = """
Collaborate in the development and implementation of artificial intelligence solutions by performing technical tasks under guidance and supervision to ensure the delivery of digital products.

Enable the construction of AI models by ensuring the quality and availability of data.
Contribute to the development of AI solutions by implementing and validating models under supervision.
Ensure the proper functionality of models in production environments by providing support for their deployment and integration.
Experience
Development and maintenance of machine learning models.
Solid experience with cloud platforms, preferably Google Cloud Platform (Vertex AI, Cloud Run, etc.).
Knowledge
Strong fundamentals in machine learning (supervised and unsupervised).
Proficiency in Python and key libraries such as Pandas, NumPy, and Scikit-learn.
Familiarity with the operation of LLMs (Large Language Models) and prompt engineering concepts.
Basic knowledge of REST APIs, version control (Git), and networking concepts for service exposure.
Basic understanding of principles of AI ethics (bias, fairness) and data security.
Intermediate-advanced English level.
Education
Bachelor’s Degree in Data Science or related fields.
"""

prompt = f"""
### Instruction:
Rewrite the following CV profile to make it more attractive for the Job Position. Keep it concise and professional.

### CV:
{CV}

### Job Position:
{JP}

### Response:
"""

output = llm(
    prompt,
    max_tokens=600,
    echo=False,
    temperature=0.7,
    stop=["###"]
)

print(output['choices'][0]['text'].strip())
