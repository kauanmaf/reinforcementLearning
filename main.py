import os
from agents.coder import Coder
from agents.reviewer import CodeReviewer
from agents.judger import Judger
from dotenv import load_dotenv
from groq import Groq
import pandas as pd
from environment import Environment


load_dotenv()
# Pegamos nossa API_KEY
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Pegamos a descrição do problema
with open("prompts/problem_description.txt", "r") as file:
    prompt_problem = file.read()

# Começamos um cliente
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Lemos os nossos dados do problema
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
sample_submission = pd.read_csv("data/sample_submission.csv")

# setamos num dicinário
data = {"train": train, "sample_submission": sample_submission, "test": test}

# Começamos nosso coder, reviewer e judger com o prompt formatado
coder = Coder(client, prompt_problem.format(data = data))
reviewer = CodeReviewer(client, prompt_problem.format(data = data))
judger = Judger(client, prompt_problem.format(data = data))

# Setamos um environment
env = Environment(coder, reviewer, judger)

# Rodamos nosso modelo para aprender a política
env.run()