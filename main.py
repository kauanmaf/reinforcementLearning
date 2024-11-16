import os
from agents.coder import Coder
from agents.reviewer import CodeReviewer
from agents.judger import Judger
from dotenv import load_dotenv
from groq import Groq
import pandas as pd
from environment import Environment
import instructor
from pydantic import BaseModel
from agents.utils import *

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


with open("prompts/problem_description.txt", "r") as file:
    prompt_problem = file.read()

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

client = instructor.from_groq(client, mode=instructor.Mode.TOOLS)

answer = client.chat.completions.create(
            messages = "crie um codigo para printar uma arvore de a's",
            model = "llama3-8b-8192",
            response_model = ReportReviewer
        )
print(answer)

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
sample_submission = pd.read_csv("data/sample_submission.csv")

data = {"train": train, "sample_submission": sample_submission, "test": test}

coder = Coder(client, prompt_problem.format(data = data))
reviewer = CodeReviewer(client, prompt_problem.format(data = data))
judger = Judger(client, prompt_problem.format(data = data))

report_points = 0

env = Environment(coder, reviewer, judger)

env.run_episode()