import os
import instructor
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel
from typing import Tuple
load_dotenv()
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

# By default, the patch function will patch the ChatCompletion.create and ChatCompletion.create methods to support the response_model parameter
client = instructor.from_groq(client, mode=instructor.Mode.TOOLS)


# Now, we can use the response_model parameter using only a base model
# rather than having to use the OpenAISchema class
class UserExtract(BaseModel):
    code : Tuple[int, int] 


user: UserExtract = client.chat.completions.create(
    model="llama3-8b-8192",
    response_model=UserExtract,
    messages=[
        {"role": "user", "content": "Gere uma tupla de dois inteiros"},
    ],
)
print(user)
print(type(user.code))
assert isinstance(user, UserExtract), "Should be instance of UserExtract"
# assert user.name.lower() == "jason"
# assert user.age == 25

# print(user.model_dump_json(indent=2))
# """
# {
#   "name": "jason",
#   "age": 25
# }
# """