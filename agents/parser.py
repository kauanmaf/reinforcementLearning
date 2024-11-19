import re

def extract_code(text):
    # Encontra todos os blocos de código e remove o texto após ``` na abertura
    code_blocks = re.findall(r'```(?:\w*\n)?(.+?)```', text, re.DOTALL)
    return "\n".join(block.strip() for block in code_blocks)