import re

def extract_code(text):
    # Encontra todos os blocos de código e remove o texto após ``` na abertura
    code_blocks = re.findall(r'```(?:\w*\n)?(.+?)```', text, re.DOTALL)
    return "\n".join(block.strip() for block in code_blocks)

def parse_tuple(response: str) -> tuple:

    match = re.search(r'\((\d{1,2}(?:,\s?\d{1,2}){9})\)', response)  
    if not match:
        match = re.search(r'\((\d{1,2}(?:,\s?\d{1,2}){13})\)', response)
    
    if match:
        return tuple(map(int, match.group(1).split(',')))
    else:
        raise ValueError("No valid tuple found in the response")