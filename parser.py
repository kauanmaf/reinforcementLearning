import re

def extract_code(text):
    # Encontramos todos os blocos de código e remove o texto após ``` na abertura
    code_blocks = re.findall(r'```(?:\w*\n)?(.+?)```', text, re.DOTALL)
    # Retornamos todos os blocos jutnos
    return "\n".join(block.strip() for block in code_blocks)

# Durante o código precisamos extrair tuplas com 10 inteiros ou 14 inteirtos e para isso fazemos um regex
def parse_tuple(response: str) -> tuple:

    # Procuramos tuplas com 10 inteiros de 0 a 10
    match = re.search(r'\((\d{1,2}(?:,\s?\d{1,2}){9})\)', response)  
    if not match:
        # Procuramos tuplas com 14 inteiros de 0 a 10
        match = re.search(r'\((\d{1,2}(?:,\s?\d{1,2}){13})\)', response)
    
    # Se acharmos retornamos
    if match:
        return tuple(map(int, match.group(1).split(',')))
    else:
        return None