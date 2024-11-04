import subprocess
import tempfile
import re

def execute_and_score_code(code: str) -> int:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as temp_file:
        temp_file.write(code)
        temp_file.flush()
        
        # Tenta executar o código e captura exceções
        try:
            exec(open(temp_file.name).read())
            execution_score = 10  # Nenhum erro, pontuação máxima
        except Exception as e:
            # Reduz a pontuação dependendo do tipo de exceção
            if isinstance(e, (SyntaxError, NameError, TypeError, AttributeError)):
                execution_score = -30  # Erros críticos
            elif isinstance(e, (IndexError, KeyError, ValueError)):
                execution_score = -20  # Erros moderados
            else:
                execution_score = -10  # Erros menos graves
    return execution_score

# Exemplo de código para análise
code = """
x = [1, 2, 3]
print(x[5])  # IndexError
eval("print('unsafe')")
"""

execution_score = execute_and_score_code(code)

# Exibe as pontuações
print(f"Execution Score (Erros em Execução): {execution_score}")