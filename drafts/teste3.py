import subprocess
import tempfile
import re

def analyze_with_ruff(code: str) -> int:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as temp_file:
        temp_file.write(code)
        temp_file.flush()
        
        result = subprocess.run(
            ["ruff", temp_file.name],
            capture_output=True,
            text=True
        )
        
    # Calcula a pontuação com base na quantidade de problemas encontrados
    issues = result.stdout.strip().count("\n") + 1 if result.stdout else 0
    score = -issues  # Cada problema reduz 1 ponto até o mínimo de 0
    return score

def analyze_with_mypy(code: str) -> int:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as temp_file:
        temp_file.write(code)
        temp_file.flush()
        
        result = subprocess.run(
            ["mypy", temp_file.name],
            capture_output=True,
            text=True
        )
        
    # Calcula a pontuação com base na quantidade de erros de tipo
    issues = result.stdout.strip().count("\n") if result.stdout else 0
    score = -2*issues  # Cada problema reduz 2 pontos até o mínimo de 0
    return score

def analyze_with_bandit(code: str) -> int:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as temp_file:
        temp_file.write(code)
        temp_file.flush()
        
        result = subprocess.run(
            ["bandit", "-r", temp_file.name],
            capture_output=True,
            text=True
        )
    
    # Calcula a pontuação com base na severidade dos problemas de segurança
    severity_pattern = re.compile(r'Severity: (\w+)')
    severities = severity_pattern.findall(result.stdout)
    
    score = 0  # Começa com uma pontuação máxima de 10
    for severity in severities:
        if severity == 'High':
            score -= 8
        elif severity == 'Medium':
            score -= 5
        elif severity == 'Low':
            score -= 3

    return score


# Código de exemplo para análise
code = """
x = "hello"
y = 5 + x  # erro de tipo int + str
print(y)
"""

# Executa as análises de qualidade
ruff_score = analyze_with_ruff(code)
mypy_score = analyze_with_mypy(code)
bandit_score = analyze_with_bandit(code)

# Exibe as pontuações
print(f"Ruff Score (Qualidade): {ruff_score}")
print(f"Mypy Score (Qualidade): {mypy_score}")
print(f"Bandit Score (Qualidade): {bandit_score}")
