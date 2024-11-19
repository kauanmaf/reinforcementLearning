import re

def parse_tuple(response: str) -> tuple:

    match = re.search(r'\((\d{1,2}(?:,\s?\d{1,2}){9})\)', response)  
    if not match:
        match = re.search(r'\((\d{1,2}(?:,\s?\d{1,2}){13})\)', response)
    
    if match:
        return tuple(map(int, match.group(1).split(',')))
    else:
        raise ValueError("No valid tuple found in the response")