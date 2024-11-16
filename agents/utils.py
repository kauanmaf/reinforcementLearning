from pydantic import BaseModel, field_validator
from typing import Tuple

class ReportJudger(BaseModel):
    scores: Tuple[int, int, int, int, int, int, int, int, int, int, int, int, int, int]

    @field_validator('scores')
    @classmethod
    def validate_scores(cls, v: Tuple[int, ...]) -> Tuple[int, ...]:
        # Check if we have exactly 14 scores
        if len(v) != 14:
            raise ValueError(f'Expected 14 scores, got {len(v)}')
        
        # Check if all scores are between 0 and 10
        if not all(0 <= score <= 10 for score in v):
            raise ValueError('All scores must be between 0 and 10')
            
        return v


class ReportReviewer(BaseModel):
    grades: Tuple[int, int, int, int, int, int, int, int, int, int]

    @field_validator('grades')
    @classmethod
    def validate_scores(cls, v: Tuple[int, ...]) -> Tuple[int, ...]:
        # Check if we have exactly 14 scores
        if len(v) != 10:
            raise ValueError(f'Expected 14 scores, got {len(v)}')
        
        # Check if all scores are between 0 and 10
        if not all(0 <= score <= 10 for score in v):
            raise ValueError('All scores must be between 0 and 10')
            
        return v
    
class Joker(BaseModel):
    generic_ans: str