
import re


def quality_check_prompt(draft_letter: str) -> str:
    return f"""
    Evaluate this cover letter and return structured result.

    Draft Letter:
    {draft_letter}
    """.strip()

def analysis_prompt(cv_text: str, job_text: str) -> str:

    prompt = f"""
    Analyze the candidate's CV against the job description.

    CV:
    {cv_text}

    Job description:
    {job_text}

    Identify the most relevant matching skills and experience.
    Do not invent information.
    """
    return prompt.strip()

def draft_prompt(analysis: str) -> str:
    prompt = f"""
    Draft a cover letter based on the analysis of the candidate's CV and the job description.

    Analysis:
    {analysis}

    Do not invent information.
    """
    return prompt.strip()