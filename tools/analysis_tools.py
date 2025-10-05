from langchain_core.tools import tool
from typing import List


# Data Analysis Tools
@tool
def calculate_average(numbers: List[float]) -> str:
    """Calculate the average of a list of numbers."""
    if not numbers:
        return "Error: The list is empty, cannot calculate an average."

    average = sum(numbers) / len(numbers)
    return f"The average of the numbers is {average:.2f}."


@tool
def find_max_min(numbers: List[float]) -> str:
    """Find the maximum and minimum values in a list of numbers."""
    if not numbers:
        return "Error: The list is empty, cannot find max or min."

    max_val = max(numbers)
    min_val = min(numbers)
    return f"In the provided list, the maximum value is {max_val} and the minimum value is {min_val}."


@tool
def calculate_percentage(part: float, whole: float) -> str:
    """Calculate what percentage the 'part' is of the 'whole'."""
    if whole == 0:
        return "Error: The 'whole' value cannot be zero for a percentage calculation."

    percentage = (part / whole) * 100
    return f"{part} is {percentage:.2f}% of {whole}."