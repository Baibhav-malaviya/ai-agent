from typing import Union
from langchain_core.tools import tool

# ----------------------------
# Define Enhanced Tool Suite
# ----------------------------

# Math Tools
@tool
def add(a: float, b: float) -> str:
    """Add two numbers together."""
    result = a + b
    return f"The sum of {a} and {b} is {result}."

@tool
def multiply(a: float, b: float) -> str:
    """Multiply two numbers together."""
    result = a * b
    return f"The product of {a} and {b} is {result}."

@tool
def subtract(a: float, b: float) -> str:
    """Subtract the second number from the first number."""
    result = a - b
    return f"The result of subtracting {b} from {a} is {result}."

@tool
def divide(a: float, b: float) -> str:
    """Divide the first number by the second number."""
    if b == 0:
        return "Error: Division by zero is not allowed."
    result = a / b
    return f"The result of dividing {a} by {b} is {result}."

@tool
def power(a: float, b: float) -> str:
    """Calculate 'a' raised to the power of 'b'."""
    result = a ** b
    return f"{a} raised to the power of {b} is {result}."

@tool
def square_root(a: float) -> str:
    """Calculate the square root of a number."""
    if a < 0:
        return "Error: Cannot calculate the square root of a negative number."
    result = a ** 0.5
    return f"The square root of {a} is {result}."