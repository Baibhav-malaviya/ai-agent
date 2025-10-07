from .analysis_tools import calculate_average, calculate_percentage, find_max_min
from .math_tools import add, subtract, multiply, divide, power, square_root
from .utility_tools import search_web, word_count, weather_info, save_to_memory, get_current_datetime
from .signup import sign_up

all_tools = [
    calculate_average,
    calculate_percentage,
    find_max_min,
    add,
    subtract,
    multiply,
    divide,
    power,
    square_root,
]

utility_tools = [
    search_web,
    word_count,
    weather_info,
    get_current_datetime
]