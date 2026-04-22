import re
import numpy as np
# from math import prod
# import json
import torch

def _tokenize(pattern_str):
    """
    Tokenize a pattern string into a list of items:
    - '...' (ellipsis)
    - '(...)' (group)
    - '\w+' (single axis, e.g. 'b', 'h', 'h1')
    """

    tokens = re.findall(r'\.\.\.|\([\w\s\-]+\)|-?\w+', pattern_str)
    return tokens

def to_numpy_array(input_data):
    """
    Converts the input data to a NumPy array.
    Supports:
        - list
        - NumPy array (returned as-is)
        - PyTorch tensor (converted to NumPy array)
    Raises:
        - TypeError if input data is of an unsupported type.
    """
    if isinstance(input_data, np.ndarray):
        return input_data
    elif isinstance(input_data, list):
        return np.array(input_data)
    elif isinstance(input_data, torch.Tensor):
        return input_data.numpy()
    else:
        raise TypeError(f"Unsupported input type: {type(input_data)}. Expected list, NumPy array, or PyTorch tensor.")

def clean_singletons_in_parentheses(pattern):
    """
    Cleans up singletons in parentheses in einops rearrange patterns.

    Parameters:
        pattern (str): The rearrange pattern to clean.

    Returns:
        str: The cleaned pattern.

    For example,
    Input: b (c 1 1) h w -> b c h w
    Output:  b c h w -> b c h w
    """

    # Define a regex to match parentheses containing dimensions
    paren_regex = re.compile(r'\(([^()]+)\)')

    def replace_function(match):
        # Extract inside of parentheses
        content = match.group(1)
        # Remove `1` from the dimensions
        cleaned_content = ' '.join([dim for dim in content.split() if dim != '1'])
        # If only one dimension remains, remove parentheses
        return cleaned_content if ' ' not in cleaned_content else f'({cleaned_content})'

    # Replace all parentheses content using the defined function
    cleaned_pattern = paren_regex.sub(replace_function, pattern)

    # Remove redundant spaces and ensure clean formatting
    cleaned_pattern = re.sub(r'\s+', ' ', cleaned_pattern).strip()

    return cleaned_pattern

def unexpected_chars_checker(s):
    """
    Check for unexpected characters in the pattern.

    Returns:
        bool: True if no unexpected characters are found, raises a ValueError otherwise.

    Raises:
        ValueError: If unexpected characters are found in the pattern.
    """

    # Define the valid pattern for allowed tokens
    valid_pattern = re.compile(r'^([a-zA-Z]+[1-9][0-9]*|[a-zA-Z]+|\.\.\.|1)$')  # Matches h1, h11, a, b, ..., or 1

    # Remove '->' and parentheses for simpler parsing
    tokens = s.replace('->', '').translate(str.maketrans('', '', '()')).split()

    # Check each token against the valid pattern
    unexpected_chars = [token for token in tokens if not valid_pattern.match(token)]

    if unexpected_chars:
        raise ValueError(f"Unexpected characters found in pattern: {unexpected_chars}")

    # return True  # No unexpected characters found

def tokens_from_paranthesis(input_set):
    """
    Get identifiers from paranthesis
    """

    paranth_set = set()
    for item in input_set:
        # Check if the item contains parentheses
        if '(' in item and ')' in item:
            # Extract elements inside parentheses and split by whitespace
            inner_elements = re.findall(r'\((.*?)\)', item)
            for group in inner_elements:
                paranth_set.update(group.split())

    return paranth_set

def check_extra_arguments(input_mapping, **kwargs):
    """
    Validates that no extra arguments are provided that are not part of the input mapping.

    Args:
        input_mapping (dict): Input token mapping.
        kwargs: Additional arguments provided to the function.

    Raises:
        ValueError: If extra arguments are detected.
    """
    allowed_tokens = tokens_from_paranthesis(set(input_mapping.keys()))
    provided_tokens = set(kwargs.keys())

    # Find extra arguments
    extra_tokens = provided_tokens - allowed_tokens

    if extra_tokens:
        raise ValueError(f"Extra arguments provided: {extra_tokens}. "
                         f"Allowed arguments are: {list(allowed_tokens)}.")

    # print("No extra and unnecessary arguments provided.")

def get_additional_args(input_mapping, shape_mapping, **kwargs):
    """
    Validates parentheses in input token mappings and ensures the arguments
    provided match the required shapes in the shape mapping. Returns all
    arguments, including inferred ones.

    Args:
        input_mapping (dict): Input token mapping.
        shape_mapping (dict): Input shape mapping.
        kwargs: Additional arguments corresponding to tokens inside parentheses.

    Returns:
        dict: A dictionary of all arguments (inferred + original).

    Raises:
        ValueError: If parentheses validation fails.
    """
    inferred_args = {}

    for token, index in input_mapping.items():
        if '(' in token and ')' in token:
            # Extract tokens inside parentheses
            inner_tokens = token.strip('()').split()

            # Retrieve the shape value for the grouped token
            expected_shape = shape_mapping[token]

            # Check provided arguments
            provided_args = [kwargs[arg] for arg in inner_tokens if arg in kwargs]

            if len(provided_args) < len(inner_tokens):
                # Try to infer the other argument
                product = 1
                for arg_value in provided_args:
                    product *= arg_value
                if expected_shape % product != 0:
                    raise ValueError(f"Could not infer sizes for {set(inner_tokens) - set(list(kwargs.keys()))}.")
                else:
                    inferred_value = expected_shape // product
                    inferred_token = [t for t in inner_tokens if t not in kwargs][0]
                    inferred_args[inferred_token] = inferred_value
                    # print(f"Inferred value for {inferred_token}: {inferred_value}")

            elif len(provided_args) == len(inner_tokens):
                # Multiple arguments must exactly match the shape value when multiplied
                product = 1
                for arg_value in provided_args:
                    product *= arg_value
                if product != expected_shape:
                    raise ValueError(f"Product of arguments {inner_tokens} ({provided_args}) does not match "
                                     f"the expected shape {expected_shape} for token {token}.")
            else:
                # No arguments provided, cannot validate
                raise ValueError(f"Missing required arguments for token: {token}. "
                                 f"Expected at least one of {inner_tokens}.")

    # Combine original and inferred arguments
    all_args = {**kwargs, **inferred_args}
    # print("All parentheses checks passed.")
    return all_args