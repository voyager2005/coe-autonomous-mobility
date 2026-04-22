from math import prod

def input_based_transformation(array, input_mapping, input_shape_mapping, **kwargs):
    """
    Transforms the input array based on input mapping, ellipsis (`...`), and additional parameters for parentheses expansions.

    Parameters:
    - array: np.ndarray, the input array to transform.
    - pattern: str, the input pattern.
    - input_mapping: dict, maps each dimension name in the pattern to its index in the array.
    - input_shape_mapping: dict, maps each dimension name in the pattern to its size in the array.
    - kwargs: additional arguments for dimensions inside parentheses, e.g., h1, h, w1, w.

    Returns:
    - np.ndarray: transformed array.
    """
    original_shape = list(array.shape)
    new_shape = []

    for key, index in input_mapping.items():
        if key == '...':  # Handle ellipsis
            ellipsis_dims = [original_shape[i] for i in index]
            new_shape.extend(ellipsis_dims)
        elif '(' in key and ')' in key:  # Handle parentheses
            # Get the components inside the parentheses
            components = key.strip('()').split()

            # Ensure the components exist in the kwargs
            if not all(dim in kwargs for dim in components):
                raise ValueError(f"Missing dimensions {components} for expanding '{key}'.")

            # Replace the single dimension with the expanded dimensions
            expanded_dims = [kwargs[dim] for dim in components]
            new_shape.extend(expanded_dims)
        else:
            # Add the size of the dimension directly from the input shape
            new_shape.append(original_shape[index])

    # Reshape the array
    transformed_array = array.reshape(*new_shape)
    return transformed_array

def update_input_tokens_mapping(input_tokens_mapping, **kwargs):
    """
    Expands the input tokens mapping based on additional arguments, while preserving ellipsis (`...`) position.

    Parameters:
    - input_tokens_mapping: dict, maps tokens to indices in the array.
    - kwargs: dict, additional arguments for expanding dimensions (e.g., h, w, etc.).

    Returns:
    - dict: Expanded token mapping with proper indices for all tokens, including ellipsis.
    """
    expanded_tokens_mapping = {}
    current_index = 0  # Track the current index in the mapping

    for token, index in input_tokens_mapping.items():
        if token == '...':  # Handle ellipsis
            if index == []:  # If ellipsis is empty, keep it as is
                expanded_tokens_mapping[token] = index
            else:
                # Preserve the ellipsis with its original indices
                expanded_tokens_mapping[token] = index
                current_index = max(index) + 1 if index else current_index
        elif '(' in token and ')' in token:  # Handle grouped tokens like '(h w)'
            # Extract components inside parentheses
            components = token.strip('()').split()

            # Ensure all components are present in kwargs
            if not all(dim in kwargs for dim in components):
                raise ValueError(f"Missing dimensions {components} for expanding '{token}'.")

            # Expand the components and assign sequential indices
            for dim in components:
                expanded_tokens_mapping[dim] = current_index
                current_index += 1
        else:
            # Assign sequential indices for direct tokens
            expanded_tokens_mapping[token] = current_index
            current_index += 1

    return expanded_tokens_mapping

class Output_Transformations:
    def __init__(self, array, token_mapping, output_mapping):
        self.array = array
        self.token_mapping = token_mapping
        self.output_mapping = output_mapping

        self.output_order = list(self.output_mapping.keys())
        self.input_order = list(self.token_mapping.keys())

        singleton_values = [
                int(key.split("_")[1])  # Extract and convert the number part to an integer
                for key in list(self.token_mapping.keys())
                if key.startswith("singleton_")  # Check if the key starts with "singleton_"
            ]

        self.last_singleton_value = max(singleton_values, default=0)

        self.last_index_val = max(
                    max(val) if isinstance(val, list) and len(val) > 0 else val
                    for val in self.token_mapping.values()
                )

        # print("Last Singleton Value: ", self.last_singleton_value)

    def remove_singleton(self):
        """
        If singleton is in input mapping, but is not needed in output mapping, then remove it from input mapping and array.
        """

        remove_arr = []

        for token in self.input_order:
            if "singleton" in token:
                if token not in self.stripped_order(self.output_order):
                    remove_arr.append(self.token_mapping[token])
                    del self.token_mapping[token]

        if len(remove_arr) > 0:
            # print("Removing singletons")
            # print(remove_arr)
            for count, key in enumerate(self.token_mapping.keys()):
                self.token_mapping[key] = count
            # print(self.token_mapping)
            self.array = self.array.squeeze(axis=tuple(remove_arr))

    def stripped_order(self, d):
        return " ".join(d).replace("(", "").replace(")", "").split(" ")

    def add_singleton(self):
        """
        If singleton is in output mapping, but is not in input mapping, then add it to output array shape.
        """

        count = 0
        for token in self.stripped_order(self.output_order):
            if "singleton" in token:
                if token not in self.stripped_order(self.input_order):
                    count += 1
                    self.token_mapping["singleton_"+str(self.last_singleton_value+count)] = self.last_index_val + count

        new_shape = list(self.array.shape)

        for i in range(count):
            new_shape.append(1)

        self.array = self.array.reshape(*new_shape)

    def reorder_array(self):
        """
        Based on the order of output identifiers (after removing paranthesis), re-order the array.
        """

        output_order_stripped = self.stripped_order(self.output_order)
        # print(output_order_stripped)

        reshape_order = []
        for token in output_order_stripped:
            if '...' in token:
                reshape_order += self.token_mapping[token]
            else:
                reshape_order.append(self.token_mapping[token])

        # print(reshape_order)
        self.reshaped_array = self.array.transpose(*reshape_order)

    def resum_array(self):
        """
        Based on if paranthesis exist in output, reshape the array.
        """

        s = self.array.shape

        new_order = []
        for token in self.output_order:
            if "(" in token:
                inner_tokens = token.strip('()').split()
                # print(inner_tokens)
                new_shape = prod([s[self.token_mapping[inner_t]] for inner_t in inner_tokens])
                new_order.append(new_shape)
            elif '...' in token:
                new_order += [s[inner_t] for inner_t in self.token_mapping[token]]
            else:
                new_order.append(s[self.token_mapping[token]])
        # print(new_order)
        self.resummed_array = self.reshaped_array.reshape(*new_order)
        # print("Array resummed: ", self.resummed_array.shape)

    def _requires_reshaping(self):
        """
        Checks to see if paranthesis exist in output.
        """

        paranth_flag = False
        for token in self.output_order:
            if("(" in token):
                paranth_flag = True

        return paranth_flag

    def transform(self):
        """
        1. Calls remove singleton function
        2. Calls add singleton function
        3. Calls reorder array function
        4. Calls resum array function is reshaping is required
        """

        self.remove_singleton()
        self.add_singleton()

        # print(self.array.shape)

        self.reorder_array()

        # print("Array reshaped: ", self.reshaped_array.shape)

        if self._requires_reshaping():
            self.resum_array()   # Step 4: Reshape grouped dimensions
            return self.resummed_array
        else:
            return self.reshaped_array