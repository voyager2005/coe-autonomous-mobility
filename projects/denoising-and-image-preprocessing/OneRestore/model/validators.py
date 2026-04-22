from .einops_utils import to_numpy_array, unexpected_chars_checker, clean_singletons_in_parentheses, _tokenize

class Validator:
    def __init__(self, array, pattern, **kwargs):
        self.array = to_numpy_array(array)
        self._is_empty_array()
        unexpected_chars_checker(pattern)
        self.pattern = clean_singletons_in_parentheses(pattern)
        self.kwargs = kwargs
        self.array_shape = self.array.shape

        self.input_str, self.output_str = self._parse_pattern()
        self.input_tokens = _tokenize(self.input_str)
        self.output_tokens = _tokenize(self.output_str)

    def _is_empty_array(self):

        """
        Checks if the given NumPy array is empty.

        Args:
            array (numpy.ndarray): The input array to check.

        Raises:
            ValueError: If the array is empty.
        """
        if self.array.size == 0:
            raise ValueError("The input NumPy array is empty. Please provide a valid array.")

    def _parse_pattern(self):
        """
        Parses the pattern and splits it into input and output parts.

        Returns:
            tuple: A tuple containing the input and output parts of the pattern.
        """

        try:
            input_str, output_str = self.pattern.split('->')
            return input_str.strip(), output_str.strip()

        except ValueError:
            raise ValueError("Pattern must be in the form 'input -> output'")

    def stripped_order(self, d):
        """
        Removes all paranthesis from a list and returns a list

        ["a", "(b c)"] -> ["a", "b", "c"]
        """

        return " ".join(d).replace("(", "").replace(")", "").split(" ")

    def ellipsis_checker(self):
        """
        Checks if more than ellipsis is in input or output pattern.
        Raises ValueError if found.
        """

        if self.input_tokens.count('...') > 1 or self.output_tokens.count('...') > 1:
            raise ValueError("Expression may contain dots only inside ellipsis (...); only one ellipsis for tensor")

    def identifier_match_checker(self):
        """
        Checks for discrepancies between identifiers in the input and output pattern.
        Raises ValeError if there are differences in identifiers or if some identifiers are duplicated.
        """

        input_tokens_stripped = self.stripped_order(self.input_tokens)
        output_tokens_stripped = self.stripped_order(self.output_tokens)

        input_tokens_stripped_filtered = [item for item in input_tokens_stripped if item != '1']
        output_tokens_stripped_filtered = [item for item in output_tokens_stripped if item != '1']

        if len(set(input_tokens_stripped_filtered)) != len(input_tokens_stripped_filtered):
            raise ValueError(f"Input pattern {self.input_str} contains duplicate dimension")

        if len(set(output_tokens_stripped_filtered)) != len(output_tokens_stripped_filtered):
            raise ValueError(f"Input pattern {self.output_str} contains duplicate dimension")

        missing_in_output = set(input_tokens_stripped_filtered) - set(output_tokens_stripped_filtered)
        extra_in_output = set(output_tokens_stripped_filtered) - set(input_tokens_stripped_filtered)

        if missing_in_output:
            raise ValueError(f"Identifiers only on one side of expression (should be on both): {missing_in_output}")
        if extra_in_output:
            raise ValueError(f"Identifiers only on one side of expression (should be on both): {extra_in_output}")

    def input_token_mapper(self):
        """
        Map the input numpy array shape index positions to the input part of the pattern string.

        Input -
            array = np.random.randn(2, 3, 4)
            input_tokens = ["a", "b", "c"]
        Output -
            input_tokens_mapping = {
                'a': 0,
                'b': 1,
                'c': 2
            }
            input_tokens_shape_mapping = {
                'a': 2,
                'b': 3,
                'c': 4
            }


        """

        ellipsis_count = self.input_tokens.count('...')
        non_ellipsis_tokens = [tok for tok in self.input_tokens if tok != '...']

        if ellipsis_count > 1:
            raise ValueError("Pattern can have at most one ellipsis ('...').")

        # Check if token count matches array dimensions (unless ellipsis exists)
        if ellipsis_count == 0 and len(self.input_tokens) != len(self.array_shape):
            raise ValueError(
                f"Number of input tokens ({len(self.input_tokens)}) must match the array dimensions ({len(self.array_shape)}) unless using ellipsis ('...')."
            )
        if ellipsis_count == 1 and len(non_ellipsis_tokens) > len(self.array_shape):
            raise ValueError(
                f"Pattern with ellipsis ('...') must not have more explicit tokens ({len(non_ellipsis_tokens)}) than array dimensions ({len(self.array_shape)})."
            )

       # Assign index positions to tokens
        input_tokens_mapping = {}
        array_shape_indices = list(range(len(self.array_shape)))  # [0, 1, 2, ..., len(shape)-1]
        input_tokens_shape_mapping = {}

        if '...' in self.input_tokens:
            # Handle ellipsis (map it to remaining index positions)
            ellipsis_index = self.input_tokens.index('...')
            ellipsis_dims = len(array_shape_indices) - len(non_ellipsis_tokens)
            if ellipsis_dims < 0:
                raise ValueError(
                    f"Ellipsis ('...') is invalid: not enough array dimensions to map remaining tokens."
                )
            ellipsis_mapping = array_shape_indices[:ellipsis_dims]
            input_tokens_mapping['...'] = ellipsis_mapping
            array_shape_indices = array_shape_indices[ellipsis_dims:]  # Remove mapped indices
            input_tokens_shape_mapping['...'] = ellipsis_mapping

        # Validate and map remaining tokens to index positions

        singleton_count = 0

        for token, index in zip(non_ellipsis_tokens, array_shape_indices):
            if token == '1':
                # Validate that the corresponding dimension is 1
                if self.array_shape[index] != 1:
                    raise ValueError(
                        f"Dimension for token '1' must be 1, but got {self.array_shape[index]} at index {index}."
                    )
                singleton_count += 1
                input_tokens_mapping["singleton_"+ str(singleton_count)] = index
                input_tokens_shape_mapping["singleton_"+ str(singleton_count)] = self.array_shape[index]
            else:
                input_tokens_mapping[token] = index
                input_tokens_shape_mapping[token] = self.array_shape[index]

        self.input_tokens_mapping = input_tokens_mapping
        self.input_tokens_shape_mapping = input_tokens_shape_mapping
        # print("Input Token Mapping: ", self.input_tokens_mapping)
        # print("Input Token Shape Mapping: ", self.input_tokens_shape_mapping)

    def output_tokens_mapper(self):
        """
        Create mapping from output part of the pattern string.

        Input -
            output_tokens = ["a", "b", "c"]
        Output -
            output_tokens_mapping = {
                'a': 0,
                'b': 1,
                'c': 2
            }
        """

        _, output_str = self.pattern.split('->')
        self.output_str = output_str.strip()
        self.output_tokens = _tokenize(output_str)

        # Map output tokens to index positions
        output_tokens_mapping = {}
        current_index = 0

        singleton_count = 0
        for token in self.output_tokens:
            # if token == '...':
            #     output_tokens_mapping[token] = current_index
            #     current_index += 1
            if token == '1':
                singleton_count += 1
                # Fixed dimension (explicitly adds dimension of size 1)
                output_tokens_mapping["singleton_"+str(singleton_count)] = current_index
                current_index += 1
            else:
                # Map single axis
                output_tokens_mapping[token] = current_index
                current_index += 1

        self.output_tokens_mapping = output_tokens_mapping
        # print("Output Token Mapping: ", self.output_tokens_mapping)

    def empty_ellipsis_checker(self):
        """
        Ellipsis in the input can be mapped to an empty list i.e. there are no axes that it points to.
        In cases like this, ellipsis becomes unnecessary to handle. So, this function removes them from the mappings and updates the pattern
        """

        if('...') in list(self.input_tokens_mapping.keys()) and len(self.input_tokens_mapping['...']) == 0:
            del self.input_tokens_mapping['...']
            del self.input_tokens_shape_mapping['...']
            new_pattern = self.pattern.replace('...', '')

            self.pattern = new_pattern.lstrip().rstrip()

    def validate_and_return(self):
        """
        Validates the pattern, tokens, and array shape, then returns mappings.
        """

        self.ellipsis_checker()
        self.identifier_match_checker()
        self.input_token_mapper()
        self.empty_ellipsis_checker()
        self.output_tokens_mapper()

        return (
            self.array,
            self.input_tokens_mapping,
            self.input_tokens_shape_mapping,
            self.output_tokens_mapping,
        )