from ..choice import Choice


class MixedChoice(Choice):
    def __init__(self, dir_choice: Choice, choices: list[Choice], epsilon=1e-11):
        super().__init__()
        self.choice = dir_choice
        self.choices = choices

        self._dir_params = dir_choice.count_params()
        self._segments = []
        offset = self._dir_params
        for choice in choices:
            end = offset + choice.count_params()
            self._segments.append(slice(offset, end))
            offset = end
        self._total_params = offset

    def count_params(self):
        return self._total_params

    def __call__(self, n, params):
        selector_params = self.choice.apply_constraints(params[:self._dir_params])
        index = self.choice(len(self.choices), selector_params)

        selected_choice = self.choices[index]
        selected_params = selected_choice.apply_constraints(params[self._segments[index]])
        return selected_choice(n, selected_params)

    def apply_constraints(self, params):
        constrained = list(self.choice.apply_constraints(params[:self._dir_params]))
        for choice, segment in zip(self.choices, self._segments):
            constrained.extend(choice.apply_constraints(params[segment]))
        return constrained