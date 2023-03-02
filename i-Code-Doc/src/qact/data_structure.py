from dataclasses import dataclass, asdict


@dataclass
class PromptStep:
    name: str
    args: dict
    type: str

    def __repr__(self):
        # To visualize the prompt easier when debugging
        return f'{self.name}{": " + self.args["string"] if "string" in self.args and self.args["string"] else ""}'
    
    # Create a dict to be used as a json
    def to_dict(self):
        return asdict(self)

    # Create class from dict
    @classmethod
    def from_dict(cls, data):
        return cls(**data)

@dataclass
class UDOPExample:
    instruction_history: 'list[PromptStep]'
    screenshot: str
    step: PromptStep

    def __repr__(self):
        # To visualize the example easier when debugging
        return f'Instr: {self.instruction_history}, Act: {self.step}'

    # Create a dict to be used as a json
    def to_dict(self):
        return asdict(self)
