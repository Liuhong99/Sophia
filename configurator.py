class Config:
    def __init__(self, config_file=None, **kwargs):
        if config_file is not None:
            with open(config_file) as f:
                print(f"Overriding config with {config_file}:")
                print(f.read())
                exec(f.read(), self.__dict__)

        for key, val in kwargs.items():
            try:
                attempt = literal_eval(val)
            except (SyntaxError, ValueError):
                attempt = val
            if hasattr(self, key):
                assert isinstance(attempt, type(getattr(self, key)))
            print(f"Overriding: {key} = {attempt}")
            setattr(self, key, attempt)
