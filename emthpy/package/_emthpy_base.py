
class Evaluable:
    def evaluate(self, *args, **kwargs):
        pass

    def evaluated(self, *args, **kwargs):
        pass

    def reduce(self, *args, **kwargs):
        pass

    def reduced(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self.evaluated(*args, **kwargs)