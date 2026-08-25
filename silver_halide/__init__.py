from .pipeline import Develop, Recipe


def develop(recipe: Recipe, source: str, target: str) -> dict:
    return Develop(recipe).run(source, target)
