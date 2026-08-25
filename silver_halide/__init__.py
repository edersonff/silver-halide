import numpy as np

from .pipeline import Develop, Recipe


def develop(recipe: Recipe, source: str, target: str, texture_exemplar: str | None = None) -> dict:
    dev = Develop(recipe)
    if texture_exemplar:
        dev.micro = __import__("silver_halide.stages.microtexture", fromlist=["MicroTexture"]).MicroTexture(
            np.load(texture_exemplar), seed=recipe.seed
        )
    return dev.run(source, target)
