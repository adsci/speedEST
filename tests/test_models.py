import pytest
from src.utils import preprocess_raw_input


class TestModelLoad:
    @pytest.mark.parametrize("model_name",
                             ["tre",
                              "mlp",
                              "rle",
                              "sve",
                              "fve"])
    def test_import_model(self, model_name):
        try:
            module = __import__("src.models", fromlist=[model_name])
            assert hasattr(module, model_name), f"{model_name} not found in {module}."
        except ImportError:
            pytest.fail(f"Failed to import {model_name}")


class TestModelPredictions:
    @pytest.mark.parametrize("model_name, input",
                             [
                                 ("tre", [1300, 17, 800, 2, 1]),
                                 ("mlp", [1300, 17, 800, 2, 1]),
                                 ("rle", [1300, 17, 800, 2, 1]),
                                 ("sve", [1300, 17, 800, 2, 1])
                             ])
    def test_model_prediction(self, model_name, input):
        module = __import__("src.models", fromlist=[model_name])
        model = getattr(module, model_name)

        query = preprocess_raw_input(*input)
        prediction, flag = model.predict(query)
        assert flag
        assert 70 <= prediction <= 150

    @pytest.mark.parametrize("input",
                             [
                                 ([1300, 17, 800, 2, 1])
                             ])
    def test_ensemble_prediction(self, input):
        module = __import__("src.models", fromlist=["fve"])
        model = "fve"
        model = getattr(module, model)

        query = preprocess_raw_input(*input)
        _, prediction = model.ensemble_predict(query)
        assert 70 <= prediction <= 150
