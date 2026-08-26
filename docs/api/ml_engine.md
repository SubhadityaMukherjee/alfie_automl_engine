# ML Engine

The consolidated engine package (`app/ml_engine`) that performs all model
training for the vision, audio, text, and tabular endpoints.

## Generic engine (Optuna + Lightning Fabric over Hugging Face models)

::: app.ml_engine.tasks

::: app.ml_engine.configs

::: app.ml_engine.dataset

::: app.ml_engine.datamodule

::: app.ml_engine.model

::: app.ml_engine.trainer

::: app.ml_engine.hpo.optuna_objectives

::: app.ml_engine.feature_mapping

::: app.ml_engine.model_search

The tabular (AutoGluon) pieces live in the same modules:
`AutoGluonTrainer` and the tabular defaults in `app.ml_engine.trainer`, and
the tabular task models + `SUPPORTED_TABULAR_TASK_TYPES` in
`app.ml_engine.model`.
