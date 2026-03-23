# Ideas for the project

## Tasks(priority), task

### General

### core

### AutoML+

### Tabular

### Vision

- (HIGH) refactor training for other types of models
  - Make a list of task types (from the list I gave for model.py), and for each task type do the following where needed (eg: classification)
  - In trainer.py, the optuna_objective at the moment is very specific for image classification, I need to to support more classes (as detailed below)
    - Create a folder with config files for different task types, use json to store things like - small/medium/big model lists, lr suggestions, batch size suggestions etc
    - Move the current optuna_objective to optuna_objective_for_classification (and similarly for the other types)
    - in router.py, update the find_best_model_for_vision, where the task_type is one of the list of task types (eg: classification), and also pass this into the train_automl function
    - in datamodule, add the relevant dataloaders
    - update the train_automl function to accept this type of task and decide which objective to choose accordingly
  - in model.py, add classes for - AutoModelForCausalLM,AutoModelForSeq2SeqLM, AutoModelForMaskedLM, AutoModelForSequenceClassification, AutoModelForQuestionAnswering, AutoModelForVideoClassification, AutoModelForKeypointDetection, AutoModelForObjectDetection, AutoModelForImageSegmentation, AutoModelForAudioClassification
  - update the types in models.py accordingly
  - in services.py, update the validate_vision_inputs function accord
  - in services.py, add the task type to the build_upload_payload
  - Update the tests accordingly in tests/test_vision/
  - Update the docs to source the functions that were created here in docs/api/vision_automl.py

- (LOW) Add how much environmental impact the trained model had
  - For automl+ its a bit hard to tell because most of it is an API call and nothing is trained, so just give an estimate of how much a gpt-4o mini token length was used

### Future (ignore for now)

- (VERY LOW) agentic tool based selection for the given task

## Plan
