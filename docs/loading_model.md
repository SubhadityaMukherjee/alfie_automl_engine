# Loading a model

- Once the AutoML tool is done, it will point you to a folder with the results/or you can find the results.zip from AutoDW

## Tabular dataset
- Once you have this folder (extract the zip if it is one), you can simply do
```
from autogluon.tabular import TabularPredictor
predictor = TabularPredictor.load(predictor_path)
```
- Now to predict something you can do, `predictor.predict(test_data)` or `predictor.predict(test_data, model = 'X')` for a specific model
- For more instructions on how to use this, please refer to the [AutoGluon documentation page](https://auto.gluon.ai/stable/tutorials/tabular/tabular-essentials.html#loading-a-trained-predictor)

## Vision Dataset
- This is a WIP
