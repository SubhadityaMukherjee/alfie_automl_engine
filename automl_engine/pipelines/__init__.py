from .website_accessibility import WebsiteAccesibilityPipeline
from .automltabular import (
    TabularSupervisedClassificationTask,
    TabularSupervisedTimeSeriesTask,
    TabularSupervisedRegressionTask,
)
from .general_llm_question import GeneralLLMPipeline

PIPELINES = {
    WebsiteAccesibilityPipeline.__name__: WebsiteAccesibilityPipeline,
    TabularSupervisedRegressionTask.__name__: TabularSupervisedRegressionTask,
    TabularSupervisedClassificationTask.__name__: TabularSupervisedClassificationTask,
    TabularSupervisedTimeSeriesTask.__name__: TabularSupervisedTimeSeriesTask,
    GeneralLLMPipeline.__name__: GeneralLLMPipeline,

}
