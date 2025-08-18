from .automltabular import (TabularSupervisedClassificationTask,
                            TabularSupervisedRegressionTask,
                            TabularSupervisedTimeSeriesTask)
from .general_llm_question import GeneralLLMPipeline
from .website_accessibility import WebsiteAccesibilityPipeline

PIPELINES = {
    WebsiteAccesibilityPipeline.__name__: WebsiteAccesibilityPipeline,
    TabularSupervisedRegressionTask.__name__: TabularSupervisedRegressionTask,
    TabularSupervisedClassificationTask.__name__: TabularSupervisedClassificationTask,
    TabularSupervisedTimeSeriesTask.__name__: TabularSupervisedTimeSeriesTask,
    GeneralLLMPipeline.__name__: GeneralLLMPipeline,
}
