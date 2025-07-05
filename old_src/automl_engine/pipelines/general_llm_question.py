from automl_engine.pipelines.base import BasePipeline


class GeneralLLMPipeline(BasePipeline):
    """This pipeline is used when the user asks a question directly and nothing else. Do not assume anything. Eg queries: What is automl?, how do I train a model? How do I do XYZ, given X how do I do Y"""

    @staticmethod
    def get_required_files():
        return {}
