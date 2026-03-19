# Ideas for the project

## Tasks(priority), task

### General

### core

### AutoML+

- (High) Make the automlplus tools more modular and extensible so it would work for other similar tasks
  - Separate the automlplus/ tools from imagetools.py, utils.py, website_accessibility into three types of tasks - vlm, text, static
  - Move all general things like extract_text_from_html_bytes into the utils
  - Static would be something like ReadabilityAnalyzer that uses textstat, vlm (vision + llm) would be something like AltTextChecker
  - Website accessibility is a special case of image+static - since it uses an LLM call + static code tools so refactor that accordingly
  - In automlplus/router.py, the post urls should be the same, just that it needs to call more modular functions

### Tabular

### Vision

- (LOW) Add how much environmental impact the trained model had
  - For automl+ its a bit hard to tell because most of it is an API call and nothing is trained, so just give an estimate of how much a gpt-4o mini token length was used

### Future (ignore for now)

- (VERY LOW) agentic tool based selection for the given task

## Plan
