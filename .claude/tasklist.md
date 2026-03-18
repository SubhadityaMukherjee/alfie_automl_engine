# Ideas for the project

## Tasks(priority), task

### General

- (HIGH) Use routers for the fastapi endpoints across all of the packages. But keep the services separated
- (HIGH) Add unified logging per service. Most of it already exists, but it should write to a rotating log file with logs per service

### core

### AutoML+

- (MEDIUM) Make the tools more modular and extensible so it would work for other similar tasks
  - Separate into Image tools, Language tools, general tools Combined tasks
  - Website accessibility is a special case of combined - since it uses an LLM call + static code tools

### Tabular

- ~~(HIGH) remove the old SQL bits from the API and tests, they were used before AutoDW. Now AutoDW is used to store everything~~ ✅ DONE

### Vision

- (LOW) Add how much environmental impact the trained model had
  - For automl+ its a bit hard to tell because most of it is an API call and nothing is trained, so just give an estimate of how much a gpt-4o mini token length was used

### Future (ignore for now)

- (VERY LOW) agentic tool based selection for the given task

## Plan
