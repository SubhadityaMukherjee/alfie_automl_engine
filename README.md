# ALFIE AutoML Engine

- Branch off from develop not main!!!

An AutoML engine written for the ALFIE project with the following features

- AutoML for tabular
- AutoML for vision (this is a WIP)
- Website accessibility checker
- Image to website tool
- Part of a host of tools made for better and more informed generation of AI models

## Documentation

- For proper documentation, please refer [here](https://subhadityamukherjee.github.io/alfie_automl_engine/)
- For how the repo is structured and how the services fit together, see
  [docs/architecture.md](docs/architecture.md)

## Getting started

The documentation site's [index](docs/index.md) covers the full setup; the short
version:

1. [Install](docs/installation.md) the project and set up your `.env` from
   `.env.template`.
2. Install and run [AutoDW](docs/autodw.md) (not needed if you only want
   AutoML+).
3. Optionally generate [sample data](docs/index.md#generate-sample-data-optional)
   with `download_sample_data.py`.
4. [Run the services](docs/running_the_services.md) you need — tabular, vision,
   and/or AutoML+ (Docker instructions [here](docs/docker_instructions.md)).

Once training finishes, models can be loaded for inference following
[docs/loading_model.md](docs/loading_model.md). Tests are described in
[docs/testing.md](docs/testing.md).
