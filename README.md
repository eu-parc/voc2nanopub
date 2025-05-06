# LinkML to Nanopub Publishing Workflow

This reusable GitHub workflow converts vocabulary terms defined in LinkML schemas to nanopublications and publishes them to a nanopub server. Nanopublications are small units of publishable information represented as RDF triples with provenance information, making your vocabulary terms citable, discoverable, and reusable.

## Overview

The workflow performs the following steps:
1. Checks out your repository at the specified reference
2. Sets up Python and installs required dependencies
3. Loads your LinkML schema and data file
4. Converts vocabulary terms to nanopublications
5. Publishes the nanopublications to a server (test or production)
6. Updates your data file with nanopublication IDs (unless in dry-run mode)

## Usage

To use this workflow in your GitHub repository, create a workflow file (e.g., `.github/workflows/publish-vocab.yml`) with the following structure:

```yaml
name: Publish Vocabulary

on:
  # Define your trigger events here (push, pull_request, workflow_dispatch, etc.)
  workflow_dispatch:
    inputs:
      ref:
        description: 'Branch, tag, or commit SHA to use'
        default: 'main'
        required: true
      server_mode:
        description: 'Server to publish to'
        type: choice
        options:
          - test
          - production
        default: 'test'

jobs:
  publish:
    uses: your-org/your-repo/.github/workflows/publish-vocabulary-nanopub.yml@main
    with:
      ref: ${{ inputs.ref || 'main' }}
      linkml_schema: 'path/to/your/schema.yaml'
      data: 'path/to/your/data.yaml'
      target_name: 'your_entity_list_name'
      preflabel: 'name'  # Field used to create identifier hash
      type_prefix: 'YourPrefix'  # Added to identifier after namespace
      server_mode: ${{ inputs.server_mode || 'test' }}
    secrets:
      token: ${{ secrets.GITHUB_TOKEN }}
      NANOPUB_ORCID_ID: ${{ secrets.NANOPUB_ORCID_ID }}
      NANOPUB_NAME: ${{ secrets.NANOPUB_NAME }}
      NANOPUB_PRIVATE_KEY: ${{ secrets.NANOPUB_PRIVATE_KEY }}
      NANOPUB_PUBLIC_KEY: ${{ secrets.NANOPUB_PUBLIC_KEY }}
      NANOPUB_INTRO_URI: ${{ secrets.NANOPUB_INTRO_URI }}
```

## Required Inputs

| Input | Description | Required |
|-------|-------------|----------|
| `ref` | Git reference (branch, tag, or commit SHA) to checkout | Yes |
| `linkml_schema` | Path to the LinkML schema file | Yes |
| `data` | Path to the data file containing entities to publish | Yes |
| `target_name` | Name of the target entity list in the data file | Yes |
| `preflabel` | Name of field used to create identifier using hash function | Yes |
| `type_prefix` | Default part added to identifier following namespace | Yes |

## Optional Inputs

| Input | Description | Default |
|-------|-------------|---------|
| `version` | Version of the vocabulary being published | '' |
| `server_mode` | Server mode: `test` or `production` | 'test' |
| `dry_run` | If true, signs nanopubs but does not publish them | false |
| `parent_key` | Parent key for topological sort | '' |
| `output` | Output path for modified YAML file | Same as input file |
| `output_pairs` | Output path for identifier nanopub pairs | '' |

## Required Secrets

| Secret | Description |
|--------|-------------|
| `token` | GitHub token for repository checkout |
| `NANOPUB_ORCID_ID` | ORCID ID for nanopub profile |
| `NANOPUB_NAME` | Name for nanopub profile |
| `NANOPUB_PRIVATE_KEY` | Private key for nanopub profile |
| `NANOPUB_PUBLIC_KEY` | Public key for nanopub profile |
| `NANOPUB_INTRO_URI` | Introduction nanopub URI |

## Setting Up Nanopub Credentials

Before using this workflow, you need to set up nanopub credentials. Follow these steps:

1. Install the nanopub client: `pip install nanopub`
2. Run setup: `nanopub-setup`
3. Follow the prompts to create a profile with your ORCID ID
4. Store the generated keys and profile information as GitHub secrets

## Data File Format

Your data file should be in YAML format and contain a list of vocabulary terms under the key specified by `target_name`. Each term should have at least the field specified by `preflabel`.

## Example Data File

```yaml
your_entity_list_name:
  - id: ""  # Will be populated with nanopub ID after publishing
    name: "Example Term 1"
    description: "This is an example term"
  - id: ""
    name: "Example Term 2"
    description: "Another example term"
```

## Output

After the workflow runs successfully:
1. Your terms will be published as nanopublications
2. The data file will be updated with nanopublication IDs (unless in dry-run mode)
3. A file containing .htaccess redirect information will be created.

## Script Source

The workflow uses the `publish_nanopubs.py` script, which it will:
1. Try to find in the `./rules/` directory of your repository
2. Download from the `eu-parc/voc2nanopub` repository if not found locally

## License

This workflow is provided under the terms of your repository's license.