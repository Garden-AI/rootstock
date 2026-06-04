# Clusters

Rootstock is deployed on a growing set of HPC clusters. The current model-by-cluster coverage — which checkpoints are installed and verified where — lives in the [Matter Model Almanac](https://garden-ai.github.io/almanac). Start there to find a model that runs on your cluster.

## Live manifest

The dashboard exposes a JSON manifest per cluster at
[`garden-ai-prod--rootstock-admin-dashboard.modal.run`](https://garden-ai-prod--rootstock-admin-dashboard.modal.run/). Each manifest lists the install root, the built environments, their Python and dependency versions, the verified checkpoints, and the environment source file.

This is the raw, current state. Maintainers standing up a new cluster can copy a working environment source from a manifest as a starting point — see [Writing Environments](environments.md) and [Cluster Setup](cluster-setup.md).
