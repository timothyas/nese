from omegaconf import OmegaConf

if __name__ == "__main__":

    conf = OmegaConf.load("training.yaml")
    # Only the graph subtree, so unrelated interpolations (SLURM env vars) stay untouched.
    graph = OmegaConf.to_container(conf.graph, resolve=True)
    OmegaConf.save(OmegaConf.create(graph), "graph.resolved.yaml")
