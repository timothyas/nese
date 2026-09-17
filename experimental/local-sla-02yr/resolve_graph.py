from omegaconf import OmegaConf

if __name__ == "__main__":

    conf = OmegaConf.load("training.yaml")
    OmegaConf.resolve(conf)
    OmegaConf.save(conf.graph, "graph.resolved.yaml")
