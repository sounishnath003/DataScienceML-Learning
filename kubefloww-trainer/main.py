import sys
import logging
from config import Configuration

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, handlers=[
        logging.StreamHandler()
    ])
    LOG=logging.getLogger('kfp_trainer')
    LOG.setLevel(logging.INFO)

    LOG.info("running kubeflow = {}".format(Configuration.KubeflowArtifactStore))

