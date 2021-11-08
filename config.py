import logging
import yaml

logging.basicConfig(
    format="%(asctime)s.%(msecs)03d %(message)s",
    datefmt="%Y-%m-%d,%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def load_config(path):

    with open(path) as f:
        config = yaml.load(f, Loader=yaml.Loader)

    return config
