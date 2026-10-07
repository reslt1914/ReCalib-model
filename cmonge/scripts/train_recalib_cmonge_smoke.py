from pathlib import Path

from loguru import logger

from cmonge.datasets.conditional_loader import (
    ConditionalDataModule,
)
from cmonge.datasets.recalib_loader import (
    ReCalibSciPlexModule,
)
from cmonge.trainers.ae_trainer import (
    AETrainerModule,
)
from cmonge.trainers.conditional_monge_trainer import (
    ConditionalMongeTrainer,
)
from cmonge.utils import load_config


CONFIG = Path(
    "configs/recalib_sciplex_smoke.yml"
)

LOG = Path(
    "logs/recalib_sciplex_smoke.yml"
)


# Use our ReCalib-compatible SciPlex reader
# without modifying the original CMonge source.
ConditionalDataModule.datamodule_factory[
    "sciplex"
] = ReCalibSciPlexModule


config = load_config(CONFIG)

logger.info(
    f"REAL SCIPLEX SMOKE conditions: "
    f"{config.condition.conditions}"
)


# ============================================================
# Stage 1: AutoEncoder
# ============================================================

config.data.ae = True
config.data.reduction = None

datamodule = ConditionalDataModule(
    config.data,
    config.condition,
)

ae_trainer = AETrainerModule(
    config.ae
)

ae_trainer.train(
    datamodule
)


# ============================================================
# Stage 2: Conditional Monge
# ============================================================

config.data.ae = False

# AETrainerModule changes act_fn internally.
# Reset it exactly as the official demo script does.
config.ae.model.act_fn = "gelu"

config.data.reduction = "ae"

datamodule = ConditionalDataModule(
    config.data,
    config.condition,
    ae_config=config.ae,
)

trainer = ConditionalMongeTrainer(
    jobid=1,
    logger_path=LOG,
    config=config.model,
    datamodule=datamodule,
)

trainer.train(
    datamodule
)

trainer.evaluate(
    datamodule
)

print()
print("=" * 90)
print("REAL RECALIB SCIPLEX CMONGE SMOKE PASSED")
print("=" * 90)
