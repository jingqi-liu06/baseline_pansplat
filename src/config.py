from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Type, TypeVar

from dacite import Config, from_dict
from omegaconf import DictConfig, OmegaConf

from .dataset.data_module import DataLoaderCfg, DatasetCfg
from .loss import LossCfgWrapper
from .model.decoder import DecoderCfg
from .model.encoder import EncoderCfg
from .model.model_wrapper import OptimizerCfg, TestCfg, TrainCfg, ValCfg, PredictCfg


@dataclass
class CheckpointingCfg:
    load: Optional[str]  # Not a path, since it could be something like wandb://...
    every_n_train_steps: int | None
    train_time_interval: int | None
    pretrained_model: Optional[str]


@dataclass
class ModelCfg:
    weights_path: Optional[Path]
    decoder: DecoderCfg
    encoder: EncoderCfg
    wo_defbp2: bool = False


@dataclass
class TrainerCfg:
    max_epochs: int | None
    max_steps: int | None
    val_check_interval: int | float | None
    check_val_every_n_epoch: int | None
    gradient_clip_val: int | float | None
    num_sanity_val_steps: int
    limit_train_batches: int | float | None
    limit_val_batches: int | float | None
    limit_test_batches: int | float | None
    limit_predict_batches: int | float | None
    precision: int | str | None
    deterministic: bool


# 这是整个项目的主配置类，包含所有子配置
@dataclass
class RootCfg:
    wandb: dict
    mode: Literal["train", "test", "predict", "all"]
    dataset: DatasetCfg
    data_loader: DataLoaderCfg
    model: ModelCfg
    optimizer: OptimizerCfg
    checkpointing: CheckpointingCfg
    trainer: TrainerCfg
    loss: list[LossCfgWrapper]
    test: TestCfg
    train: TrainCfg
    val: ValCfg
    predict: PredictCfg
    seed: int
    mvs_only: bool


TYPE_HOOKS = {
    Path: Path,
}


T = TypeVar("T")


def load_typed_config(
    cfg: DictConfig, 
    data_class: Type[T], 
    extra_type_hooks: dict = {},
    ) -> T:
    """
    将一个原始的、无类型的配置字典 (cfg)，
    填充到一个指定类型的 dataclass 模板 (data_class) 中，
    并返回一个填充好的、类型严格的 dataclass 实例
    假设你有一个字典 {'max_epochs': 10, 'deterministic': True} 
    和一个 TrainerCfg 的 dataclass，
    调用 load_typed_config 就会返回一个 
    TrainerCfg(max_epochs=10, deterministic=True) 的对象。
    """
    # from_dict(...)是 dacite 库的核心功能。
    # 它接收一个 ** dataclass类和一个字典 **，
    # 然后尝试创建一个该类的实例，并将字典中的值赋给对应的字段
    return from_dict(
        data_class,
        OmegaConf.to_container(cfg),
        config=Config(type_hooks={**TYPE_HOOKS, **extra_type_hooks}),
    )


def separate_loss_cfg_wrappers(joined: dict) -> list[LossCfgWrapper]:
    # The dummy allows the union to be converted.
    # 1. 定义一个临时的 "包裹" dataclass
    @dataclass
    class Dummy:
        dummy: LossCfgWrapper

    # 2. 列表推导式，遍历 YAML 中的 loss 字典
    return [
        load_typed_config(DictConfig({"dummy": {k: v}}), Dummy).dummy
        for k, v in joined.items()
    ]
    # k = "mse", v = {"weight": 1.0}
    # 步骤1：创建临时结构
    # temp_dict = {"dummy": {"mse": {"weight": 1.0}}}
    # temp_config = DictConfig(temp_dict)
    # 步骤2：调用load_typed_config
    # result = load_typed_config(temp_config, Dummy)
    # Dummy(dummy={"mse": {"weight": 1.0}})

    # - dacite看到Dummy类有一个dummy字段，类型是LossCfgWrapper
    # - LossCfgWrapper是联合类型：LossMseCfgWrapper | LossLpipsCfgWrapper | ...
    # - dacite看到数据{"mse": {"weight": 1.0}}有唯一键"mse"
    # - 它知道"mse"对应LossMseCfgWrapper类型
    # - 所以创建：LossMseCfgWrapper(mse=LossMseCfg(weight=1.0))
    '''
    e.g. 输出是：
    [
    LossMseCfgWrapper(mse=LossMseCfg(weight=1.0)),
    LossLpipsCfgWrapper(lpips=LossLpipsCfg(weight=0.1, apply_after_step=1000))
    ]
    '''


def load_typed_root_config(cfg: DictConfig) -> RootCfg:
    return load_typed_config(
        cfg,
        RootCfg,
        {list[LossCfgWrapper]: separate_loss_cfg_wrappers}, #这是一个类型钩子（type hook）字典，告诉 dacite 库：当遇到 list[LossCfgWrapper] 类型时，使用 separate_loss_cfg_wrappers 函数来处理转换
    )
