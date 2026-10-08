# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) 2023-2024, SwanLab. All rights reserved.
import os
import os.path as osp
import warnings
from typing import Dict, Optional, Union

from mmcv.utils import scandir
from mmcv.runner import HOOKS,master_only
from mmcv.runner.hooks import LoggerHook


@HOOKS.register_module()
class SwanlabLoggerHook(LoggerHook):
    """用于通过 SwanLab 记录指标和文件的类。

    此类需要预先安装 `swanlab`_ 。

    参数:
        init_kwargs (dict, optional): 一个包含初始化键的字典。更多初始化参数请参考
            https://swanlab.cn/guide/sdk/init。默认为 None。
        interval (int): 记录间隔（每 k 次迭代）。
            默认值: 10。
        ignore_last (bool): 如果每个 epoch 的最后几次迭代次数小于 `interval`，则忽略日志。
            默认值: True。
        reset_flag (bool): 记录后是否清除输出缓冲区。
            默认值: False。
        by_epoch (bool): 是否使用 EpochBasedRunner。
            默认值: True。
        with_step (bool): 如果为 True，将从 ``self.get_iters`` 记录步数。
            否则，将不记录步数。
            默认值: True。
        log_artifact (bool): 如果为 True，训练结束后 {work_dir} 中的文件将被上传到 swanlab。
            实验目录下将创建 `files` 子目录来存储这些文件。
            默认值: True。
        out_suffix (str or tuple[str], optional): 以 ``out_suffix`` 结尾的文件
            将被上传到 swanlab。
            默认值: ('.log.json', '.log', '.py', '.pth')。

    .. _swanlab:
        https://swanlab.cn/
    """

    def __init__(self,
                 init_kwargs: Optional[Dict] = None,
                 interval: int = 50,
                 ignore_last: bool = True,
                 reset_flag: bool = False,
                 by_epoch: bool = True,
                 with_step: bool = True,
                 log_artifact: bool = True,
                 out_suffix: Union[str, tuple] = ('.log.json', '.log', '.py', '.pth')):
        super().__init__(interval, ignore_last, reset_flag, by_epoch)
        self.import_swanlab()
        self.init_kwargs = init_kwargs
        self.with_step = with_step
        self.log_artifact = log_artifact
        self.out_suffix = out_suffix

    def import_swanlab(self) -> None:
        """导入 `swanlab` 包，如果未安装则引发错误。"""
        try:
            import swanlab
        except ImportError:
            raise ImportError(
                '请运行 "pip install swanlab" 来安装 swanlab')
        self.swanlab = swanlab
        
    @master_only
    def before_run(self, runner) -> None:
        """初始化 swanlab 运行。"""
        super().before_run(runner)
        if self.swanlab is None:
            self.import_swanlab()

        self.swanlab.init(**self.init_kwargs) if self.init_kwargs else self.swanlab.init()


    @master_only
    def log(self, runner) -> None:
        """向 swanlab 记录指标。"""
        tags = self.get_loggable_tags(runner)
        if tags:
            if self.with_step:
                self.swanlab.log(tags, step=self.get_iter(runner))
            else:
                # SwanLab 的 log 函数在未提供步数时会隐式处理，
                # 但如果需要，我们也可以为了保持一致性手动记录。
                self.swanlab.log(tags)

    @master_only
    def after_run(self, runner) -> None:
        """记录文件并结束 swanlab 运行。"""
        self.swanlab.finish()
        # 在这种情况下，SwanLab 不需要显式的 join 或 finish 调用。
        # 当脚本退出时，运行会自动完成。
