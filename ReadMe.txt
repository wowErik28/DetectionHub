继承汇总
1.utils 模块(数据处理,模型输入处理)
dataloader.AbstractDetectionDataset(必须实现接口)
data_processing.BaseDetectionDataProcessing(非必须实现接口)

2.nets 模块
#basenet.BaseDetectionNet
#basenet.BaseDetectionTrainer
对于基于torch框架的模型
torch_net.BaseTorchDetectionNet       #用于测试模型(必须实现接口)
torch_net.BaseTorchTrainDetectionNet  #用于训练模型(必须实现接口)



*建议最终算法模块结构(基于torch)

model-------
-----__init__.py     ModelTorchDetectionNet, ModelTorchTrainDetectionNet
-----config.py       Model_Default_Config
-----model_utils.py  ModelDetectionDataset, ModelDetectionDataProcessing, Dataset, ModelDetectionDataProcessing
-----model.py        ModelTorchDetectionNet, ModelTorchTrainDetectionNet
-----model_net.py    realize the model(nn.Module)
