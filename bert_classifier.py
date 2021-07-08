import logging

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import Callback
from pytorch_lightning import LightningModule
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchmetrics import Accuracy
from transformers import BertModel
from transformers import BertPreTrainedModel
from transformers import BertTokenizer
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%Y/%m/%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class CustomModel(BertPreTrainedModel):
    """自定义模型"""

    def __init__(self, config):
        super(CustomModel, self).__init__(config)
        self.bert = BertModel(config)  # transformers的写法，方便保存，加载模型
        self.classifier = nn.Linear(768, 2)  # 设置多分类的类别数，注意权重的初始化

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # sequence_output, pooler_output, hidden_states = outputs[0], outputs[1], outputs[2]
        y = self.classifier(outputs.pooler_output)

        return y


class CustomDataset(Dataset):
    """自定义Dataset，text为文本，label为标签"""

    def __init__(self, dataframe):
        self._data = dataframe
        self._sentence = self._data.sentence
        self._label = self._data.label

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        return self._sentence.iloc[index], self._label.iloc[index]


class CustomAccuracy(Accuracy):
    """基于torchmetrics自定义指标"""

    def __init__(self, save_metrics_history=True):
        super(CustomAccuracy, self).__init__()
        self.save_metrics_history = save_metrics_history  # 是否保留历史指标
        self.metrics_history = []  # 记录每个epoch_end计算的指标，方便checkpoint

    def compute_epoch_end(self):
        metrics = self.compute()
        if self.save_metrics_history:
            self.metrics_history.append(metrics.item())
        self.reset()

        return metrics


class ModelCheckpoint(Callback):
    def __init__(self, save_path='output', mode='max', patience=10):
        super(ModelCheckpoint, self).__init__()
        self.path = save_path
        self.mode = mode
        self.patience = patience
        self.check_patience = 0
        self.best_value = 0.0 if mode == 'max' else 1e6  # 记录验证集最优值

    def on_validation_epoch_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule'):
        """
        验证集计算结束后检查
        :param trainer:
        :param pl_module:
        :return:
        """
        if self.mode == 'max' and pl_module.valid_accuracy.metrics_history[-1] >= self.best_value:
            self.check_patience = 0
            self.best_value = pl_module.valid_accuracy.metrics_history[-1]
            logger.info(f'save best model with metric: {self.best_value:.5f}')
            pl_module.model.save_pretrained(self.path)  # 保存模型

        if self.mode == 'max' and pl_module.valid_accuracy.metrics_history[-1] < self.best_value:
            self.check_patience += 1

        if self.mode == 'min' and pl_module.valid_accuracy.metrics_history[-1] <= self.best_value:
            self.check_patience = 0
            self.best_value = pl_module.valid_accuracy.metrics_history[-1]
            logger.info(f'save best model with metric: {self.best_value:.5f}')
            pl_module.model.save_pretrained(self.path)  # 保存模型

        if self.mode == 'min' and pl_module.valid_accuracy.metrics_history[-1] > self.best_value:
            self.check_patience += 1

        if self.check_patience >= self.patience:
            trainer.should_stop = True  # 停止训练


class FGM(object):
    """fgm对抗训练，对embedding层添加扰动"""

    def __init__(self, model):
        self.model = model
        self.epsilon = 1.0
        self.backup = {}

    def attack(self, emb_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm and not torch.isnan(norm):
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class BertClassifier(LightningModule):
    """采用pytorch-lightning训练的分类器"""

    def __init__(self, train_data: pd.DataFrame, valid_data: pd.DataFrame, model_path: str = 'output'):
        super(BertClassifier, self).__init__()

        self.model = CustomModel.from_pretrained(model_path)  # 自定义的模型
        self.tokenizer = BertTokenizer.from_pretrained(model_path)  # 加载分词器

        self.train_dataset = CustomDataset(dataframe=train_data)  # 加载dataset
        self.valid_dataset = CustomDataset(dataframe=valid_data)

        self.max_length = 32  # 句子最大长度
        self.train_batch_size = 32
        self.valid_batch_size = 32

        self.train_accuracy = CustomAccuracy(save_metrics_history=False)  # 计算训练集的指标
        self.valid_accuracy = CustomAccuracy(save_metrics_history=True)  # 计算验证集的指标

        self.automatic_optimization = True  # 最好关闭，训练速度变快
        if not self.automatic_optimization:
            self.optimizer = self.configure_optimizers()[0]
            # self.optimizers, self.schedulers = self.configure_optimizers()
            # self.optimizer = self.optimizers[0]  # 初始化优化器
            # self.scheduler = self.schedulers[0]['scheduler']  # 初始化学习率策略

        self.scaler = torch.cuda.amp.GradScaler()  # 半精度训练

        self.use_attack = False  # 是否对抗训练
        if self.use_attack:
            self.fgm = FGM(self.model)

    def train_collate_batch(self, batch):
        """
        处理训练集batch，主要是文本转成相应的tokens
        :param batch:
        :return:
        """
        sentences = [sentence for sentence, _ in batch]
        outputs = self.tokenizer(sentences, truncation=True, padding=True, max_length=self.max_length,
                                 return_tensors='pt')

        labels = torch.tensor([label for _, label in batch], dtype=torch.int64)

        return outputs['input_ids'], outputs['attention_mask'], outputs['token_type_ids'], labels

    def val_collate_batch(self, batch):
        """
        :param batch:
        :return:
        """
        sentences = [sentence for sentence, _ in batch]
        outputs = self.tokenizer(sentences, truncation=True, padding=True, max_length=self.max_length,
                                 return_tensors='pt')

        labels = torch.tensor([label for _, label in batch], dtype=torch.int64)

        return outputs['input_ids'], outputs['attention_mask'], outputs['token_type_ids'], labels

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=self.train_collate_batch
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid_dataset,
            batch_size=self.valid_batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=self.val_collate_batch,
        )

    @staticmethod
    def compute_loss(y_pred, y_true):
        """
        计算loss
        :param y_pred:
        :param y_true:
        :return:
        """
        loss = F.cross_entropy(y_pred, y_true)

        return loss

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, y_true = batch
        y_pred = self.model(input_ids, attention_mask, token_type_ids)
        loss = self.compute_loss(y_pred, y_true)
        acc = self.train_accuracy(y_pred, y_true)

        if not self.automatic_optimization:
            self.optimizer.zero_grad()  # 梯度置零

            if self.use_amp and not self.use_attack:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

            elif self.use_amp and self.use_attack:
                self.scaler.scale(loss).backward()
                self.fgm.attack()

                y_pred = self.model(input_ids, attention_mask, token_type_ids)
                adv_loss = self.compute_loss(y_pred, y_true)
                self.scaler.scale(adv_loss).backward()
                self.fgm.restore()

                self.scaler.step(self.optimizer)
                self.scaler.update()

            elif not self.use_amp and not self.use_attack:
                loss.backward()
                self.optimizer.step()

            else:
                loss.backward()
                self.fgm.attack()

                y_pred = self.model(input_ids, attention_mask, token_type_ids)
                adv_loss = self.compute_loss(y_pred, y_true)
                adv_loss.backward()

                self.fgm.restore()
                self.optimizer.step()

            # self.scheduler.step()  # 学习率更新

        self.print(
            f'epoch:{self.current_epoch}, global_step:{self.global_step}, train_step_loss:{loss:.5f}, train_step_acc:{acc:.5f}')

        return loss

    def training_epoch_end(self, outputs):
        acc = self.train_accuracy.compute_epoch_end()
        logger.info(f'epoch:{self.current_epoch}, global_step:{self.global_step}, train_acc:{acc:.5f}')

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, y_true = batch
        y_pred = self.model(input_ids, attention_mask, token_type_ids)
        acc = self.valid_accuracy(y_pred, y_true)
        loss = self.compute_loss(y_pred, y_true)

        self.print(
            f'epoch:{self.current_epoch}, global_step:{self.global_step}, valid_step_loss:{loss:.5f}, valid_step_acc:{acc:.5f}')

    def validation_epoch_end(self, outputs):
        acc = self.valid_accuracy.compute_epoch_end()
        logger.info(f'epoch:{self.current_epoch}, global_step:{self.global_step}, valid_acc:{acc:.5f}')

    def configure_optimizers(self, bert_lr=2e-5, other_lr=5e-5, total_step=10000):
        """设置优化器"""
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if n.startswith('bert')], 'lr': bert_lr},
            {'params': [p for n, p in self.model.named_parameters() if n.startswith('classifier')], 'lr': other_lr}
        ]
        optimizer = AdamW(optimizer_grouped_parameters)
        # return [optimizer]

        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_step), num_training_steps=total_step)
        scheduler = {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}

        return [optimizer], [scheduler]


if __name__ == '__main__':
    pl.seed_everything(42)

    data = pd.read_csv('./data/jingdong.csv')

    off_test = np.zeros(len(data))
    skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    for n, (train_idx, valid_idx) in enumerate(skf.split(data, data['label'])):
        logger.info(f'运行第{n + 1}折......')
        train_data, valid_data = data.iloc[train_idx], data.iloc[valid_idx]

        checkpoint_callback = ModelCheckpoint(save_path=f'output_model_{n + 1}')
        trainer = pl.Trainer(
            default_root_dir=f'pl_model_{n + 1}',
            gpus=None,
            precision=32,
            max_epochs=100,
            val_check_interval=1.0,
            callbacks=[checkpoint_callback],
            logger=False,
            gradient_clip_val=0.0,
            distributed_backend=None,
            num_sanity_val_steps=-1,
            accumulate_grad_batches=1,
            check_val_every_n_epoch=1,
            progress_bar_refresh_rate=0,
        )
        bmc = BertClassifier(train_data=train_data, valid_data=valid_data, model_path='./simbert')
        trainer.fit(bmc)
