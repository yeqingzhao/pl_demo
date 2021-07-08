import logging

import pandas as pd
from numpy import mean
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning import Callback
from pytorch_lightning import LightningModule
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers import BasicTokenizer
from transformers import BertModel
from transformers import BertPreTrainedModel
from transformers import BertTokenizer
from transformers.models.bert.modeling_bert import BertOnlyMLMHead

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%Y/%m/%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def custom_bert(bert_base_model_path: str = './custom_model',
                word_embeddings_weight_path: str = './custom_model/word_embeddings.pt',
                custom_bert_model_path: str = './custom_model',
                vocab_size: int = 100,
                hidden_size: int = 768):
    """
    自定义bert结构，主要修改word embedding的大小
    :param bert_base_model_path: 原始模型的路径
    :param custom_bert_model_path: 新模型保存路径
    :param word_embeddings_weight_path: 词嵌入的路径，可以是w2v训练好的
    :param vocab_size: 新模型词库大小
    :param hidden_size: 新模型词嵌入向量大小，一般无需修改
    :return: None
    """
    model = CustomModel.from_pretrained(bert_base_model_path)

    model.resize_token_embeddings(vocab_size)  # 修改word embedding的vocab_size大小
    word_embeddings_weight = torch.load(word_embeddings_weight_path)
    word_embeddings = nn.Embedding(vocab_size, hidden_size, _weight=word_embeddings_weight)
    model.set_input_embeddings(word_embeddings)

    model.save_pretrained(custom_bert_model_path)


class CustomDataset(Dataset):
    """自定义Dataset，sentence为文本"""

    def __init__(self, dataframe):
        self._data = dataframe
        self._sentence = self._data.sentence

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        return self._sentence.iloc[index]


class CustomTokenizer(BertTokenizer):
    """自定义分词器，英语按空格分开，长词不切分，中文按字分开"""

    def __init__(self, do_lower_case=False, **kwargs):
        super().__init__(**kwargs)
        self.base_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    def tokenize(self, text, **kwargs):
        tokenized_text = self.base_tokenizer.tokenize(text)

        return tokenized_text


class CustomModel(BertPreTrainedModel):
    """自定义模型"""

    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config)

    def get_output_embeddings(self):
        # decoder与word_embedding权值共享，https://zhuanlan.zhihu.com/p/132554155
        return self.cls.predictions.decoder

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):
        """
        预测mask token的概率值
        :param input_ids:
        :param attention_mask:
        :param token_type_ids:
        :return:
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pred_logits = self.cls(outputs.last_hidden_state)

        return pred_logits


class ModelCheckpoint(Callback):
    def __init__(self, save_path='output', mode='min', patience=100):
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
        if self.mode == 'max' and pl_module.valid_loss[-1] >= self.best_value:
            self.check_patience = 0
            self.best_value = pl_module.valid_loss[-1]
            logger.info(f'save best model with metric: {self.best_value:.5f}')
            pl_module.model.save_pretrained(self.path)  # 保存模型

        if self.mode == 'max' and pl_module.valid_loss[-1] < self.best_value:
            self.check_patience += 1

        if self.mode == 'min' and pl_module.valid_loss[-1] <= self.best_value:
            self.check_patience = 0
            self.best_value = pl_module.valid_loss[-1]
            logger.info(f'save best model with metric: {self.best_value:.5f}')
            pl_module.model.save_pretrained(self.path)  # 保存模型

        if self.mode == 'min' and pl_module.valid_loss[-1] > self.best_value:
            self.check_patience += 1

        if self.check_patience >= self.patience:
            trainer.should_stop = True  # 停止训练


class BertClassifier(LightningModule):
    """采用pytorch-lightning训练"""

    def __init__(self, train_data: pd.DataFrame,  model_path: str = 'output'):
        super(BertClassifier, self).__init__()

        self.model = CustomModel.from_pretrained(model_path)  # 自定义的模型
        self.tokenizer = BertTokenizer.from_pretrained(model_path)  # 加载分词器

        self.train_dataset = CustomDataset(dataframe=train_data)  # 加载dataset

        self.max_length = 32  # 句子最大长度
        self.train_batch_size = 32

        self.automatic_optimization = False  # 最好关闭，训练速度变快
        if not self.automatic_optimization:
            self.optimizer = self.configure_optimizers()[0]
            # self.optimizers, self.schedulers = self.configure_optimizers()
            # self.optimizer = self.optimizers[0]  # 初始化优化器
            # self.scheduler = self.schedulers[0]['scheduler']  # 初始化学习率策略

        self.scaler = torch.cuda.amp.GradScaler()  # 半精度训练

        self.train_loss = []
        self.valid_loss = []

    def mask_tokens(self, inputs: torch.Tensor, special_tokens_mask=None, mlm_probability=0.15):
        """
        :param inputs:
        :param tokenizer:
        :param special_tokens_mask:
        :param mlm_probability:
        :return:
        """
        labels = inputs.clone()
        probability_matrix = torch.full(labels.shape, mlm_probability)  # 生成概率矩阵
        if special_tokens_mask:
            special_tokens_mask = special_tokens_mask.bool()
        else:
            special_tokens_mask = [self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in
                                   labels.tolist()]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)  # 特殊token(cls,sep,pad,unk)不进行mask

        masked_indices = torch.bernoulli(probability_matrix).bool()  # 随机抽取
        labels[~masked_indices] = -100  # 计算loss时，忽略-100

        # 80%的替换成[MASK]
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10%的随机替换成别的token
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(self.tokenizer.vocab_size, labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        return inputs, labels

    def train_collate_batch(self, batch):
        """
        处理训练集batch，主要是文本转成相应的tokens
        :param batch:
        :return:
        """
        outputs = self.tokenizer(batch, truncation=True, padding=True, max_length=self.max_length, return_tensors='pt')
        outputs['mlm_input_ids'], outputs['mlm_labels'] = self.mask_tokens(outputs['input_ids'])

        return outputs['input_ids'], outputs['attention_mask'], outputs['token_type_ids'], outputs['mlm_input_ids'], outputs['mlm_labels']

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=self.train_collate_batch,
            num_workers=1,
        )

    def val_dataloader(self):
        return 'N'

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, mlm_input_ids, mlm_labels = batch
        pred_logits = self.model(mlm_input_ids, attention_mask, token_type_ids)
        loss = F.cross_entropy(pred_logits.view(-1, self.tokenizer.vocab_size), mlm_labels.view(-1))
        self.train_loss.append(loss.item())
        if not self.automatic_optimization:
            self.optimizer.zero_grad()  # 梯度置零

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            # self.scheduler.step()  # 学习率更新

        self.print(f'epoch:{self.current_epoch}, global_step:{self.global_step}, train_step_loss:{loss:.5f}')

        return loss

    def training_epoch_end(self, outputs):
        loss = 0.0
        for output in outputs:
            loss += output.items()

        logger.info(f'epoch:{self.current_epoch}, global_step:{self.global_step}, train_loss:{loss:.5f}')

    def validation_step(self, batch, batch_idx):
        return None

    def validation_epoch_end(self, outputs):
        self.valid_loss.append(mean(self.train_loss))
        self.train_loss = []

    def configure_optimizers(self, bert_lr=2e-5, other_lr=2e-5, total_step=10000):
        """设置优化器"""
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if n.startswith('bert')], 'lr': bert_lr},
            {'params': [p for n, p in self.model.named_parameters() if n.startswith('cls')], 'lr': other_lr}
        ]
        optimizer = AdamW(optimizer_grouped_parameters)
        return [optimizer]

        # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_step),
        #                                             num_training_steps=total_step)  # 参数需要设置
        # scheduler = {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}
        #
        # return [optimizer], [scheduler]


if __name__ == '__main__':
    # custom_bert()
    pl.seed_everything(42)

    data = pd.read_csv('./data/jingdong.csv')[['sentence']]

    checkpoint_callback = ModelCheckpoint(save_path=f'output_model')
    trainer = pl.Trainer(
        default_root_dir=f'pl_model',
        gpus=None,
        precision=32,
        max_epochs=100,
        val_check_interval=2,
        callbacks=[checkpoint_callback],
        logger=False,
        gradient_clip_val=0.0,
        distributed_backend=None,
        num_sanity_val_steps=0,
        accumulate_grad_batches=1,
        check_val_every_n_epoch=1,
        progress_bar_refresh_rate=0,
    )
    bmc = BertClassifier(train_data=data, model_path='./simbert')
    trainer.fit(bmc)

# word_embeddings = torch.full((100, 768), 0.15)
# torch.save(word_embeddings, "./custom_model/word_embeddings.pt")

# config = BertConfig.from_pretrained('./custom_model', tie_word_embeddings=False)
# model = CustomModel(config)
# model = CustomModel.from_pretrained('./bert-base-chinese')
# # print(id(model.cls.predictions.decoder.weight) == id(model.bert.base_model.embeddings.word_embeddings.weight))
# tokenizer = CustomTokenizer.from_pretrained('./bert-base-chinese')
# output = tokenizer(['因此，在直接使用Google 的BERT预训练模型时，输入最多512个词（还要除掉[CLS]和[SEP]），最多两个句子合成一句。',
#                     '这之外的词和句子会没有对应的embedding。',
#                     '当然，如果有足够的硬件资源自己重新训练BERT，可以更改 BERT config。',
#                     '设置更大max_position_embeddings 和 type_vocab_size值去满足自己的需求。',
#                     'A 45464687 yeqingzhao'], truncation=True, padding=True, max_length=512, return_tensors='pt')
# input_ids, attention_mask, token_type_ids = output['input_ids'], output['attention_mask'], output['token_type_ids']
# inputs, labels = mask_tokens(input_ids, tokenizer)
# logits = model(input_ids=inputs, attention_mask=attention_mask, token_type_ids=token_type_ids)
# print('Done')
