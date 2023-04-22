import argparse

import pandas as pd

from tqdm.auto import tqdm

import transformers
import torch
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
import torch.nn.functional as F

# lr_scheduler를 위해 import
# from torch.optim.lr_scheduler import StepLR

# num_worker를 추가했을 때, tokenizer가 parallel되어 뜨는 warning을 해결해주기 위해 추가하였습니다.
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 로깅을 위해 wandb를 import했습니다.
# wandb가 설치되어있지 않다면, 아래 코드를 터미널에 입력해주세요
# pip install wandb -qU
from pytorch_lightning.loggers import WandbLogger

# pearson 산점도 플롯을 위한 모듈을 import합니다
import wandb


class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets=[]):
        self.inputs = inputs
        self.targets = targets

    # 학습 및 추론 과정에서 데이터를 1개씩 꺼내오는 곳
    def __getitem__(self, idx):
        # 정답이 있다면 else문을, 없다면 if문을 수행합니다
        if len(self.targets) == 0:
            return {
                "input_ids": self.inputs[idx]["input_ids"],
                # "token_type_ids": self.inputs[idx]["token_type_ids"],
                "attention_mask": self.inputs[idx]["attention_mask"],
            }
        else:
            return {
                "input_ids": self.inputs[idx]["input_ids"],
                # "token_type_ids": self.inputs[idx]["token_type_ids"],
                "attention_mask": self.inputs[idx]["attention_mask"],
            }, torch.tensor(self.targets[idx])

    # 입력하는 개수만큼 데이터를 사용합니다
    def __len__(self):
        return len(self.inputs)


class Dataloader(pl.LightningDataModule):
    def __init__(
        self,
        model_name,
        batch_size,
        shuffle,
        train_path,
        dev_path,
        test_path,
        predict_path,
        num_workers,
    ):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.predict_path = predict_path

        self.num_workers = num_workers

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name, max_length=200
        )
        self.target_columns = ["label"]
        self.delete_columns = ["id"]
        self.text_columns = ["sentence_1", "sentence_2"]

    def tokenizing(self, dataframe):
        data = []
        for idx, item in tqdm(
            dataframe.iterrows(), desc="tokenizing", total=len(dataframe)
        ):
            text = [item[text_column] for text_column in self.text_columns]

            outputs = self.tokenizer(
                *text, add_special_tokens=True, padding="max_length", truncation=True,
                max_length = 256
            )

            for key in outputs:
                outputs[key] = torch.tensor(outputs[key], dtype=torch.long)
            data.append(outputs)

        return data

    def preprocessing(self, data):
        # 안쓰는 컬럼을 삭제합니다.
        data = data.drop(columns=self.delete_columns)

        # 타겟 데이터가 없으면 빈 배열을 리턴합니다.
        try:
            targets = data[self.target_columns].values.tolist()
        except:
            targets = []
        # 텍스트 데이터를 전처리합니다.
        inputs = self.tokenizing(data)

        return inputs, targets

    def setup(self, stage="fit"):
        if stage == "fit":
            # 학습 데이터와 검증 데이터셋을 호출합니다
            train_data = pd.read_csv(self.train_path)
            val_data = pd.read_csv(self.dev_path)

            # 학습데이터 준비
            train_inputs, train_targets = self.preprocessing(train_data)

            # 검증데이터 준비
            val_inputs, val_targets = self.preprocessing(val_data)

            # train 데이터만 shuffle을 적용해줍니다, 필요하다면 val, test 데이터에도 shuffle을 적용할 수 있습니다
            self.train_dataset = Dataset(train_inputs, train_targets)
            self.val_dataset = Dataset(val_inputs, val_targets)
        else:
            # 평가데이터 준비
            test_data = pd.read_csv(self.test_path)
            test_inputs, test_targets = self.preprocessing(test_data)
            self.test_dataset = Dataset(test_inputs, test_targets)

            predict_data = pd.read_csv(self.predict_path)
            predict_inputs, predict_targets = self.preprocessing(predict_data)
            self.predict_dataset = Dataset(predict_inputs, [])

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=args.shuffle,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )


# 기존의 L1Loss가 아닌, train에 쓸 Custom weighted-Loss를 정의합니다.
def weighted_Loss(logits, ground_truth, loss_func, penalty_zero, threshold=0.05):
    # 각 element에 대해 L1Loss를 계산, weight를 위해 mean을 하지 않습니다.
    basic_loss = loss_func(reduction="none")(logits, ground_truth)

    # GT는 0이 아닌데, logit이 반올림해서 0이 나오면 penalty를 부여합니다.
    penalty_mask_zero = (ground_truth != 0) & (logits <= threshold)
    penalty_loss = basic_loss * penalty_zero * penalty_mask_zero

    if args.penalty_five:
        penalty_mask_five = (ground_truth != 5) & (logits >= 5 - threshold)
        penalty_loss += basic_loss * args.penalty_five * penalty_mask_five

    loss = basic_loss + penalty_loss
    return loss


# swap_sentences를 해주는 함수를 정의합니다
def swap_sentences(input_ids, attention_mask):  # 2 * (batch_size, max_seq_len)
    sep_poss = (input_ids == SEP_ID).nonzero(as_tuple=True)[1]
    sep_poss = [
        (sep_poss[i], sep_poss[i + 1]) for i in range(0, len(sep_poss) - 1, 2)
    ]  # [(sep1, sep2),(sep1, sep2)]

    input_ids_swapped = torch.full_like(input_ids, fill_value=PAD_ID)
    input_ids_swapped[:, 0] = CLS_ID
    # token_type_ids_swapped = torch.zeros_like(token_type_ids)

    for i, sep_pos in enumerate(sep_poss):
        sep_1, sep_2 = sep_pos
        input_ids_swapped[i, 1 : sep_2 - sep_1 + 1] = input_ids[
            i, sep_1 + 1 : sep_2 + 1
        ]
        input_ids_swapped[i, sep_2 - sep_1 + 1 : sep_2 + 1] = input_ids[
            i, 1 : sep_1 + 1
        ]
        # token_type_ids_swapped[i, sep_2 - sep_1 + 1 : sep_2 + 1] = 1

    return input_ids_swapped, attention_mask  # , token_type_ids_swapped


# rdrop을 구현한 함수입니다
def rdrop_L1(logits_1, logits_2, alpha=0.1):
    return alpha*torch.abs(logits_1 - logits_2).mean()


def rdrop_MSE(logits_1, logits_2, alpha=1.0):
    return alpha*torch.nn.MSELoss()(logits_1, logits_2)


class Model(pl.LightningModule):
    def __init__(self, model_name, lr, L1Loss, penalty_zero):
        super().__init__()
        self.save_hyperparameters()
        self.model_name = model_name
        self.lr = lr
        self.L1Loss = L1Loss
        self.penalty_zero = penalty_zero

        # self.automatic_optimization = True
        # 사용할 모델을 호출합니다.
        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_name, num_labels=1
        )

        self.loss_func = torch.nn.L1Loss if self.L1Loss else torch.nn.MSELoss

        self.validation_epoch_logits = []
        self.validation_epoch_y = []

    def forward(self, x):
        logits_1 = self.plm(**x)["logits"]
        logits_2 = (
            self.plm(*swap_sentences(**x))["logits"]
            if args.S_swap
            else self.plm(**x)["logits"]
            if args.R_drop
            else None
        )

        return logits_1, logits_2

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits_1, logits_2 = self(x)
        loss = (
            torch.mean(
                weighted_Loss(logits_1, y.float(), self.loss_func, self.penalty_zero)
            )
            if self.penalty_zero
            else self.loss_func()(logits_1, y.float())
        )
        if args.S_swap:
            loss_logits_2 = (
                torch.mean(
                    weighted_Loss(
                        logits_2, y.float(), self.loss_func, self.penalty_zero
                    )
                    * (y.float() != 0)
                )
                if self.penalty_zero
                else torch.mean(
                    self.loss_func(reduction="none")(logits_2, y.float())
                    * (y.float() != 0)
                )
            )

            loss += loss_logits_2
            loss *= 0.5

        if args.R_drop:
            if args.R_drop == "L1":
                loss += rdrop_L1(logits_1, logits_2)
            elif args.R_drop == "MSE":
                loss += rdrop_MSE(logits_1, logits_2)
            else:
                raise ValueError("Check your R_drop argument")

        self.log("train_loss", loss)
        return loss

    def on_validation_epoch_start(self):
        self.validation_epoch_preds = []
        self.validation_epoch_targets = []

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits, _ = self(x)
        loss = self.loss_func()(logits, y.float())
        self.log("val_loss", loss)

        self.log(
            "val_pearson",
            torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()),
        )
        self.validation_epoch_preds.extend(logits.detach().cpu().numpy())
        self.validation_epoch_targets.extend(y.detach().cpu().numpy())

        return loss

    def on_validation_epoch_end(self):
        logits = self.validation_epoch_preds
        y = self.validation_epoch_targets
        columns=["logits", "y"]
        data = [[float(x), float(y)] for (x, y) in zip(logits, y)]

        self.logger.log_table(
            key="Pearson Analysis", columns=columns, data=data
        )

        # super().on_validation_epoch_end()

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits, _ = self(x)

        self.log(
            "test_pearson",
            torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()),
        )

    def predict_step(self, batch, batch_idx):
        x = batch
        logits, _ = self(x)

        return logits.squeeze()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        # scheduler = StepLR(optimizer, step_size=1, gamma=0.1, verbose = True)
        # return [optimizer], [scheduler]
        return optimizer


if __name__ == "__main__":

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    # 하이퍼 파라미터 등 각종 설정값을 입력받습니다
    # 터미널 실행 예시 : python3 run.py --batch_size=64 ...
    # 실행 시 '--batch_size=64' 같은 인자를 입력하지 않으면 default 값이 기본으로 실행됩니다
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="klue/roberta-small", type=str)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--max_epoch", default=100, type=int)
    parser.add_argument(
        "--shuffle", type=str2bool, nargs="?", const=False, default=True
    )
    parser.add_argument("--learning_rate", default=7e-6, type=float)
    parser.add_argument("--train_path", default="./data/train.csv")
    parser.add_argument("--dev_path", default="./data/dev.csv")
    parser.add_argument("--test_path", default="./data/dev.csv")
    parser.add_argument("--predict_path", default="./data/test.csv")
    parser.add_argument("--patience", default=3, type=int)

    # bottle neck problem을 없애기 위해 dataloader의 num_worker 수를 받는 argument를 추가하였습니다
    parser.add_argument("--num_workers", default=8, type=int)

    # wandb 프로젝트 명과 실험 명을 받을 수 있는 argument를 추가했습니다.
    parser.add_argument("--wandb_project", default="")
    parser.add_argument("--wandb_name", default="")

    # L1Loss을 쓸 지, 0점에 페널티를 얼마나 줄지(입력하지 않으면 페널티 사용 X) 정합니다.
    parser.add_argument("--L1Loss", type=str2bool, nargs="?", const=False, default=True)
    parser.add_argument("--penalty_zero", default=0, type=int)
    parser.add_argument("--penalty_five", default=0, type=int)

    # Sentence Swap을 사용할 것인지, R drop을 사용할 것인지 정합니다.
    parser.add_argument("--S_swap", type=str2bool, nargs="?", const=False, default=True)
    parser.add_argument("--R_drop", default="")

    # Auxiliary task(MLM)를 추가로 추가할지를 정합니다.
    parser.add_argument("--MLM", type=str2bool, nargs="?", const=True, default=False)

    args = parser.parse_args()

    SEP_ID = transformers.AutoTokenizer.from_pretrained(args.model_name).sep_token_id

    PAD_ID = transformers.AutoTokenizer.from_pretrained(args.model_name).pad_token_id

    CLS_ID = transformers.AutoTokenizer.from_pretrained(args.model_name).cls_token_id

    # dataloader와 model을 생성합니다.
    dataloader = Dataloader(
        args.model_name,
        args.batch_size,
        args.shuffle,
        args.train_path,
        args.dev_path,
        args.test_path,
        args.predict_path,
        args.num_workers,
    )

    model = Model(
        args.model_name,
        args.learning_rate,
        args.L1Loss,
        args.penalty_zero,
    )

    # wandb logger를 설정합니다.
    # 만약 터미널에서 --wandb_name에 대한 인자값을 주지 않았다면, default로 wandb에서 제공하는 기본값이 이름으로 들어갑니다.
    # 만약 터미널에서 --wandb_project에 대한 인자값을 주지 않았다면, default로 현재 시각을 project 이름으로 합니다
    # 만약 wandb를 실행하고 싶지 않다면 name과 project 모두 생략
    if args.wandb_name != "" and args.wandb_project != "":
        wandb_logger = WandbLogger(project=args.wandb_project, name=args.wandb_name)
    else:
        wandb_logger = None

    # Early stop callback 함수를 생성합니다.
    # 기준은 objective인 val_pearson으로 하였으며 그렇기에 mode도 max로 하였습니다.
    # verbose도 True로 놓았습니다. 한 번의 validaion epoch이 끝날 때마다 pearson이 얼마나 올랐는지, 안 올랐다면 곧 학습이 종료될 수도 있다는 메세지가 뜹니다.

    early_stop_custom_callback = EarlyStopping(
        "val_pearson", patience=args.patience, verbose=True, mode="max"
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='val_pearson',
        save_top_k=1,
        dirpath = "./",
        filename="-".join(args.model_name.split("/") + args.wandb_name.split()),
        save_weights_only=False,
        verbose=True,
        mode='max')

    # gpu가 없으면 accelerator='cpu', 있으면 accelerator='gpu'
    trainer = pl.Trainer(
        accelerator="gpu",
        max_epochs=args.max_epoch,
        log_every_n_steps=1,
        callbacks=[checkpoint_callback, early_stop_custom_callback],
        logger=wandb_logger,
    )

    trainer.fit(model=model, datamodule=dataloader)
    
    model = Model(args.model_name, args.learning_rate, args.L1Loss, args.penalty_zero)
    filename = "-".join(args.model_name.split("/") + args.wandb_name.split()) + '.ckpt'
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])
    
    trainer.test(model=model, datamodule=dataloader)
    # 학습이 완료된 모델을 저장합니다.
    filename = "-".join(args.model_name.split("/") + args.wandb_name.split()) + ".pt"
    torch.save(model, filename)