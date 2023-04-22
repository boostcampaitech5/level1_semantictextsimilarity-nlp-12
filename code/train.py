import argparse

import pandas as pd

from tqdm.auto import tqdm

import transformers
import torch
import torch.nn as nn
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

# voting을 위한 cls_head를 복사할 deepcopy를 import합니다
from copy import deepcopy


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

        special_tokens_dict = {
            "additional_special_tokens": [
                "[petition]",
                "[nsmc]",
                "[slack]",
                "[sampled]",
                "[rtt]",
            ]
        }
        self.tokenizer.add_special_tokens(
            special_tokens_dict
        )  # Added new tokens to vocabulary.
        self.toks = len(self.tokenizer)

        self.target_columns = ["label"]
        self.delete_columns = ["id"]
        self.text_columns = ["sentence_1", "sentence_2"]
        self.source_columns = ["source"]

    def tokenizing(self, dataframe):
        data = []
        for idx, item in tqdm(
            dataframe.iterrows(), desc="tokenizing", total=len(dataframe)
        ):
            text = [item[text_column] for text_column in self.text_columns]
            s1, s2 = item[self.source_columns].item().split("-")

            text[0] = "[" + s1 + "]" + "[" + s2 + "]" + text[0]
            outputs = self.tokenizer(
                *text,
                add_special_tokens=True,
                padding="max_length",
                truncation=True,
                max_length=args.max_padding
            )

            for key in outputs:
                outputs[key] = torch.tensor(outputs[key], dtype=torch.long)  # tensor화
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


def get_special_token_hidden_states(input_ids, hidden_states):
    batch_size = input_ids.size(0)
    hidden_size = hidden_states.size(-1)

    special_token_positions = torch.zeros_like(input_ids).bool()

    for token_id in ALL_SPECIAL_IDS:
        special_token_positions |= input_ids == token_id

    special_token_hidden_states = []
    for i in range(batch_size):
        batch_special_token_hidden_states = torch.masked_select(
            hidden_states[i], special_token_positions[i].unsqueeze(-1)
        )
        if args.pool_special_voting:
            batch_special_token_hidden_states = batch_special_token_hidden_states.view(
                -1, hidden_size
            )

        special_token_hidden_states.append(batch_special_token_hidden_states)

    return torch.stack(special_token_hidden_states, dim=0)


# 기존의 L1Loss가 아닌, train에 쓸 Custom weighted-Loss를 정의합니다.
def weighted_Loss(logits, ground_truth, loss_func, penalty_zero, threshold=0.05):
    # 각 element에 대해 Loss를 계산, weight를 위해 mean을 하지 않습니다.
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
def rdrop_L1(logits_1, logits_2, alpha):
    return torch.abs(logits_1 - logits_2).mean() * alpha


def rdrop_MSE(logits_1, logits_2, alpha):
    return torch.nn.MSELoss()(logits_1, logits_2) * alpha


class Model(pl.LightningModule):
    def __init__(self, model_name, lr, L1Loss, penalty_zero, R_drop_alpha):
        super().__init__()
        self.save_hyperparameters()
        self.model_name = model_name
        self.lr = lr
        self.L1Loss = L1Loss
        self.penalty_zero = penalty_zero
        self.R_drop_alpha = args.R_drop_alpha

        # self.automatic_optimization = True
        # 사용할 모델을 호출합니다.
        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_name,
            num_labels=1,
            output_hidden_states=True,
        )

        # edited for sources tokens
        self.plm.resize_token_embeddings(dataloader.toks)
        self.loss_func = torch.nn.L1Loss if self.L1Loss else torch.nn.MSELoss

        if args.pool_special_linear:
            self.pool_special_linear_block = nn.Sequential(
                nn.Linear(
                    5 * self.plm.config.hidden_size, 128
                ),  # 5 for 1 [CLS], 2 [SEP], 2 custom special token
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(128, 1),
            )

        elif args.pool_special_voting:
            self.pool_special_voting_token_heads = nn.ModuleList(
                [deepcopy(self.plm.classifier) for _ in range(5)]
            )  # 5 for 1 [CLS], 2 [SEP], 2 custom special token
            self.pool_special_voting_weight = nn.Parameter(
                torch.tensor([9.0, 0.5, 0.5, 0.5, 0.5])
            )

    def forward(self, x):
        output_1 = self.plm(**x, output_hidden_states=True)
        output_2 = (
            self.plm(*swap_sentences(**x))
            if args.S_swap
            else self.plm(**x, output_hidden_states=True)
            if args.R_drop
            else {"logits": None, "hidden_states": [None]}
        )

        return (
            output_1["logits"],
            output_2["logits"],
            output_1["hidden_states"][-1],
            output_2["hidden_states"][-1],
        )

    def training_step(self, batch, batch_idx):
        x, y = batch
        input_ids_1 = x["input_ids"]
        input_ids_2 = swap_sentences(**x)[0]
        cls_logits_1, cls_logits_2, hidden_states_1, hidden_states_2 = self(x)

        if args.pool_special_linear:
            special_hidden_states_1 = get_special_token_hidden_states(
                input_ids_1, hidden_states_1
            )  # (batches, num_special_tokens * hidden_size)
            logits_1 = self.pool_special_linear_block(
                special_hidden_states_1
            ).unsqueeze(1)
            del special_hidden_states_1

            if args.S_swap:
                special_hidden_states_2 = get_special_token_hidden_states(
                    input_ids_2, hidden_states_2
                )
                logits_2 = self.pool_special_linear_block(
                    special_hidden_states_2
                ).unsqueeze(1)
                del special_hidden_states_2

        elif args.pool_special_voting:
            special_hidden_states_1 = get_special_token_hidden_states(
                input_ids_1, hidden_states_1
            )  # (batches, num_special_tokens, hidden_size)

            special_token_logits_1 = [
                classifier(hidden).squeeze(-1)
                for hidden, classifier in zip(
                    special_hidden_states_1.split(1, dim=1),
                    self.pool_special_voting_token_heads,
                )
            ]

            # Stack the logits along dimension 1
            special_token_logits_1 = torch.stack(special_token_logits_1, dim=1)
            logits_1 = torch.sum(
                self.pool_special_voting_weight * special_token_logits_1, dim=1
            ).unsqueeze(1)

            del special_hidden_states_1
            del special_token_logits_1

            if args.S_swap:
                special_hidden_states_2 = get_special_token_hidden_states(
                    input_ids_2, hidden_states_2
                )  # (batches, num_special_tokens, hidden_size)

                special_token_logits_2 = [
                    classifier(hidden).squeeze(-1)
                    for hidden, classifier in zip(
                        special_hidden_states_2.split(1, dim=1),
                        self.pool_special_voting_token_heads,
                    )
                ]

                # Stack the logits along dimension 1
                special_token_logits_2 = torch.stack(special_token_logits_2, dim=1)
                logits_2 = torch.sum(
                    self.pool_special_voting_weight * special_token_logits_2, dim=1
                ).unsqueeze(1)

                del special_hidden_states_2
                del special_token_logits_2

        else:
            logits_1 = cls_logits_1
            logits_2 = cls_logits_2

        del cls_logits_1
        del cls_logits_2
        del hidden_states_1
        del hidden_states_2

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
                loss += rdrop_L1(logits_1, logits_2, self.R_drop_alpha)
            elif args.R_drop == "MSE":
                loss += rdrop_MSE(logits_1, logits_2, self.R_drop_alpha)
            else:
                raise ValueError("Check your R_drop argument")

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        input_ids = x["input_ids"]
        cls_logits, dummy_logits, hidden_states, dummy_hidden_states = self(x)

        if args.pool_special_linear:
            special_hidden_states = get_special_token_hidden_states(
                input_ids, hidden_states
            )  # (batches, num_special_tokens, hidden_size)
            logits = self.pool_special_linear_block(special_hidden_states).unsqueeze(1)
            del special_hidden_states

        elif args.pool_special_voting:
            special_hidden_states = get_special_token_hidden_states(
                input_ids, hidden_states
            )  # (batches, num_special_tokens, hidden_size)

            # calculate each special token's logit for all batches
            special_token_logits = [
                classifier(hidden).squeeze(-1)
                for hidden, classifier in zip(
                    special_hidden_states.split(1, dim=1),
                    self.pool_special_voting_token_heads,
                )
            ]

            # Stack the logits along dimension 1
            special_token_logits = torch.stack(special_token_logits, dim=1)

            logits = torch.sum(
                self.pool_special_voting_weight * special_token_logits, dim=1
            ).unsqueeze(1)

            del special_hidden_states
            del special_token_logits

        else:
            logits = cls_logits

        del cls_logits
        del dummy_logits
        del hidden_states
        del dummy_hidden_states

        loss = self.loss_func()(logits, y.float())
        self.log("val_loss", loss)

        self.log(
            "val_pearson",
            torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()),
        )

        return loss

    def on_test_epoch_start(self):
        self.test_epoch_preds = []
        self.test_epoch_targets = []

    def test_step(self, batch, batch_idx):
        x, y = batch
        input_ids = x["input_ids"]
        cls_logits, dummy_logits, hidden_states, dummy_hidden_states = self(x)

        if args.pool_special_linear:
            special_hidden_states = get_special_token_hidden_states(
                input_ids, hidden_states
            )  # (batches, num_special_tokens, hidden_size)
            logits = self.pool_special_linear_block(special_hidden_states).unsqueeze(1)
            del special_hidden_states

        elif args.pool_special_voting:
            special_hidden_states = get_special_token_hidden_states(
                input_ids, hidden_states
            )  # (batches, num_special_tokens, hidden_size)

            # calculate each special token's logit for all batches
            special_token_logits = [
                classifier(hidden).squeeze(-1)
                for hidden, classifier in zip(
                    special_hidden_states.split(1, dim=1),
                    self.pool_special_voting_token_heads,
                )
            ]

            # Stack the logits along dimension 1
            special_token_logits = torch.stack(special_token_logits, dim=1)

            logits = torch.sum(
                self.pool_special_voting_weight * special_token_logits, dim=1
            ).unsqueeze(1)

            del special_hidden_states
            del special_token_logits

        else:
            logits = cls_logits

        del cls_logits
        del dummy_logits
        del hidden_states
        del dummy_hidden_states

        self.log(
            "test_pearson",
            torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()),
        )

        self.test_epoch_preds.extend(logits.cpu().numpy())
        self.test_epoch_targets.extend(y.cpu().numpy())

    def predict_step(self, batch, batch_idx):
        x = batch
        logits, _ = self(x)

        return logits.squeeze()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        # scheduler = StepLR(optimizer, step_size=1, gamma=0.1, verbose = True)
        return optimizer  # , [scheduler]


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
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--train_path", default="./data/train.csv")
    parser.add_argument("--dev_path", default="./data/dev.csv")
    parser.add_argument("--test_path", default="./data/dev.csv")
    parser.add_argument("--predict_path", default="./data/test.csv")
    parser.add_argument("--patience", default=3, type=int)
    # tokenizer 했을 때 길이를 줄여주는! 학습 시간을 빨리 해주기 위한 padding 수를 줄이는 hyperparameter
    parser.add_argument("--max_padding", default=256)

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
    parser.add_argument("--S_swap", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--R_drop", default="")
    parser.add_argument("--R_drop_alpha", default=0.1, type=float)
    # Auxiliary task(MLM)를 추가로 추가할지를 정합니다.
    parser.add_argument("--MLM", type=str2bool, nargs="?", const=True, default=False)

    # Special token을 concat하여 linear layer를 통과하여 pooling합니다.
    parser.add_argument(
        "--pool_special_linear", type=str2bool, nargs="?", const=True, default=False
    )

    # Pretrained된 cls head를 각 special token에도 fine-tuning시켜 voting합니다.
    parser.add_argument(
        "--pool_special_voting", type=str2bool, nargs="?", const=True, default=False
    )

    args = parser.parse_args()

    print(
        "\n".join(
            [
                "argument " + k + " is " + str(v) + ". it's type is " + str(type(v))
                for k, v in args.__dict__.items()
            ]
        )
    )

    print(
        "\n".join(
            [
                "argument " + k + " is " + str(v) + ". it's type is " + str(type(v))
                for k, v in args.__dict__.items()
            ]
        )
    )

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

    SEP_ID = transformers.AutoTokenizer.from_pretrained(args.model_name).sep_token_id

    PAD_ID = transformers.AutoTokenizer.from_pretrained(args.model_name).pad_token_id

    CLS_ID = transformers.AutoTokenizer.from_pretrained(args.model_name).cls_token_id

    ALL_SPECIAL_IDS = set(
        [SEP_ID, CLS_ID] + dataloader.tokenizer.additional_special_tokens_ids
    )

    model = Model(
        args.model_name,
        args.learning_rate,
        args.L1Loss,
        args.penalty_zero,
        args.R_drop_alpha,
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
        monitor="val_pearson",
        save_top_k=1,
        dirpath="./",
        filename="-".join(args.model_name.split("/") + args.wandb_name.split()),
        save_weights_only=False,
        verbose=True,
        mode="max",
    )

    class LogScatterPlotCallback(pl.Callback):
        def on_test_epoch_end(self, trainer, pl_module):
            # Extract predictions and targets from the pl_module
            preds = pl_module.test_epoch_preds
            targets = pl_module.test_epoch_targets

            # Create the table with predictions and targets
            data = [[float(x), float(y)] for (x, y) in zip(preds, targets)]
            table = wandb.Table(data=data, columns=["predictions", "targets"])

            # Log the table
            pl_module.logger.experiment.log(
                {
                    "scatter_plot": wandb.plot.scatter(
                        table, "predictions", "targets", title="Predictions vs Targets"
                    )
                },
                step=trainer.current_epoch,
            )

            super().on_test_epoch_end(trainer, pl_module)

    # gpu가 없으면 accelerator='cpu', 있으면 accelerator='gpu'
    trainer = pl.Trainer(
        accelerator="gpu",
        max_epochs=args.max_epoch,
        log_every_n_steps=1,
        callbacks=[
            checkpoint_callback,
            early_stop_custom_callback,
            LogScatterPlotCallback(),
        ],
        logger=wandb_logger,
    )

    # Train part
    trainer.fit(model=model, datamodule=dataloader)

    model = Model(
        args.model_name,
        args.learning_rate,
        args.L1Loss,
        args.penalty_zero,
        args.R_drop_alpha,
    )
    filename = "-".join(args.model_name.split("/") + args.wandb_name.split()) + ".ckpt"
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint["state_dict"])

    trainer.test(model=model, datamodule=dataloader)

    # 학습이 완료된 모델을 저장합니다.
    filename = "-".join(args.model_name.split("/") + args.wandb_name.split()) + ".pt"
    torch.save(model, filename)
