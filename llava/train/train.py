# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import argparse
import copy
import json
import os
import logging
from typing import Dict, Literal, Sequence, Tuple

import mindspore as ms
import numpy as np
from mindnlp import transformers
from mindnlp.engine import Trainer, TrainingArguments
from mindspore.dataset import GeneratorDataset
from PIL import Image

from llava import conversation as conversation_lib
from llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IGNORE_INDEX,
)
from llava.mm_utils import tokenizer_image_token
from llava.model import LlavaLlamaForCausalLM
from llava.utils import str2bool


def parse_args():
    parser = argparse.ArgumentParser(
        description="Training script for Llava-1.5 (Mindspore)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # model
    parser.add_argument(
        "--version", default="v1", choices=["v1"], help="model version."
    )
    parser.add_argument(
        "--compute_dtype",
        default="bf16",
        choices=["fp32", "fp16", "bf16"],
        help="model computation data type.",
    )
    parser.add_argument(
        "--vision_tower",
        default="openai/clip-vit-large-patch14-336",
        help="vision tower name or path.",
    )
    parser.add_argument(
        "--model_name_or_path",
        default="liuhaotian/llava-v1.5-13b",
        help="model name or path.",
    )
    parser.add_argument("--cache_dir", help="cache directory storing the model file")
    parser.add_argument(
        "--freeze_backbone",
        default=False,
        type=str2bool,
        help="Freeze the model backbone during training.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        default=False,
        type=str2bool,
        help="Enable gradient checkpoint (recompute) during training.",
    )
    parser.add_argument(
        "--tune_mm_mlp_adapter",
        default=False,
        type=str2bool,
        help="Whether to train the mlp adapter only.",
    )
    parser.add_argument(
        "--freeze_mm_mlp_adapter",
        default=False,
        type=str2bool,
        help="Whether to freeze the mlp adapter.",
    )
    parser.add_argument(
        "--mm_projector_type",
        default="mlp2x_gelu",
        choices=["mlp2x_gelu"],
        help="mlp adapter structure.",
    )
    parser.add_argument(
        "--mm_vision_select_layer",
        default=-2,
        type=int,
        help="The layer for selecting feature.",
    )
    parser.add_argument(
        "--mm_vision_select_feature",
        default="patch",
        choices=["patch"],
        help="The method for selecting feature.",
    )
    parser.add_argument(
        "--mm_patch_merge_type",
        default="flat",
        choices=["flat"],
        help="Patch merging method.",
    )
    parser.add_argument(
        "--pretrain_mm_mlp_adapter", help="The path of the mlp adapter checkpoint."
    )

    # data
    parser.add_argument(
        "--data_path", required=True, help="Path to the data json file."
    )
    parser.add_argument(
        "--image_folder", required=True, help="Dictory storing the image."
    )
    parser.add_argument(
        "--mm_use_im_start_end",
        default=False,
        type=str2bool,
        help="Whether to use image start end token.",
    )
    parser.add_argument(
        "--mm_use_im_patch_token",
        default=False,
        type=str2bool,
        help="Whether to use image patch token.",
    )
    parser.add_argument(
        "--image_aspect_ratio",
        default="pad",
        choices=["pad"],
        help="Image process method.",
    )
    parser.add_argument(
        "--model_max_length",
        default=2048,
        type=int,
        help="Max. lengh for the tokenizer.",
    )
    parser.add_argument(
        "--group_by_modality_length",
        default=False,
        type=str2bool,
        help="Group the data by length.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        default=4,
        type=int,
        help="Number of workers in data loader",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        default=16,
        type=int,
        help="Batch size per device,",
    )

    # training
    parser.add_argument(
        "--num_train_epochs", default=1, type=int, help="Number of the training epoch."
    )
    parser.add_argument(
        "--output_dir", default="./output", help="Path of the output directory."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        default=1,
        type=int,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--save_total_limit",
        default=1,
        type=int,
        help="Total number of checkpoint for saving",
    )
    parser.add_argument(
        "--learning_rate", default=2e-5, type=float, help="Learning rate."
    )
    parser.add_argument("--weight_decay", default=0, type=float, help="Weight decay.")
    parser.add_argument(
        "--warmup_ratio", default=0.03, type=float, help="Warm-Up ratio."
    )
    parser.add_argument(
        "--lr_scheduler_type", default="cosine", help="LR scheduler type."
    )
    parser.add_argument("--logging_steps", default=1, type=int, help="Logging steps")
    parser.add_argument("--save_steps", default=50000, type=int, help="Saving frquency")
    parser.add_argument(
        "--save_strategy", default="steps", choices=["steps"], help="Saving strategy."
    )
    parser.add_argument(
        "--optim", default="adamw", choices=["adamw"], help="Optimizer name."
    )
    parser.add_argument(
        "--remove_unused_columns",
        default=False,
        type=str2bool,
        help="Whether to remove unused columns.",
    )

    args = parser.parse_args()
    return args


def _tokenize_fn(
    strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer
) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="np",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx + 2 : cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = "unknown"
        sentence["value"] = (
            BEGIN_SIGNAL + from_str + ": " + sentence["value"] + END_SIGNAL
        )
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(sources: Sequence[str], mm_use_im_start_end: bool = False):
    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence["value"]:
                sentence["value"] = (
                    sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
                )
                sentence["value"] = DEFAULT_IMAGE_TOKEN + "\n" + sentence["value"]
                sentence["value"] = sentence["value"].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence["value"] = sentence["value"].replace(
                        DEFAULT_IMAGE_TOKEN,
                        "<Image>" + DEFAULT_IMAGE_TOKEN + "</Image>",
                    )
            replace_token = DEFAULT_IMAGE_TOKEN
            if mm_use_im_start_end:
                replace_token = (
                    DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                )
            sentence["value"] = sentence["value"].replace(
                DEFAULT_IMAGE_TOKEN, replace_token
            )

    return sources


def preprocess_llama_2(
    sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = np.stack(
            [
                tokenizer_image_token(prompt, tokenizer, return_tensors="np")
                for prompt in conversations
            ],
            axis=0,
        )
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="np",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_v1(
    sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = np.stack(
            [
                tokenizer_image_token(prompt, tokenizer, return_tensors="np")
                for prompt in conversations
            ],
            axis=0,
        )
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="np",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.copy()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(np.not_equal(target, tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy:
                round_len -= 1
                instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]["value"]
        source[0]["value"] = DEFAULT_IMAGE_TOKEN
        conversation = (
            source[0]["value"]
            + source[1]["value"]
            + conversation_lib.default_conversation.sep
        )
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [
        tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        for prompt in conversations
    ]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]["value"], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if (
        conversation_lib.default_conversation.sep_style
        == conversation_lib.SeparatorStyle.PLAIN
    ):
        return preprocess_plain(sources, tokenizer)
    elif (
        conversation_lib.default_conversation.sep_style
        == conversation_lib.SeparatorStyle.LLAMA_2
    ):
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)

    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [
            tokenizer_image_token(prompt, tokenizer, return_tensors="np")
            for prompt in conversations
        ]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


class LazySupervisedDataset:
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str,
        image_folder: str,
        tokenizer: transformers.PreTrainedTokenizer,
        image_processor: transformers.CLIPImageProcessor,
        image_aspect_ratio: Literal["pad"] = "pad",
        mm_use_im_start_end: bool = False,
    ):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.image_folder = image_folder
        self.image_processor = image_processor
        self.image_aspect_ratio = image_aspect_ratio
        self.mm_use_im_start_end = mm_use_im_start_end

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(
                sum(len(conv["value"].split()) for conv in sample["conversations"])
                + img_tokens
            )
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(
                len(conv["value"].split()) for conv in sample["conversations"]
            )
            cur_len = cur_len if "image" in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        sources = self.list_data_dict[i]
        if isinstance(i, (int, np.int64)):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if "image" in sources[0]:
            image_file = self.list_data_dict[i]["image"]
            image = Image.open(os.path.join(self.image_folder, image_file)).convert(
                "RGB"
            )
            if self.image_aspect_ratio == "pad":
                image = expand2square(
                    image, tuple(int(x * 255) for x in self.image_processor.image_mean)
                )
                image = self.image_processor.preprocess(image, return_tensors="np")[
                    "pixel_values"
                ][0]
            else:
                image = self.image_processor.preprocess(image, return_tensors="np")[
                    "pixel_values"
                ][0]
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                mm_use_im_start_end=self.mm_use_im_start_end,
            )
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess(
            sources, self.tokenizer, has_image=("image" in self.list_data_dict[i])
        )
        if isinstance(i, int):
            data_dict = dict(
                input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0]
            )

        # image exist in the data
        if "image" in self.list_data_dict[i]:
            data_dict["image"] = image
        else:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.image_processor.crop_size
            data_dict["image"] = np.zeros(3, crop_size["height"], crop_size["width"])

        assert len(data_dict["input_ids"]) == 1
        assert len(data_dict["labels"]) == 1
        input_ids = data_dict["input_ids"][0]
        attention_mask = np.ones_like(input_ids, dtype=np.bool_)
        labels = data_dict["labels"][0]
        return (
            input_ids,
            attention_mask,
            labels,
            data_dict["image"],
        )


def train():
    args = parse_args()

    _dtype = {"fp32": ms.float32, "fp16": ms.float16, "bf16": ms.bfloat16}

    # load model
    model = LlavaLlamaForCausalLM.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        ms_dtype=_dtype[args.compute_dtype],
    )
    model.config.use_cache = False
    if args.freeze_backbone:
        for x in model.model.get_parameters():
            x.requires_grad = False  # freeze backbone except model head

    if args.gradient_checkpointing:
        raise NotImplementedError("Does not support gradient checkpointing currently.")

    # load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    conversation_lib.default_conversation = conversation_lib.conv_templates[
        args.version
    ]

    # load vision tower
    model.get_model().initialize_vision_modules(args)
    vision_tower = model.get_vision_tower()
    vision_tower.to(dtype=_dtype[args.compute_dtype])

    # training on mlp adapter or not
    if args.tune_mm_mlp_adapter:
        for x in model.model.get_parameters():
            x.requires_grad = False
        for p in model.get_model().mm_projector.get_parameters():
            p.requires_grad = True

    if args.freeze_mm_mlp_adapter:
        for p in model.get_model().mm_projector.get_parameters():
            p.requires_grad = False

    model.initialize_vision_tokenizer(args, tokenizer=tokenizer)

    # dataloader
    if args.group_by_modality_length:
        raise NotImplementedError("`group_by_modality_length` is not supported yet.")
    dataset = LazySupervisedDataset(
        data_path=args.data_path,
        image_folder=args.image_folder,
        tokenizer=tokenizer,
        image_processor=vision_tower.image_processor,
        image_aspect_ratio=args.image_aspect_ratio,
        mm_use_im_start_end=args.mm_use_im_start_end,
    )
    dataset = GeneratorDataset(
        dataset,
        column_names=["input_ids", "attention_mask", "labels", "images"],
        column_types=[ms.int64, ms.bool_, ms.int64, ms.float32],
        num_parallel_workers=args.dataloader_num_workers,
        shuffle=True,
    )
    dataset = dataset.padded_batch(
        batch_size=args.per_device_train_batch_size,
        drop_remainder=True,
        pad_info={
            "input_ids": ([tokenizer.model_max_length], tokenizer.pad_token_id),
            "attention_mask": ([tokenizer.model_max_length], 0),
            "labels": ([tokenizer.model_max_length], IGNORE_INDEX),
        },
    )

    # prepare the training args, refer to the MindNLP tutorial
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_total_limit=args.save_total_limit,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_strategy=args.save_strategy,
        optim=args.optim,
        remove_unused_columns=args.remove_unused_columns,
    )

    # create the trainer, refer to the MindNLP tutorial
    trainer = Trainer(model=model, train_dataset=dataset, args=training_args)
    trainer.train()


if __name__ == "__main__":
    train()
