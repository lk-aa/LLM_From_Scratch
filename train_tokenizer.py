import random
from tqdm import tqdm
from transformers import AutoTokenizer
import json
from datasets import load_dataset
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)
import os

random.seed(42)

def train_tokenizer():
    """
    训练一个自定义的基于BPE的tokenizer，并保存训练好的tokenizer及其配置文件。

    该函数执行以下步骤：
    1. 从指定的JSONL文件中读取文本数据。
    2. 初始化一个基于字节对编码（BPE）的tokenizer。
    3. 定义特殊token，如未知词、句子开始和结束标记。
    4. 设置训练器并添加特殊token。
    5. 使用读取的文本数据训练tokenizer。
    6. 设置解码器。
    7. 检查特殊token的索引。
    8. 保存训练好的tokenizer及其配置文件。
    """
    # 读取JSONL文件并提取文本数据
    def read_texts_from_jsonl(file_path):
        """
        从指定的JSONL文件中读取每一行，解析为JSON格式，并提取其中的'text'字段作为训练数据。

        :param file_path: JSONL文件的路径。
        :yield: 从JSONL文件中提取的文本数据。
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                yield data['text']

    data_path = './dataset/tokenizer_train.jsonl'

    # 初始化tokenizer
    # 创建了一个基于BPE的tokenizer，并设置了预分词器为字节级别的预分词器，不添加前缀空格
    tokenizer = Tokenizer(models.BPE())   
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # 定义特殊token
    # 定义了三个特殊token：<unk> 表示未知词，<s> 表示句子开始，</s> 表示句子结束
    special_tokens = ["<unk>", "<s>", "</s>"]

    # 设置训练器并添加特殊token
    # 创建了一个BPE训练器，设置词汇表大小为6400，并确保特殊token被包含在词汇表中
    trainer = trainers.BpeTrainer(
        vocab_size=6400,
        special_tokens=special_tokens,  # 确保这三个token被包含
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )

    # 读取文本数据
    # 从指定的JSONL文件中读取文本数据
    texts = read_texts_from_jsonl(data_path)

    # 训练tokenizer
    # 使用读取的文本数据和训练器来训练tokenizer
    tokenizer.train_from_iterator(texts, trainer=trainer)

    # 设置解码器
    # 设置解码器为字节级别的解码器
    tokenizer.decoder = decoders.ByteLevel()

    # 检查特殊token的索引
    # 确保特殊token的索引正确
    assert tokenizer.token_to_id("<unk>") == 0
    assert tokenizer.token_to_id("<s>") == 1
    assert tokenizer.token_to_id("</s>") == 2

    # 保存tokenizer
    # 将训练好的tokenizer保存到指定目录
    tokenizer_dir = "./model/mateconv_tokenizer"
    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))
    tokenizer.model.save("./model/mateconv_tokenizer")

    # 手动创建配置文件
    # 手动创建一个配置文件，包含tokenizer的各种设置
    config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "add_prefix_space": True,
        "added_tokens_decoder": {
            "0": {
                "content": "<unk>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            "1": {
                "content": "<s>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            "2": {
                "content": "</s>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            }
        },
        "additional_special_tokens": [],
        "bos_token": "<s>",
        "clean_up_tokenization_spaces": False,
        "eos_token": "</s>",
        "legacy": True,
        "model_max_length": 1000000000000000019884624838656,
        "pad_token": None,
        "sp_model_kwargs": {},
        "spaces_between_special_tokens": False,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "unk_token": "<unk>",
        "use_default_system_prompt": False,
        "chat_template": "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{% endif %}{% if system_message is defined %}{{ system_message }}{% endif %}{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<s>user\\n' + content + '</s>\\n<s>assistant\\n' }}{% elif message['role'] == 'assistant' %}{{ content + '</s>' + '\\n' }}{% endif %}{% endfor %}"
    }

    # 保存配置文件
    # 将配置文件保存到指定目录
    with open(os.path.join(tokenizer_dir, "tokenizer_config.json"), "w", encoding="utf-8") as config_file:
        json.dump(config, config_file, ensure_ascii=False, indent=4)

    # 打印训练完成的信息
    print("Tokenizer training completed and saved.")


def eval_tokenizer():
    """
    评估预训练的tokenizer，包括加载tokenizer、应用聊天模板、获取词汇表长度、对文本进行编码和解码，并检查解码后的文本是否与原始文本一致。
    """
    # 导入必要的库
    from transformers import AutoTokenizer

    # 加载预训练的tokenizer
    tokenizer = AutoTokenizer.from_pretrained("./model/mateconv_tokenizer")

    # 定义聊天消息
    # 定义了一个包含系统消息、用户消息和助手消息的列表
    messages = [
        {"role": "system", "content": "你是一个优秀的聊天机器人，总是给我正确的回应！"},
        {"role": "user", "content": '你来自哪里？'},
        {"role": "assistant", "content": '我来自地球'}
    ]

    # 应用聊天模板
    # 使用分词器的 apply_chat_template 方法将聊天消息转换为一个字符串，并打印出来
    new_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False
    )
    print(new_prompt)

    # 获取实际词汇表长度（包括特殊符号）
    actual_vocab_size = len(tokenizer)
    print('tokenizer实际词表长度：', actual_vocab_size)

    # 对新提示进行编码
    # 使用分词器对新提示进行编码，并打印出编码后的输入ID的长度
    model_inputs = tokenizer(new_prompt)
    print('encoder长度：', len(model_inputs['input_ids']))

    # 对编码后的输入进行解码
    # 使用分词器对编码后的输入进行解码，并检查解码后的文本是否与原始文本一致
    input_ids = model_inputs['input_ids']
    response = tokenizer.decode(input_ids)
    print('decoder和原始文本是否一致：', response == new_prompt)

    # 测试tokenizer
    prompt = "第16届亚冬会在哈尔滨举行。"
    model_inputs = tokenizer(prompt)
    response = tokenizer.decode(model_inputs['input_ids'])
    print(model_inputs)
    print(response)


def main():
    # train_tokenizer()
    eval_tokenizer()


if __name__ == '__main__':
    main()
