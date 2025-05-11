from typing import Iterable, Dict, Iterator, Generator, TypedDict, Literal, List, Callable, TypeVar, Tuple, Any
import torch
from collections import namedtuple
from pyarrow.parquet import ParquetFile
from transformers import AutoTokenizer
import torch.multiprocessing as mp
import traceback

T = TypeVar("T")

TensorDict = Dict[str, torch.Tensor]

class Message(TypedDict):
    role: Literal["user", "assistant", "system"]
    content: str

Conversation = List[Message]


def _generate_from_buffer(
    buffer: TensorDict, seq_len: int
) -> Generator[TensorDict, None, None]:
    """
    Generates TensorDicts from slices of the buffer that all have shape (seq_len,).

    We update the buffer in-place, so that at the end of generation, the shape
    of  buffer is (n,) where n < seq_len.
    """

    a_key = next(iter(buffer.keys()))
    while buffer[a_key].shape[0] >= seq_len:
        yield {k: buffer[k][:seq_len] for k in buffer.keys()}
        for k in buffer.keys():
            buffer[k] = buffer[k][seq_len:]


def pack_tensors(
    items: Iterator[TensorDict],
    *,
    seq_len: int,
    sep_ids: Dict[str, int],
) -> Generator[TensorDict, None, None]:
    """
    Packs and splits the TensorDicts from items into new TensorDicts of
    length at most seq_len.

    When multiple tensors are packed together, we separate them with sep_ids.
    You must specify a sep_id for each key: sep_ids[x] separates tensors for
    key x.
    """

    keys = sep_ids.keys()
    a_key = next(iter(keys))

    the_iter = iter(items)

    try:
        item = next(the_iter)
    except StopIteration:
        return
    current_buffer = {}
    separators = {}
    needs_separator = item[a_key].shape[0] % seq_len != 0
    for k in keys:
        separators[k] = torch.tensor([sep_ids[k]], dtype=item[k].dtype)
        current_buffer[k] = item[k]
    yield from _generate_from_buffer(current_buffer, seq_len)

    for item in the_iter:
        item_len = item[a_key].shape[0]
        for k in keys:
            if needs_separator:
                current_buffer[k] = torch.cat(
                    (current_buffer[k], separators[k], item[k])
                )
            else:
                current_buffer[k] = torch.cat((current_buffer[k], item[k]))
        needs_separator = current_buffer[a_key].shape[0] % seq_len != 0
        yield from _generate_from_buffer(current_buffer, seq_len)

    if current_buffer[a_key].shape[0] > 0:
        yield current_buffer


def chat_tensors(
    conversation: Conversation,
    tokenizer: AutoTokenizer,
    eot_token_id: int,
) -> TensorDict:
    """
    Tokenizes conversations. The returned TensorDict has input_ids,
    attention_mask, and prompt_mask. The prompt_mask is a binary mask with
    0 for the prompt token IDs, suitable for prompt loss masking.
    """
    completion = conversation[-1]["content"]
    prompt = conversation[:-1]
    prompt_input_ids = tokenizer.apply_chat_template(
        prompt,
        return_tensors="pt",
        tokenize=True,
        add_generation_prompt=True,
        padding=False,
    )[0]
    completion_input_ids = tokenizer(completion, return_tensors="pt", padding=False).input_ids[0]
    input_ids = torch.cat([
        prompt_input_ids,
        completion_input_ids,
        torch.tensor([eot_token_id], dtype=prompt_input_ids.dtype),
    ], dim=0)
    attention_mask = torch.ones_like(input_ids)
    prompt_mask = torch.zeros_like(input_ids)
    prompt_mask[prompt_input_ids.shape[0]:] = 1
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "prompt_mask": prompt_mask,
    }

def read_constant_length_chat_tensors(
    *,
    parquet_file_name: str,
    tokenizer: AutoTokenizer,
    seq_len: int,
    eot_token_id: int,
    content_field: str,
    batch_size: int,
) -> Generator[TensorDict, None, None]:
    """
    Reads conversations from a parquet file and packs them into constant-length
    sequences.

    We use tokenizer.eos_token_id to separate conversations in a sequence,
    and eot_token_id at the end of each conversation.
    """
    f = ParquetFile(parquet_file_name)

    def gen():
        for batch in f.iter_batches(columns=[content_field], batch_size=batch_size):
            conversations = [ item.as_py() for item in batch[content_field] ]
            for conversation in conversations:
                yield chat_tensors(conversation, tokenizer, eot_token_id)

    sep_ids = {
        "input_ids": tokenizer.eos_token_id,
        "attention_mask": 1,
        "prompt_mask": 1,   
    }
    
    try:
        yield from pack_tensors(gen(), seq_len=seq_len, sep_ids=sep_ids)
    finally:
        f.close()


def prompt_completion_tensors(
    prompt: str,
    completion: str,
    tokenizer: AutoTokenizer,
) -> TensorDict:
    """
    Tokenizes a prompt and completion into a TensorDict with input_ids,
    attention_mask, and prompt_mask. The prompt_mask is a binary mask with
    0 for the prompt token IDs, suitable for prompt loss masking.
    """
    prompt_input_ids = tokenizer(prompt, return_tensors="pt", padding=False).input_ids[0]
    completion_input_ids = tokenizer(completion, return_tensors="pt", padding=False).input_ids[0]
    input_ids = torch.cat([
        prompt_input_ids,
        completion_input_ids,
    ], dim=0)
    attention_mask = torch.ones_like(input_ids)
    prompt_mask = torch.zeros_like(input_ids)
    prompt_mask[prompt_input_ids.shape[0]:] = 1
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "prompt_mask": prompt_mask,
    }

def read_constant_length_prompt_completion_tensors(
    *,
    parquet_file_name: str,
    tokenizer: AutoTokenizer,
    seq_len: int,
    batch_size: int,
) -> Generator[TensorDict, None, None]:
    """
    Reads prompts and completions from a parquet file and packs them into constant-length
    sequences.
    """
    f = ParquetFile(parquet_file_name)

    def gen():
        for batch in f.iter_batches(columns=["prompt", "completion"], batch_size=batch_size):
            for prompt, completion in zip(batch["prompt"], batch["completion"]):
                yield prompt_completion_tensors(prompt.as_py(), completion.as_py(), tokenizer)

    sep_ids = {
        "input_ids": tokenizer.eos_token_id,
        "attention_mask": 1,
        "prompt_mask": 1,
    }
    try:
        yield from pack_tensors(gen(), seq_len=seq_len, sep_ids=sep_ids)
    finally:
        f.close()


def _multiprocess_worker(func, args, q, shutdown, done_ack):
    """Top-level worker function for multiprocess_generator (must be picklable)."""
    try:
        result = func(*args)
        for item in result:
            if shutdown.is_set():
                return
            q.put((True, item))
    except Exception as e:
        
        traceback.print_exc()
        try:
            q.put((False, e))
        except:
            pass
    finally:
        try:
            q.put((False, None))
        except:
            pass
        done_ack.wait(timeout=10.0)

def multiprocess_generator(
    *,
    func: Callable[[], Generator[T, None, None]],
    args: Tuple[Any, ...],
    queue_size: int
) -> Generator[T, None, None]:
    assert queue_size > 0, "have a positive queue size. Do not use an unbounded queue."

    q = mp.Queue(maxsize=queue_size)
    shutdown = mp.Event()
    done_ack = mp.Event()

    proc = mp.Process(
        target=_multiprocess_worker,
        args=(func, args, q, shutdown, done_ack),
    )
    proc.start()
    try:
        while True:
            has_item, item = q.get()
            if not has_item:
                if isinstance(item, Exception):
                    raise item
                break
            yield item
    finally:
        shutdown.set()
        done_ack.set()
        proc.join(timeout=1.0)
        if proc.is_alive():
            proc.terminate()
