import re

from loguru import logger

from .py_parser import check_is_python


offset_rx = re.compile(r'^<think>\n?')
par_sep_rx = re.compile(r'\n\n+|(\n?</think>\n*)', re.MULTILINE)
fence_start_line_rx = re.compile(r'^```(.*)\n+', re.MULTILINE)
fence_end_line_rx = re.compile(r'^```($|\n+)', re.MULTILINE)
fence_rx = re.compile(r'```(.*)\n+')
has_input_rx = re.compile(r'input\(\)|stdin')
has_output_rx = re.compile(r'print\(|stdout')
leading_space_rx = re.compile(r'^\s+')
nonspace_char_rx = re.compile(r'\S')
thinks_end_rx = re.compile(r'\n*</think>\n*')

def looks_like_answer(code: str) -> bool:
    """
    Allows filtering the results of find_code_blocks.
    """
    return bool(
        has_input_rx.search(code) and
        has_output_rx.search(code)
    )


def clean_backtick_fences(string: str) -> str:
    offset = 0
    end = len(string)
    if m := leading_space_rx.search(string):
        offset = m.end()
    if m := fence_rx.match(string, pos=offset):
        offset = m.end()
    if m := fence_end_line_rx.search(string, pos=offset):
        end = m.start()
    return string[offset:end]


def find_code_blocks(
    response: str,
    offset: int = 0,
    only_process_thinks: bool = True,
) -> tuple[list[tuple[str, int]], int]:
    results: list[tuple[str, int]] = []

    cur_idx = offset
    cur_par_start = cur_idx
    cur_match = par_sep_rx.search(response, cur_idx)
    last_par_was_code = False
    loop = True
    while loop:
        par = None
        cur_log_block_start = cur_par_start
        if cur_match and (cur_match.group(1) is None or not only_process_thinks):
            cur_par_end = cur_match.start()

            cur_idx = cur_match.end()
            cur_match = par_sep_rx.search(response, cur_idx)


            par = response[cur_par_start:cur_par_end]
            cur_par_start = cur_idx
        else:
            loop = False
            if cur_match:
                cur_par_end = cur_match.start()
                cur_idx = cur_match.end()
                par = response[cur_par_start:cur_par_end]
            else:
                cur_idx = cur_par_end = len(response)

        found_code = False
        whitespace_offset = 0
        if par is not None:
            if m := nonspace_char_rx.search(par):
                whitespace_offset = m.start()

        if par and (
            par.startswith(('```', '#', 'import ', 'from '), whitespace_offset) or
            len(par) < 100 or
            '\n' in par
        ):
            par = clean_backtick_fences(par)
            if last_par_was_code:
                last_par_str, cur_log_block_start = results[-1]
                par = ''.join((last_par_str, '\n\n', par))

            if check_is_python(par)[0]:
                cur_res = (par, cur_log_block_start)
                if last_par_was_code:
                    results[-1] = cur_res
                else:
                    results.append(cur_res)
                found_code = True

        last_par_was_code = found_code

    return results, cur_idx


def find_final_answer_block(
    response: str,
    offset: int,
    answer_must_be_valid_python: bool = False,
) -> str | None:
    if offset == len(response):
        return None

    results = []

    cur_idx = offset
    cur_par_start = cur_idx
    cur_match = fence_start_line_rx.search(response, cur_idx)
    loop = True
    while loop:
        if cur_match:
            cur_par_start = cur_match.end()
            fence_end_match = fence_end_line_rx.search(response, cur_match.end())
            if not fence_end_match:
                break

            cur_par_end = fence_end_match.start()

            par = response[cur_par_start:cur_par_end]

            cur_idx = fence_end_match.end()
            cur_match = fence_start_line_rx.search(response, cur_idx)
        else:
            loop = False
            if cur_match:
                cur_par_end = cur_match.start()
                cur_idx = cur_match.end()
            else:
                cur_idx = cur_par_end = len(response)
            par = response[cur_par_start:cur_par_end]

        if (
            any(par.startswith(p) for p in ('```', 'import ', 'from ')) or
            len(par) < 100 or
            '\n' in par
        ):
            par = clean_backtick_fences(par)
            if not answer_must_be_valid_python or check_is_python(par)[0]:
                results.append(par)

    return results[-1] if results else None


def find_code(response: str) -> tuple[list[str], str | None]:
    offset = 0
    if m := offset_rx.search(response):
        offset = m.end()

    think_code_blocks, offset = find_code_blocks(response, offset)

    final_answer = find_final_answer_block(response, offset)

    return [r[0] for r in think_code_blocks], final_answer