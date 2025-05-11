import pydantic


class SolutionsRow(pydantic.BaseModel):
    idx: int
    source_id: str

    prompt: str
    response: str

    problem_statement: str
    time_limit: float|None
    memory_limit: float|None
    input_format: str|None
    output_format: str|None
    examples: list['IOExample']
    problem_notes: str|None

    title: str
    contest_name: str
    contest_start_year: int


class IOExample(pydantic.BaseModel):
    input: str
    output: str


class SolutionsRowWithAnswer(SolutionsRow):
    answer: str | None