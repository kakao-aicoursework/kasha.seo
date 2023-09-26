def read_text(file_path: str) -> str:
    with open(file_path, "r") as f:
        text = f.read().splitlines()[0]
    return text

def read_prompt_template(file_path: str) -> str:
    with open(file_path, "r") as f:
        text = f.read()
    return text
