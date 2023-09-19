"""Welcome to Pynecone! This file outlines the steps to create a basic app."""

# Import pynecone.
import os
from datetime import datetime

import pynecone as pc
from pynecone.base import Base

from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain

os.environ["OPENAI_API_KEY"] = ""

def read_file(file_path: str) -> str:
    with open(file_path, "r") as f:
        prompt_template = f.read()

    return prompt_template

kakao_sync_info = read_file("./assets/kakao_sync.txt")
question_template = read_file("./assets/question_template.txt")
trim_template = read_file("./assets/trim_template.txt")
kind_template = read_file("./assets/kind_template.txt")

def create_chain(llm, template, output_key):
    return LLMChain(
        llm=llm,
        prompt=ChatPromptTemplate.from_template(
            template=template,
        ),
        output_key=output_key,
        verbose=True,
    )

def chat_using_chatgpt(text, history) -> str:
    writer_llm = ChatOpenAI(temperature=0.1, max_tokens=300, model="gpt-3.5-turbo-16k")

    gpt_chain = create_chain(writer_llm, question_template, "answer")
    trim_chain = create_chain(writer_llm, trim_template, "trim_answer")
    kind_chain = create_chain(writer_llm, kind_template, "kind_answer")

    preprocess_chain = SequentialChain(
        chains=[
            gpt_chain,
            trim_chain
        ],
        input_variables=["info", "question", "history"],
        output_variables=["answer", "trim_answer"],
        verbose=True,
    )

    context = dict(
        info=kakao_sync_info,
        question=text,
        history=history
    )
    context = preprocess_chain(context)

    context = kind_chain(context)

    return context["kind_answer"]


class Message(Base):
    question_text: str
    answer_text: str
    created_at: str


class State(pc.State):
    """The app state."""

    text: str = ""
    messages: list[Message] = []

    @pc.var
    def output(self) -> str:
        if not self.text.strip():
            return "Answer will appear here."
        answer = chat_using_chatgpt(self.text, self.messages)
        return answer

    def post(self):
        self.messages = [
            Message(
                question_text=self.text,
                answer_text=self.output,
                created_at=datetime.now().strftime("%B %d, %Y %I:%M %p")
            )
        ] + self.messages


# Define views.


def header():
    """Basic instructions to get started."""
    return pc.box(
        pc.text("HelperBot 🗺", font_size="2rem"),
        pc.text(
            "카카오싱크에 대해 물어보세요!",
            margin_top="0.5rem",
            color="#666",
        ),
    )


def down_arrow():
    return pc.vstack(
        pc.icon(
            tag="arrow_down",
            color="#666",
        )
    )


def text_box(text):
    return pc.text(
        text,
        background_color="#fff",
        padding="1rem",
        border_radius="8px",
    )


def message(message):
    return pc.box(
        pc.vstack(
            text_box(message.question_text),
            down_arrow(),
            text_box(message.answer_text),
            pc.box(
                pc.text(" · ", margin_x="0.3rem"),
                pc.text(message.created_at),
                display="flex",
                font_size="0.8rem",
                color="#666",
            ),
            spacing="0.3rem",
            align_items="left",
        ),
        background_color="#f5f5f5",
        padding="1rem",
        border_radius="8px",
    )


def smallcaps(text, **kwargs):
    return pc.text(
        text,
        font_size="0.7rem",
        font_weight="bold",
        text_transform="uppercase",
        letter_spacing="0.05rem",
        **kwargs,
    )


def output():
    return pc.box(
        pc.box(
            smallcaps(
                "Output",
                color="#aeaeaf",
                background_color="white",
                padding_x="0.1rem",
            ),
            position="absolute",
            top="-0.5rem",
        ),
        pc.text(State.output),
        padding="1rem",
        border="1px solid #eaeaef",
        margin_top="1rem",
        border_radius="8px",
        position="relative",
    )


def index():
    """The main view."""
    return pc.container(
        header(),
        pc.input(
            placeholder="Text to question",
            on_blur=State.set_text,
            margin_top="1rem",
            border_color="#eaeaef"
        ),
        output(),
        pc.button("Post", on_click=State.post, margin_top="1rem"),
        pc.vstack(
            pc.foreach(State.messages, message),
            margin_top="2rem",
            spacing="1rem",
            align_items="left"
        ),
        padding="2rem",
        max_width="600px"
    )


# Add state and page to the app.
app = pc.App(state=State)
app.add_page(index, title="helper Bot")
app.compile()
