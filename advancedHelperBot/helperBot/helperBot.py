"""Welcome to Pynecone! This file outlines the steps to create a basic app."""

# Import pynecone.
import os
from datetime import datetime

import pynecone as pc
from pynecone.base import Base

import openai
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains import ConversationChain, LLMChain
from langchain.chains import SequentialChain
from . import fileLoader
from . import chroma
from . import chatHistory

INTENT_PATH = "./assets/prompt/intent.txt"
PARSE_INTENT_PATH = "./assets/prompt/parse_intent.txt"
RESPONSE_TEMPLATE_PATH = "./assets/prompt/response_template.txt"
HISTORY_RESPONSE_TEMPLATE_PATH = "./assets/prompt/history_response_template.txt"
HISTORY_DIR = "./chat_histories"

def create_chain(llm, template, output_key):
    return LLMChain(
        llm=llm,
        prompt=ChatPromptTemplate.from_template(
            template=fileLoader.read_prompt_template(template),
        ),
        output_key=output_key,
        verbose=True,
    )
#
llm = ChatOpenAI(temperature=0.1, model="gpt-3.5-turbo")
default_chain = ConversationChain(llm=llm, output_key="output")

parse_intent_chain = create_chain(
    llm=llm,
    template=PARSE_INTENT_PATH,
    output_key="intent"
)

response_chain = create_chain(
    llm=llm,
    template=RESPONSE_TEMPLATE_PATH,
    output_key="answer"
)

history_response_chain = create_chain(
    llm=llm,
    template=HISTORY_RESPONSE_TEMPLATE_PATH,
    output_key="history_answer"
)

def generate_answer(user_message, conversation_id: str='fa1010') -> dict[str, str]:
    history_file = chatHistory.load_conversation_history(conversation_id)

    context = dict(user_message=user_message)
    context["input"] = context["user_message"]
    context["intent_list"] = fileLoader.read_prompt_template(INTENT_PATH)
    context["chat_history"] = chatHistory.get_chat_history(conversation_id)

    answer = history_response_chain.run(context)
    if (answer == "Search"):
        intent = parse_intent_chain.run(context)
        print("intent: " + intent)
        if intent == "default":
            answer = default_chain.run(context["user_message"])
        else:
            context["related_documents"] = chroma.query_db(context["user_message"])
            answer = response_chain.run(context)

    chatHistory.log_user_message(history_file, user_message)
    chatHistory.log_bot_message(history_file, answer)

    return {"answer": answer}

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
        answer = generate_answer(self.text)["answer"]
        print("kasha")
        print("answer: " + answer)
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
            "카카오 기능에 대해 물어보세요! (카카오소셜, 카카오싱크, 카카오톡채녈)",
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
