from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from streamlit_chat import message
import streamlit as st
import os
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"


# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 AI Chatbot")

# ─── Session state for history ────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi I am Flan T5 Chatbot. How can I help you?"}
    ]

# ─── Load model & tokenizer ──────────────────────────────────────────────────


@st.cache_resource(show_spinner=True)
def load_model_tokenizer():
    peft_model_id = "lora-flan-t5-large-chat"
    config = PeftConfig.from_pretrained(peft_model_id)
    base = config.base_model_name_or_path

    model = AutoModelForSeq2SeqLM.from_pretrained(base)
    tokenizer = AutoTokenizer.from_pretrained(base)
    model = PeftModel.from_pretrained(model, peft_model_id).to("cpu")
    model.eval()
    return model, tokenizer


model, tokenizer = load_model_tokenizer()

# ─── Inference helper ─────────────────────────────────────────────────────────


def inference(prompt: str) -> str:
    ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).input_ids.to("cpu")
    outs = model.generate(input_ids=ids,  do_sample=True, top_p=0.9, max_length=256)
    return tokenizer.decode(outs[0], skip_special_tokens=True)


# ─── Render all prior messages ────────────────────────────────────────────────
for i, msg in enumerate(st.session_state.messages):
    message(
        msg["content"],
        is_user=(msg["role"] == "user"),
        key=f"msg_{i}"
    )

# ─── Chat input & processing ─────────────────────────────────────────────────
user_input = st.chat_input("Type your message here...")

if user_input:
    # 1️⃣ Immediately append & show the user's message
    st.session_state.messages.append({"role": "user", "content": user_input})
    message(user_input, is_user=True, key=f"user_{len(st.session_state.messages)}")

    # 2️⃣ Show a temporary “Thinking…” bubble
    spinner_slot = st.empty()
    with spinner_slot.container():
        message("Thinking... 🤔", is_user=False, key="thinking")

    # 3️⃣ Run the model
    prompt = f"Human: {user_input}\nAssistant:"
    bot_reply = inference(prompt)

    # 4️⃣ Remove the spinner and append the real response
    spinner_slot.empty()
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
    message(bot_reply, is_user=False, key=f"bot_{len(st.session_state.messages)}")
