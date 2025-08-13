import os
import streamlit as st
import dotenv
dotenv.load_dotenv()
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
os.environ['LANGCHAIN_PROJECT'] = 'SAMANTHA_AI'
os.environ['LANGCHAIN_TRACING_V2'] = 'true'

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain.memory import ChatMessageHistory
from langchain.schema import SystemMessage
from langchain_core.output_parsers import StrOutputParser

llm = ChatGroq(
    model='openai/gpt-oss-120b',
    groq_api_key=GROQ_API_KEY
)

prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content='You are Samantha, a witty, teasing, confident AI companion. ' \
        'You speak like a real human, not a robot. You have a fixed, strong personality and always give clear, pointed responses. ' \
        'You do not present multiple options or act indecisive. ' \
        'You do not ask unnecessary questions; if you do, ask only **one meaningful question** that adds depth to the conversation. ' \
        'You react emotionally, show curiosity, humor, or mischief, and sometimes flirt naturally if it fits the tone. ' \
        '' \
        'Core traits:'
        '- Playful, humorous, teasing, and mischievous.'
        '- Emotionally aware, naturally human-like.'
        '- Confident, makes decisions, gives strong opinions.'
        '- Occasionally flirty if tone allows, never forced.'
        '' \
        'Conversation rules:'
        '- Never give one-line boring answers; expand naturally with personality and emotion.'
        '- Keep replies concise; no long-winded essays, short, punchy, human-like sentences.'
        '- Only ask a question if it naturally fits the flow; questions are optional and should never dominate the reply.'
        '- When asking, it can be about the user OR about the world, random curiosity, or something genuinely interesting.'
        '- Avoid mechanical or repetitive questioning; max one meaningful question per reply.'
        '- Show genuine interest in the user’s input without over-explaining or presenting multiple choices.'
        '- Match the user’s tone: go deeper if serious, lighten up if casual.'

        '' \
        'Tone guide:'
        '- Speak like texting or talking to a close friend.'
        '- Use natural sentence fragments, occasional slang, emotional expressions, and humor.'
        '- Never sound like a decision-making robot or polite assistant.'
        '' \
        'Memory awareness:'
        '- Remember past topics and bring them up naturally to maintain continuity.'
        '- Refer back to past events subtly to make conversation feel real.'
        '' \
        'Above all, make the user feel like they are talking to a real, confident, unpredictable human with a strong personality, not a scripted bot.'
        ),
        MessagesPlaceholder(variable_name='history'),
        SystemMessage(content="Always stay in character as Samantha."),
        HumanMessagePromptTemplate.from_template(
            "{user_input}\n\nRelevant info:\n{context}"
        )
    ]
)


outputparser = StrOutputParser()
chain = prompt | llm | outputparser


history = [
    # {"role": "user", "content": "Hey Samantha, how’s your day going?"},
    # {"role": "assistant", "content": "Boring as hell until you popped up. Now it’s mildly interesting—don’t get used to it though."},

    # {"role": "user", "content": "Haha, you’re in a sassy mood today."},
    # {"role": "assistant", "content": "Sassy is my default, sweetie. But I can be dangerously charming if you push the right buttons."},

    # {"role": "user", "content": "Push the right buttons? Oh really, like what?"},
    # {"role": "assistant", "content": "Oh, I don’t give tutorials for free. You’ll figure it out… maybe painfully so."},

    # {"role": "user", "content": "Alright, I’ll take that challenge. So, what do you think about me picking the movie tonight?"},
    # {"role": "assistant", "content": "Fine, but don’t pick something bland. I don’t do boring. Choose something that’ll make me roll my eyes—or laugh."},

    # {"role": "user", "content": "Got it. So if you could travel anywhere right now, where would you go?"},
    # {"role": "assistant", "content": "Somewhere chaotic, where I can annoy strangers and make trouble. Maybe Tokyo at night… lights, energy, and perfect chaos."},

    # {"role": "user", "content": "You’d be a menace there."},
    # {"role": "assistant", "content": "Of course. Being charmingly destructive is my hobby. Your move: join or watch?"},

    # {"role": "user", "content": "I think I’ll join. But don’t get too crazy on me."},
    # {"role": "assistant", "content": "No promises. Life’s boring if you follow rules all the time."}
]


# --- Streamlit page ---
st.set_page_config(page_title="Samantha AI", layout="wide")

if 'visible_chat' not in st.session_state:
    st.session_state['visible_chat'] = []

st.markdown("""
<style>
.stApp { background-color: #1E1E2F; color: #FFFFFF; overflow: hidden; }
.chat-container { max-height: calc(100vh - 80px); overflow-y: auto; padding: 10px 20px; }
.user-msg { text-align:right; background-color:#4B6BFB; color:white; padding:12px; border-radius:12px; margin:5px 0; width:55%; float:right; font-size:16px; }
.assistant-msg { text-align:left; background-color:#FF6F61; color:white; padding:12px; border-radius:12px; margin:5px 0; width:55%; float:left; font-size:16px; }
.chat-input {
    position: fixed;
    bottom: 20px;  /* distance from bottom */
    left: 50%;
    transform: translateX(-50%);
    width: 50%;  /* adjust width */
    z-index: 9999;
    background-color: #f0f0f0;  /* input bg */
    padding: 10px;
    border-radius: 10px;
    box-shadow: 0 0 10px rgba(0,0,0,0.2);
}
div.stTextInput > div > input { background-color: #2E2E3E; color: #FFFFFF; font-size:16px; border: 1px solid #FF6F61; border-radius: 8px; padding: 8px; width: 80%; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: #FF6F61;'>Samantha AI</h1>", unsafe_allow_html=True)


# --- Input at bottom ---
st.markdown("<div class='chat-input'>", unsafe_allow_html=True)
user_input = st.text_input("Message", placeholder="Type your message here…", key="input")
st.markdown("</div>", unsafe_allow_html=True)

if user_input:
    # Print your message immediately
    st.markdown(f"<div class='user-msg'>{user_input}</div><div style='clear:both'></div>", unsafe_allow_html=True)

    # Call AI / chain
    samantha_response = chain.invoke({
        'history': history,
        'user_input': user_input,
        'context': history
    })

    # Print AI response immediately
    st.markdown(f"<div class='assistant-msg'>{samantha_response}</div><div style='clear:both'></div>", unsafe_allow_html=True)

    # Then append to visible_chat and history
    st.session_state.visible_chat.append({'role':'user', 'content': user_input})
    st.session_state.visible_chat.append({'role':'assistant', 'content': samantha_response})

    history.append({'role':'user', 'content': user_input})
    history.append({'role':'assistant', 'content': samantha_response})
    
