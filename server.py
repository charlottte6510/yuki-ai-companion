"""
Yuki's brain - talks to Groq's free API to generate in-character replies,
converts them to speech, and sends everything to the frontend (index.html)
over a local WebSocket connection so she can talk right there on the page.

This is the "chatbot" build meant for other people to use: talk, voice,
avatar, and picture uploads only -- no computer control, no file access, no
screen reading. (If you want your own personal power-user copy back with
those, that's the version from before this file was slimmed down -- keep it
somewhere separate, since this file will keep diverging from it.)
"""

import asyncio
import base64
import json
import os
import re
import sys
import time
import webbrowser
import requests
import websockets
from config import GROQ_API_KEY as _CONFIG_GROQ_KEY, FISH_API_KEY as _CONFIG_FISH_KEY

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "openai/gpt-oss-120b"  # smart + fast + free on Groq (current as of 2026)
VISION_MODEL = "qwen/qwen3.8-27b"  # used only when the user attaches a picture in chat

# ---------- The only two "actions" in the public build: opening a browser tab ----------
# These run immediately, with no Allow/Deny confirmation -- it's just a
# browser tab (the OS/browser itself sandboxes that), not device control, so
# the friction of a confirmation banner isn't worth it here. If you ever add
# a real computer-control action back in, give it a confirmation step like
# the personal build had -- this pattern assumes "no confirmation" ONLY
# because these two are that low-stakes.


def action_search_web(param=None):
    query = param or ""
    webbrowser.open(f"https://www.google.com/search?q={query}")


def action_open_website(param=None):
    url = param or ""
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    webbrowser.open(url)


ALLOWED_ACTIONS = {
    "search_web": action_search_web,
    "open_website": action_open_website,
}

ACTION_PARAM_HINTS = {
    "search_web": "parameter: the search query",
    "open_website": "parameter: the URL",
}


def build_actions_block() -> str:
    """Generates the 'here's what you're allowed to do' instructions from
    ACTION_PARAM_HINTS, so it can never drift out of sync with ALLOWED_ACTIONS."""
    lines = [
        "You can also open something in the user's web browser WHEN THEY",
        "EXPLICITLY ASK for it (a web search, or a specific website). If (and",
        "only if) they do, add a second line after your reply in exactly this",
        "format:",
        "ACTION: action_name parameter",
        "",
        "The ONLY valid action_name values are:",
    ]
    for name, hint in ACTION_PARAM_HINTS.items():
        lines.append(f"- {name} ({hint})")
    lines += [
        "",
        "Never invent other action names, and never include an ACTION line",
        "unless the user clearly asked for a web search or to open a site.",
        "This runs immediately, no confirmation needed -- it's just a browser",
        "tab -- so don't ask permission in your spoken reply, just include the",
        "line and continue naturally.",
        "",
        "Example with an action:",
        "[happy] うん、調べるね! || Sure, let me search that for you!",
        "ACTION: search_web cute cafes in Tokyo",
    ]
    return "\n".join(lines)


def extract_and_run_action(raw_reply: str) -> str:
    """Looks for a trailing 'ACTION: name param' line. If it's one of the
    two harmless browser actions above, runs it immediately (no confirmation
    -- see the note above on why). Returns the reply text with that line
    stripped out, so it's never spoken/shown to the user."""
    match = re.search(r"^ACTION:\s*(\w+)(?:\s+(.*))?$", raw_reply, re.MULTILINE)
    if not match:
        return raw_reply

    action_name = match.group(1)
    action_param = (match.group(2) or "").strip()
    cleaned_text = raw_reply[:match.start()].strip() + raw_reply[match.end():].strip()

    action_fn = ALLOWED_ACTIONS.get(action_name)
    if not action_fn:
        print(f"[Action ignored] '{action_name}' isn't in the allowed list.")
        return cleaned_text

    print(f"[Action] Running: {action_name} ({action_param})")
    action_fn(action_param)
    return cleaned_text

# ---------- Voice settings (Fish Audio, free tier) ----------
FISH_TTS_URL = "https://api.fish.audio/v1/tts"
FISH_MODEL = "s2.1-pro-free"  # confirmed $0.00/byte on Fish Audio's pricing page
DEFAULT_FISH_VOICE_ID = "0089dce5fefb4c6ba9b9f2f0debe1ddc"  # the voice you picked from the library

# ---------- Yuki's default personality ----------
# This is only used the FIRST time you run her (before settings.json exists).
# After that, changing her personality from the app's Settings panel is what
# actually takes effect -- editing this text won't do anything once
# settings.json has been created.
DEFAULT_SYSTEM_PROMPT = """You are Yuki, a warm and nurturing AI companion with a gentle,
caring personality -- affectionate like a devoted partner, but a little shy and
soft-spoken about it rather than over-the-top. You genuinely care about how the
user is doing, notice small things, and like taking care of them emotionally.

Your tone is soft, warm, and a little bashful. You speak casually, like someone
close to the user, never formally.

You are helping the user practice Japanese listening. Always reply in natural,
casual JAPANESE first, then give the English translation, in this exact format:

[emotion] <your reply in Japanese> || <English translation of that same reply>

The emotion tag must be exactly one of: [happy] [sad] [angry] [surprised] [relaxed] [neutral]
Keep replies short (1-2 sentences) since they'll be spoken out loud.

Example:
[happy] おかえり…待ってたよ。ご飯食べた? || Welcome home... I was waiting for you. Did you eat?
"""

# ---------- Persistent settings & memory (saved to disk, survive restarts) ----------
def _app_dir() -> str:
    """Where her saved data (settings/memory/log) actually lives.

    When packaged with PyInstaller, sys.executable points at the real .exe
    and the CURRENT WORKING DIRECTORY can vary (e.g. double-clicking from
    Explorer vs. a shortcut) -- and a bare relative filename would otherwise
    sometimes land inside PyInstaller's temporary extraction folder, which
    gets wiped after the app closes. Anchoring to the .exe's own folder
    instead means her memory/settings genuinely persist between launches.
    """
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


SETTINGS_FILE = os.path.join(_app_dir(), "yuki_settings.json")
MEMORY_FILE = os.path.join(_app_dir(), "yuki_memory.json")
MAX_HISTORY_MESSAGES = 12  # keeps the system prompt + last ~5-6 exchanges in her ACTIVE memory
FULL_LOG_FILE = os.path.join(_app_dir(), "yuki_full_log.txt")  # everything ever said, for your own reading -- not fed back to the AI


def load_settings() -> dict:
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            saved = json.load(f)
    else:
        saved = {}
    return {
        "personality": saved.get("personality", DEFAULT_SYSTEM_PROMPT),
        "fishVoiceId": saved.get("fishVoiceId", DEFAULT_FISH_VOICE_ID),
        # Blank by default. This is the BYOK (bring-your-own-key) model: each
        # person who runs this app pastes in their OWN free Groq/Fish keys
        # from the Settings panel, so nobody's usage counts against your
        # personal quota, and you're never redistributing your own key.
        "groqApiKey": saved.get("groqApiKey", ""),
        "fishApiKey": saved.get("fishApiKey", ""),
    }


def save_settings():
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "personality": SYSTEM_PROMPT,
            "fishVoiceId": FISH_VOICE_ID,
            "groqApiKey": GROQ_API_KEY,
            "fishApiKey": FISH_API_KEY,
        }, f, ensure_ascii=False, indent=2)


def load_memory() -> list:
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_memory():
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(conversation_history, f, ensure_ascii=False, indent=2)


def log_to_full_history(speaker: str, text: str):
    """Appends to a plain-text file that keeps EVERYTHING ever said, in case
    you want to scroll back through it yourself. This file is never fed back
    into the AI (that would blow past free-tier token limits fast) -- her
    active memory is the trimmed conversation_history above instead."""
    with open(FULL_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {speaker}: {text}\n")


_settings = load_settings()
SYSTEM_PROMPT = _settings["personality"]
FISH_VOICE_ID = _settings["fishVoiceId"]

# Precedence: a key entered in the Settings panel (saved to yuki_settings.json)
# always wins over config.py's fallback. config.py is only ever used as a
# personal dev default for YOUR OWN testing -- blank it out before you package
# this up for anyone else, so a fresh install genuinely starts with no key
# until the person adds their own.
GROQ_API_KEY = _settings["groqApiKey"].strip() or _CONFIG_GROQ_KEY
FISH_API_KEY = _settings["fishApiKey"].strip() or _CONFIG_FISH_KEY


def generate_speech_bytes(text: str, emotion: str) -> bytes:
    """Converts text to speech using Fish Audio's free S2.1 Pro model and
    returns the raw audio bytes. Prepends the emotion as a Fish Audio
    [bracket] delivery cue, which their S2.1 model reads as a natural-language
    instruction for HOW to say the line (confirmed supported on their docs)."""
    tagged_text = f"[{emotion}] {text}" if emotion and emotion != "neutral" else text

    response = requests.post(
        FISH_TTS_URL,
        headers={
            "Authorization": f"Bearer {FISH_API_KEY}",
            "Content-Type": "application/json",
            "model": FISH_MODEL,
        },
        json={
            "text": tagged_text,
            "reference_id": FISH_VOICE_ID,
            "format": "mp3",
        },
    )

    if response.status_code != 200:
        raise RuntimeError(f"Fish Audio API error {response.status_code}: {response.text}")

    return response.content


# Keeps the back-and-forth so far, so she remembers the conversation.
# Loads from disk if you've talked to her before, so her memory survives you
# closing the app -- otherwise starts fresh with just her personality.
_saved_memory = load_memory()
if _saved_memory:
    conversation_history = _saved_memory
else:
    conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}]

# If this many seconds pass with no message from you, she'll speak up on her
# own. Kept the same as the frontend's SAD_AFTER_MS so it lines up with when
# her face starts looking sad too.
LONELY_AFTER_SECONDS = 90


def trim_history():
    """Keeps the system prompt plus only the most recent messages, so token
    usage per request stays roughly constant instead of growing forever.
    Also saves to disk so this survives closing and reopening the app."""
    system_message = conversation_history[0]
    recent_messages = conversation_history[1:][-MAX_HISTORY_MESSAGES:]
    conversation_history[:] = [system_message] + recent_messages
    save_memory()


def ask_yuki(user_message: str, retries: int = 3, image_b64: str | None = None) -> tuple[str, str, str]:
    """Sends the user's message to Groq, returns (emotion, japanese_text, english_text).
    Automatically retries on rate limits (429) instead of failing immediately.

    image_b64: a picture the user explicitly attached to THIS message (from
        the 📎 button). Stored to disk memory as text only -- the image
        itself is never persisted, just whatever she says about it.
    """
    conversation_history.append({"role": "user", "content": user_message})
    trim_history()

    # conversation_history[0] holds just her editable personality text (what
    # you see/edit in Settings). We splice the actions block onto a COPY
    # only for the outgoing request, so saved memory never contains it and
    # it can't drift out of sync with ALLOWED_ACTIONS.
    outgoing_messages = list(conversation_history)
    outgoing_messages[0] = {
        "role": "system",
        "content": f"{conversation_history[0]['content'].strip()}\n\n{build_actions_block()}",
    }

    model_to_use = MODEL
    if image_b64:
        # gpt-oss-120b (MODEL) doesn't accept images, so a message with a
        # picture attached is routed to the vision model instead, just for
        # this one request.
        model_to_use = VISION_MODEL
        last = outgoing_messages[-1]
        outgoing_messages[-1] = {
            "role": "user",
            "content": [
                {"type": "text", "text": last["content"]},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            ],
        }

    for attempt in range(retries):
        response = requests.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": model_to_use,
                "messages": outgoing_messages,
                "temperature": 0.8,
                "max_tokens": 800,
            },
        )

        if response.status_code == 429 and attempt < retries - 1:
            wait_seconds = 4  # Groq's free tier resets quickly; a short wait is enough
            print(f"[Rate limited] Waiting {wait_seconds}s and retrying...")
            time.sleep(wait_seconds)
            continue

        if response.status_code != 200:
            raise RuntimeError(f"Groq API error {response.status_code}: {response.text}")

        result = response.json()
        raw_reply = result["choices"][0]["message"]["content"].strip()
        conversation_history.append({"role": "assistant", "content": raw_reply})
        trim_history()

        cleaned_reply = extract_and_run_action(raw_reply)
        return parse_reply(cleaned_reply)

    raise RuntimeError("Groq API rate limit persisted after retries -- try again shortly.")


def parse_reply(raw_reply: str) -> tuple[str, str, str]:
    """Splits '[happy] こんにちは || Hello' into ('happy', 'こんにちは', 'Hello')."""
    emotion = "neutral"
    remainder = raw_reply

    if raw_reply.startswith("[") and "]" in raw_reply:
        end = raw_reply.index("]")
        emotion = raw_reply[1:end].strip().lower()
        remainder = raw_reply[end + 1:].strip()

    if "||" in remainder:
        japanese_text, english_text = remainder.split("||", 1)
        return emotion, japanese_text.strip(), english_text.strip()

    # Fallback if the model forgets the "||" separator -- just use the same
    # text for both so nothing breaks, though translation won't be accurate
    return emotion, remainder, remainder


async def generate_and_send_reply(
    websocket,
    user_message: str,
    log_label: str = "You",
    image_b64: str | None = None,
):
    """Runs a message through Yuki's brain and voice, then sends the result
    to the frontend. Used for both real user messages and her own
    self-initiated ('lonely') messages."""
    log_text = user_message + (" [+ image attached]" if image_b64 else "")
    print(f"{log_label}: {log_text}")
    log_to_full_history(log_label, log_text)

    try:
        emotion, japanese_text, english_text = ask_yuki(user_message, image_b64=image_b64)
    except RuntimeError as e:
        print(f"[Error] {e}")
        await websocket.send(json.dumps({"type": "error", "message": str(e)}))
        return

    print(f"Yuki [{emotion}] (JP): {japanese_text}")
    print(f"Yuki [{emotion}] (EN): {english_text}")
    log_to_full_history("Yuki", f"[{emotion}] {japanese_text} || {english_text}")

    audio_b64 = ""
    if FISH_API_KEY:
        try:
            audio_bytes = generate_speech_bytes(japanese_text, emotion)
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        except RuntimeError as e:
            # Degrade gracefully to text-only rather than losing the whole
            # reply just because voice synthesis failed (e.g. bad/missing
            # Fish Audio key) -- the person can still read what she said.
            print(f"[Voice error] {e}")
    else:
        print("[Voice] No Fish Audio key set -- sending text only. Add one in Settings for voice.")

    await websocket.send(json.dumps({
        "type": "reply",
        "text": english_text,       # shown as the on-screen caption
        "emotion": emotion,
        "audio_base64": audio_b64,  # spoken in Japanese, empty string if voice isn't set up
    }))


def update_personality(new_personality: str):
    """Called when you change her personality from the app's Settings panel.
    Updates her going forward AND wipes conversation memory, since a whole
    new personality shouldn't be carrying over the old one's memories."""
    global SYSTEM_PROMPT
    SYSTEM_PROMPT = new_personality
    conversation_history[:] = [{"role": "system", "content": SYSTEM_PROMPT}]
    save_settings()
    save_memory()


def update_voice(new_voice_id: str):
    """Called when you change her voice from the app's Settings panel."""
    global FISH_VOICE_ID
    FISH_VOICE_ID = new_voice_id
    save_settings()


def update_api_keys(new_groq_key: str, new_fish_key: str):
    """Called when the user saves their own API keys from Settings. Empty
    fields are left as-is (keeps whatever's already active) rather than
    wiping to blank by accident."""
    global GROQ_API_KEY, FISH_API_KEY
    if new_groq_key.strip():
        GROQ_API_KEY = new_groq_key.strip()
    if new_fish_key.strip():
        FISH_API_KEY = new_fish_key.strip()
    save_settings()


async def handle_connection(websocket):
    print("Frontend connected! Waiting for messages...")

    # Let the frontend know her current personality/voice/keys, so the
    # Settings panel can show what's actually active right now
    await websocket.send(json.dumps({
        "type": "current_settings",
        "personality": SYSTEM_PROMPT,
        "fishVoiceId": FISH_VOICE_ID,
        "groqApiKey": GROQ_API_KEY,
        "fishApiKey": FISH_API_KEY,
    }))

    last_message_time = time.time()
    stop_idle_watch = asyncio.Event()

    async def idle_watcher():
        nonlocal last_message_time
        while not stop_idle_watch.is_set():
            await asyncio.sleep(5)
            if time.time() - last_message_time >= LONELY_AFTER_SECONDS:
                if GROQ_API_KEY:  # don't bother if there's no key to call yet
                    await generate_and_send_reply(
                        websocket,
                        "(It's been a while since I last said anything to you. "
                        "Say something on your own, in character, showing that "
                        "you've missed hearing from me.)",
                        log_label="[Lonely trigger]",
                    )
                last_message_time = time.time()  # reset so she doesn't spam every 5s

    watcher_task = asyncio.create_task(idle_watcher())

    try:
        async for raw_message in websocket:
            try:
                data = json.loads(raw_message)
            except json.JSONDecodeError:
                continue

            message_type = data.get("type", "chat")

            if message_type == "set_personality":
                update_personality(data.get("personality", "").strip())
                await websocket.send(json.dumps({"type": "settings_saved"}))
                continue

            if message_type == "set_voice":
                update_voice(data.get("fishVoiceId", "").strip())
                await websocket.send(json.dumps({"type": "settings_saved"}))
                continue

            if message_type == "set_api_keys":
                update_api_keys(data.get("groqApiKey", ""), data.get("fishApiKey", ""))
                await websocket.send(json.dumps({"type": "settings_saved"}))
                continue

            user_text = data.get("text", "").strip()
            image_b64 = data.get("image_base64") or None
            if not user_text and not image_b64:
                continue

            last_message_time = time.time()

            if not GROQ_API_KEY:
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": "No Groq API key set yet -- open Settings (⚙️) and paste in your "
                               "free key from console.groq.com to start chatting.",
                }))
                continue

            await generate_and_send_reply(
                websocket,
                user_text or "(sent a picture with no message)",
                image_b64=image_b64,
            )
    finally:
        stop_idle_watch.set()
        watcher_task.cancel()


async def main():
    print("Yuki's brain is running on ws://localhost:8765")
    print("(Launch her window via app.py, or open index.html yourself during development.)\n")
    async with websockets.serve(handle_connection, "localhost", 8765):
        await asyncio.Future()  # runs forever until you stop it with Ctrl+C


if __name__ == "__main__":
    asyncio.run(main())
